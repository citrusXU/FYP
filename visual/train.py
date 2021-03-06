from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard import SummaryWriter

from prcg.dataset import get_dataset
from prcg.utils import save_checkpoint, load_checkpoint
from prcg.utils import Logger, mkdir_if_missing
from prcg.utils import transforms
from prcg.utils import Preprocessor
from prcg.models import ResNet, VGGNet
from prcg.loss import OIMLoss
from prcg.trainer import Trainer
from prcg.evaluator import Evaluator


def get_data(
        dataset_name, data_dir, data_type,
        crop_w, crop_h, hw_ratio,
        batch_size, workers, combine_trainval):

    # read dataset
    dataset = get_dataset(dataset_name, data_dir, data_type)

    # data transforms
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.RandomSizedRectCrop(crop_h, crop_w, ratio=(hw_ratio*0.8, hw_ratio*1.2)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ])
    test_transformer = transforms.Compose([
        transforms.Resize(size=(crop_h, crop_w)),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    if combine_trainval:
        num_classes = dataset.num_trainval_ids
        train_loader = DataLoader(
            Preprocessor(dataset.trainval, dataset.images_dir, default_size=(crop_w, crop_h), data_type=data_type, transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)
    else:
        num_classes = dataset.num_train_ids
        train_loader = DataLoader(
            Preprocessor(dataset.train, dataset.images_dir, default_size=(crop_w, crop_h), data_type=data_type, transform=test_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True, drop_last=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, dataset.images_dir, default_size=(crop_w, crop_h), data_type=data_type, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(dataset.test, dataset.images_dir, default_size=(crop_w, crop_h), data_type=data_type, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):

    mkdir_if_missing(args.logs_dir)
    writer = SummaryWriter(args.logs_dir)

    sys.stdout = Logger(osp.join(args.logs_dir, 'train_log.txt'))
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # create data loaders
    data_dir = osp.join(args.data_dir, args.dataset)
    dataset, num_classes, train_loader, val_loader, test_loader = \
        get_data(
            args.dataset, data_dir, args.data_type,
            args.crop_w, args.crop_h, args.hw_ratio,
            args.batch_size, args.workers, args.combine_trainval)

    # create model
    model = ResNet(args.depth, pretrained=True, num_features=args.features, norm=True, dropout=args.dropout)
    # model = VGGNet(
    #     args.depth, with_bn=True, pretrained=True, num_features=args.features, norm=True, dropout=args.dropout,
    #     input_size=(args.crop_w, args.crop_h))
    if args.pretrained_model is not None:
        model.init_from_pretrained_model(args.pretrained_model)
    model = torch.nn.DataParallel(model).cuda()
    # model = model.cuda()

    # load from checkpoint
    if args.resume:
        checkpoint = load_checkpoint(args.resume)
        model.load_state_dict(checkpoint['state_dict'])
        args.start_epoch = checkpoint['epoch']
        best_top1 = checkpoint['best_top1']
        print("=> start epoch {}  best top1 {:.1%}"
              .format(args.start_epoch, best_top1))
    else:
        best_top1 = 0

    # criterion
    criterion = OIMLoss(model.module.num_features, num_classes, scalar=args.oim_scalar, momentum=args.oim_momentum)
    criterion.cuda()

    # optimizer
    if args.optimizer == 'sgd':
        param_groups = model.parameters()
        base_param_ids = set(map(id, model.module.base.parameters()))
        new_params = [p for p in model.parameters() if
                      id(p) not in base_param_ids]
        param_groups = [
            {'params': model.module.base.parameters(), 'lr_mult': 0.1},
            {'params': new_params, 'lr_mult': 1.0}]
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot recognize optimizer type:", args.optimizer)

    # create trainer and evaluator
    trainer = Trainer(model, criterion)
    evaluator = Evaluator(model)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 30))
        elif args.optimizer == 'adam':
            lr = args.lr if epoch <= 50 else \
                args.lr * (0.01 ** (epoch - 50) / 30)
        else:
            raise ValueError("Cannot recognize optimizer type:", args.optimizer)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # start training
    top1 = evaluator.evaluate(val_loader)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        loss, prec = trainer.train(epoch, train_loader, optimizer)
        writer.add_scalar('Train loss', loss, epoch+1)
        writer.add_scalar('Train accuracy', prec, epoch+1)

        top1 = evaluator.evaluate(val_loader)
        writer.add_scalar('Val accuracy', top1, epoch+1)

        test_top1 = evaluator.test(test_loader, dataset.gallery, dataset.query)
        writer.add_scalar('Test accuracy', test_top1, epoch+1)

        is_best = top1 > best_top1
        best_top1 = max(top1, best_top1)
        save_checkpoint({
            'state_dict': model.state_dict(),
            'epoch': epoch + 1,
            'best_top1': best_top1,
        }, is_best, fpath=osp.join(args.logs_dir, 'checkpoint.pth.tar'))

        print('\n * Finished epoch {:3d}  top1: {:5.1%}  best: {:5.1%}{}\n'.
              format(epoch, top1, best_top1, ' *' if is_best else ''))

    # final test
    print('Test with best model:')
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])
    evaluator.test(test_loader, dataset.gallery, dataset.query)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='pipa',
                        choices=['pipa', 'cim'])
    parser.add_argument('-b', '--batch-size', type=int, default=128)
    parser.add_argument('-j', '--workers', type=int, default=12)
    # data
    parser.add_argument('--data-type', type=str, default='face',
                        choices=['face', 'head', 'upperbody', 'body'])
    parser.add_argument('--crop_w', type=int, default=256)
    parser.add_argument('--crop_h', type=int, default=256)
    parser.add_argument('--hw_ratio', type=float, default=1)
    parser.add_argument('--combine-trainval', action='store_true',
                        help="Use train and val sets together for training."
                             "Val set is still used for validation.")
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101])
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--pretrained_model', type=str, default=None)
    # loss
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['oim', 'triplet'])
    parser.add_argument('--oim-scalar', type=float, default=20)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight-decay', type=float, default=5e-4)
    # training configs
    parser.add_argument('--resume', type=str, default='', metavar='PATH')
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--start-epoch', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))

    args = parser.parse_args()
    main(args)
