from __future__ import print_function, absolute_import
import argparse
import os.path as osp
import os

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tensorboard import SummaryWriter

from prcg.dataset import get_dataset
from prcg.utils import save_checkpoint, load_checkpoint
from prcg.utils import Logger
from prcg.utils import transforms
from prcg.utils import FeatPreprocessor4b
from prcg.models import RANet
from prcg.loss import OIM4bLoss
from prcg.trainer import RATrainer
from prcg.evaluator import RAEvaluator


def merge_dicts(x, y):
    z = x.copy()
    z.update(y)
    return z


def get_data(
        dataset_name, data_dir, data_type,
        batch_size, workers, combine_trainval=False,
        head_feat_dir=None, face_feat_dir=None, body_feat_dir=None, upperbody_feat_dir=None):

    # read dataset
    dataset = get_dataset(dataset_name, data_dir, data_type)

    head_y = torch.load(os.path.join(head_feat_dir, 'feature_label_all.pth.tar'))
    head_x = torch.load(os.path.join(head_feat_dir, 'feature_label_all_noem.pth.tar'))
    dim_featx = int(head_x['train_feature'].values()[0].size()[0])
    dim_featy = int(head_y['train_feature'].values()[0].size()[0])
    face_y = torch.load(os.path.join(face_feat_dir, 'feature_label_all.pth.tar'))
    face_x = torch.load(os.path.join(face_feat_dir, 'feature_label_all_noem.pth.tar'))
    body_y = torch.load(os.path.join(body_feat_dir, 'feature_label_all.pth.tar'))
    body_x = torch.load(os.path.join(body_feat_dir, 'feature_label_all_noem.pth.tar'))
    upperbody_y = torch.load(os.path.join(upperbody_feat_dir, 'feature_label_all.pth.tar'))
    upperbody_x = torch.load(os.path.join(upperbody_feat_dir, 'feature_label_all_noem.pth.tar'))

    if combine_trainval:
        num_classes = dataset.num_trainval_ids
        train_processor = FeatPreprocessor4b(
            dataset.trainval,
            head_featx=merge_dicts(head_x['train_feature'], head_x['val_feature']),
            head_featy=merge_dicts(head_y['train_feature'], head_y['val_feature']),
            face_featx=merge_dicts(face_x['train_feature'], face_x['val_feature']),
            face_featy=merge_dicts(face_y['train_feature'], face_y['val_feature']),
            body_featx=merge_dicts(body_x['train_feature'], body_x['val_feature']),
            body_featy=merge_dicts(body_y['train_feature'], body_y['val_feature']),
            upperbody_featx=merge_dicts(upperbody_x['train_feature'], upperbody_x['val_feature']),
            upperbody_featy=merge_dicts(upperbody_y['train_feature'], upperbody_y['val_feature']))
    else:
        num_classes = dataset.num_train_ids
        train_processor = FeatPreprocessor4b(
            dataset.train,
            head_featx=head_x['train_feature'], head_featy=head_y['train_feature'],
            face_featx=face_x['train_feature'], face_featy=face_y['train_feature'],
            body_featx=body_x['train_feature'], body_featy=body_y['train_feature'],
            upperbody_featx=upperbody_x['train_feature'], upperbody_featy=upperbody_y['train_feature'])

    train_loader = DataLoader(
        train_processor, batch_size=batch_size, num_workers=workers,
        shuffle=True, pin_memory=True)

    val_loader = DataLoader(
        FeatPreprocessor4b(
            dataset.val,
            head_featx=head_x['val_feature'], head_featy=head_y['val_feature'],
            face_featx=face_x['val_feature'], face_featy=face_y['val_feature'],
            body_featx=body_x['val_feature'], body_featy=body_y['val_feature'],
            upperbody_featx=upperbody_x['val_feature'], upperbody_featy=upperbody_y['val_feature']),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        FeatPreprocessor4b(
            list(set(dataset.query) | set(dataset.gallery)),
            head_featx=head_x['test_feature'], head_featy=head_y['test_feature'],
            face_featx=face_x['test_feature'], face_featy=face_y['test_feature'],
            body_featx=body_x['test_feature'], body_featy=body_y['test_feature'],
            upperbody_featx=upperbody_x['test_feature'], upperbody_featy=upperbody_y['test_feature']),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, dim_featx, dim_featy, train_loader, val_loader, test_loader


def main(args):
    writer = SummaryWriter(args.logs_dir)

    sys.stdout = Logger(osp.join(args.logs_dir, 'train_log.txt'))
    print(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.benchmark = True

    # Create data loaders
    data_dir = osp.join(args.data_dir, args.dataset)
    dataset, num_classes, dim_featx, dim_featy, train_loader, val_loader, test_loader = \
        get_data(
            args.dataset, data_dir, args.data_type,
            args.batch_size, args.workers, args.combine_trainval,
            head_feat_dir=args.head_feat_dir,
            face_feat_dir=args.face_feat_dir,
            body_feat_dir=args.body_feat_dir,
            upperbody_feat_dir=args.upperbody_feat_dir)

    # Create model
    model = RANet(4, num_features=dim_featx)
    # model = torch.nn.DataParallel(model).cuda()
    model = model.cuda()

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

    # Criterion
    criterion = OIM4bLoss(dim_featy, num_classes, scalar=args.oim_scalar, momentum=args.oim_momentum)
    criterion.init_lut(train_loader)
    criterion.cuda()

    # Optimizer
    if args.optimizer == 'sgd':
        param_groups = model.parameters()
        optimizer = torch.optim.SGD(param_groups, lr=args.lr,
                                    momentum=args.momentum,
                                    weight_decay=args.weight_decay,
                                    nesterov=True)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,
                                     weight_decay=args.weight_decay)
    else:
        raise ValueError("Cannot recognize optimizer type:", args.optimizer)

    # Evaluator and Trainer
    evaluator = RAEvaluator(model)
    trainer = RATrainer(model, criterion)

    # Schedule learning rate
    def adjust_lr(epoch):
        if args.optimizer == 'sgd':
            lr = args.lr * (0.1 ** (epoch // 20))
        elif args.optimizer == 'adam':
            lr = args.lr if epoch <= 50 else \
                args.lr * (0.01 ** (epoch - 50) / 30)
        else:
            raise ValueError("Cannot recognize optimizer type:", args.optimizer)
        for g in optimizer.param_groups:
            g['lr'] = lr * g.get('lr_mult', 1)

    # start training
    top1 = evaluator.evaluate(val_loader, print_summary=True)
    test_top1 = evaluator.test(test_loader, dataset.gallery, dataset.query, print_summary=True)
    for epoch in range(args.start_epoch, args.epochs):
        adjust_lr(epoch)
        loss, prec = trainer.train(epoch, train_loader, optimizer, print_freq=10)
        writer.add_scalar('Train loss', loss, epoch+1)
        writer.add_scalar('Train accuracy', prec, epoch+1)

        top1 = evaluator.evaluate(val_loader, print_summary=False)
        writer.add_scalar('Val accuracy', top1, epoch+1)
        test_top1 = evaluator.test(test_loader, dataset.gallery, dataset.query, print_summary=True)
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
    parser.add_argument('-j', '--workers', type=int, default=8)
    # data
    parser.add_argument('--data-type', type=str, default='all',
                        choices=['all'])
    parser.add_argument('--combine-trainval', action='store_true',
                        help="Use train and val sets together for training."
                             "Val set is still used for validation.")
    parser.add_argument('--head_feat_dir', type=str, default='./logs/head_exp1')
    parser.add_argument('--face_feat_dir', type=str, default='./logs/face_exp1')
    parser.add_argument('--body_feat_dir', type=str, default='./logs/body_exp1')
    parser.add_argument('--upperbody_feat_dir', type=str, default='./logs/upperbody_exp1')
    # model
    parser.add_argument('--dropout', type=float, default=0.5)
    # loss
    parser.add_argument('--loss', type=str, default='oim',
                        choices=['oim'])
    parser.add_argument('--oim-scalar', type=float, default=20)
    parser.add_argument('--oim-momentum', type=float, default=0.5)
    # optimizer
    parser.add_argument('--optimizer', type=str, default='sgd',
                        choices=['sgd', 'adam'])
    parser.add_argument('--lr', type=float, default=0.1)
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
