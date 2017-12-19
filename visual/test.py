from __future__ import print_function, absolute_import
import argparse
import os.path as osp

import numpy as np
import sys
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from prcg.dataset import get_dataset
from prcg.utils import save_checkpoint, load_checkpoint
from prcg.utils import Logger
from prcg.utils import transforms
from prcg.utils import Preprocessor
from prcg.models import ResNet
from prcg.loss import OIMLoss
from prcg.trainer import Trainer
from prcg.evaluator import Evaluator


def get_data(
        dataset_name, data_dir, data_type,
        crop_w, crop_h, hw_ratio,
        batch_size, workers, combine_trainval):

    # read dataset
    dataset = get_dataset(dataset_name, data_dir, 'all')

    # data transforms
    normalizer = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
    train_transformer = transforms.Compose([
        transforms.RandomSizedRectCrop(crop_w, crop_h, ratio=[hw_ratio*0.8, hw_ratio*1.2]),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalizer,
    ])
    test_transformer = transforms.Compose([
        transforms.RectScale(crop_w, crop_h),
        transforms.ToTensor(),
        normalizer,
    ])

    # data loaders
    if combine_trainval:
        num_classes = dataset.num_trainval_ids
        train_loader = DataLoader(
            Preprocessor(dataset.trainval, data_type=data_type, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True)
    else:
        num_classes = dataset.num_train_ids
        train_loader = DataLoader(
            Preprocessor(dataset.train, data_type=data_type, transform=train_transformer),
            batch_size=batch_size, num_workers=workers,
            shuffle=True, pin_memory=True)

    val_loader = DataLoader(
        Preprocessor(dataset.val, data_type=data_type, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    test_loader = DataLoader(
        Preprocessor(dataset.test, data_type=data_type, transform=test_transformer),
        batch_size=batch_size, num_workers=workers,
        shuffle=False, pin_memory=True)

    return dataset, num_classes, train_loader, val_loader, test_loader


def main(args):

    sys.stdout = Logger(osp.join(args.logs_dir, 'test_log.txt'))
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
    model = ResNet(args.depth, pretrained=True, num_features=args.features, norm=True, embedding=(not args.no_embedding))
    model = torch.nn.DataParallel(model).cuda()
    checkpoint = load_checkpoint(osp.join(args.logs_dir, 'model_best.pth.tar'))
    model.load_state_dict(checkpoint['state_dict'])

    # create and evaluator
    evaluator = Evaluator(model)

    # extract feature
    print('Extracting training feature...')
    train_feat, train_label = evaluator.extract_feature(train_loader, print_summary=False)
    print('Extracting validation feature...')
    val_feat, val_label = evaluator.extract_feature(val_loader, print_summary=False)
    print('Extracting testing feature...')
    test_feat, test_label = evaluator.extract_feature(test_loader, print_summary=False)
    feature_label = ({
        'train_feature': train_feat, 'train_label': train_label,
        'val_feature': val_feat, 'val_label': val_label,
        'test_feature': test_feat, 'test_label': test_label
    })
    torch.save(feature_label, osp.join(args.logs_dir, args.save_name))

    # testing
    print('Testing...')
    train_top1 = evaluator.evaluate(train_loader, print_summary=False)
    val_top1 = evaluator.evaluate(val_loader, print_summary=False)
    test_top1 = evaluator.test(test_loader, dataset.gallery, dataset.query, print_summary=False)
    print(
        'Training Top1: {:.3f}\n'
        'Validation Top1: {:.3f}\n'
        'Testing Top1: {:.3f}\n'
        .format(train_top1, val_top1, test_top1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', type=str, default='pipa',
                        choices=['pipa', 'cim'])
    parser.add_argument('-b', '--batch-size', type=int, default=256)
    parser.add_argument('-j', '--workers', type=int, default=8)
    # data
    parser.add_argument('--data-type', type=str, default='face',
                        choices=['face', 'head', 'upperbody', 'body'])
    parser.add_argument('--crop_w', type=int, default=256)
    parser.add_argument('--crop_h', type=int, default=256)
    parser.add_argument('--hw_ratio', type=int, default=1)
    parser.add_argument('--combine-trainval', action='store_true',
                        help="Use train and val sets together for training."
                             "Val set is still used for validation.")
    # model
    parser.add_argument('--depth', type=int, default=50,
                        choices=[18, 34, 50, 101])
    parser.add_argument('--features', type=int, default=256)
    parser.add_argument('--no_embedding', action='store_true')
    # training configs
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--print-freq', type=int, default=1)
    # misc
    working_dir = osp.dirname(osp.abspath(__file__))
    parser.add_argument('--data-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'data'))
    parser.add_argument('--logs-dir', type=str, metavar='PATH',
                        default=osp.join(working_dir, 'logs'))
    parser.add_argument('--save_name', type=str, metavar='PATH',
                        default='feature_label.pth.tar')

    args = parser.parse_args()
    main(args)
