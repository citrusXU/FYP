from __future__ import print_function, absolute_import
import time

import torch
from torch.autograd import Variable

from .utils import get_accuracy
from .loss import OIMLoss
from .utils import AverageMeter


class BaseTrainer(object):
    def __init__(self, model, criterion):
        super(BaseTrainer, self).__init__()
        self.model = model
        self.criterion = criterion

    def train(self, epoch, data_loader, optimizer, print_freq=1):
        self.model.train()

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        precisions = AverageMeter()

        end = time.time()
        iter_count = len(data_loader)
        for i, inputs in enumerate(data_loader):
            data_time.update(time.time() - end)

            loss, prec1 = self._forward(*self._parse_data(inputs))

            losses.update(loss.data[0])
            precisions.update(prec1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})\t'
                      'Prec {:.2%} ({:.2%})\t'
                      .format(epoch, i + 1, iter_count,
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg,
                              precisions.val, precisions.avg))

        return losses.avg, precisions.avg

    def _parse_data(self, inputs):
        raise NotImplementedError

    def _forward(self, inputs, targets):
        raise NotImplementedError


class Trainer(BaseTrainer):
    def _parse_data(self, inputs):
        imgs, _, pids = inputs
        img_inputs = Variable(imgs).cuda()
        targets = Variable(pids.cuda())
        return img_inputs, targets

    def _forward(self, inputs, targets):
        outputs = self.model(inputs)
        loss, outputs = self.criterion(outputs, targets)
        prec, = get_accuracy(outputs.data, targets.data)
        prec = prec[0]
        return loss, prec


class RATrainer(BaseTrainer):
    def _parse_data(self, inputs):
        feat_x, feat_y, _, pids, flags = inputs
        input_x = Variable(feat_x).cuda()
        input_y = Variable(feat_y).cuda()
        targets = Variable(pids.cuda())
        flags = Variable(flags.cuda())
        return input_x, input_y, targets, flags

    def _forward(self, input_x, input_y, targets, flags):
        outputs, scores = self.model(input_x, input_y)
        loss, outputs = self.criterion(outputs, scores, targets, flags)
        prec, = get_accuracy(outputs.data, targets.data)
        prec = prec[0]
        return loss, prec
