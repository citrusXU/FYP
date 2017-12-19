from __future__ import print_function, absolute_import
import time
from collections import OrderedDict

import torch

from .utils import AverageMeter
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt


class Evaluator(object):
    def __init__(self, model):
        super(Evaluator, self).__init__()
        self.model = model

    def extract_feature(self, data_loader, print_summary=True):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        for i, (imgs, fnames, pids) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs = Variable(imgs).cuda()
            outputs = self.model(inputs)

            for fname, feat, pid in zip(fnames, outputs, pids):
                features[fname] = feat.cpu().data
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if print_summary:
                print(
                    'Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(
                        i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg))

        return features, labels

    def evaluate(self, data_loader, print_summary=True):

        features, labels = self.extract_feature(data_loader, print_summary)

        x_label = np.array(labels.values())
        y_label = np.array(labels.values())

        x = torch.cat([features[key].unsqueeze(0) for key in labels.keys()], 0)
        y = torch.cat([features[key].unsqueeze(0) for key in labels.keys()], 0)
        query_size = x.size(0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1).numpy()
        y = y.view(n, -1).numpy()
        affinitymat = x.dot(y.T)
        for i in range(query_size):
            affinitymat[i, i] = -99
        x_result = y_label[np.argmax(affinitymat, axis=1)]
        top1 = (x_result == x_label).sum() / float(x_label.shape[0])

        if print_summary:
            print(
                '\nquery_size: {:d}\t'
                'top1: {:.3f}\n'
                .format(
                    query_size, top1
                )
            )
        return top1

    def test(self, data_loader, query, gallery, print_summary=True):
        features, labels = self.extract_feature(data_loader, print_summary)

        query_size = 0
        top1 = 0

        x_label = np.array([labels[values[0]] for values in query])
        y_label = np.array([labels[values[0]] for values in gallery])

        x = torch.cat([features[values[0]].unsqueeze(0) for values in query], 0)
        y = torch.cat([features[values[0]].unsqueeze(0) for values in gallery], 0)
        query_size += x.size(0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1).numpy()
        y = y.view(n, -1).numpy()
        affinitymat = x.dot(y.T)
        x_result = y_label[np.argmax(affinitymat, axis=1)]
        top1 += (x_result == x_label).sum() / float(x_label.shape[0])

        if print_summary:
            print(
                '\nquery_size: {:d}\t'
                'top1: {:.3f}\n'
                .format(
                    query_size, top1
                )
            )
        return top1


class RAEvaluator(object):
    def __init__(self, model):
        super(RAEvaluator, self).__init__()
        self.model = model

    def extract_feature(self, data_loader, print_summary=False):
        self.model.eval()
        batch_time = AverageMeter()
        data_time = AverageMeter()

        features = OrderedDict()
        attentions = OrderedDict()
        labels = OrderedDict()

        end = time.time()
        for i, (feat_x, feat_y, fnames, pids, _) in enumerate(data_loader):
            data_time.update(time.time() - end)

            inputs_x = Variable(feat_x).cuda()
            inputs_y = Variable(feat_y).cuda()
            outputs, scores = self.model(inputs_x, inputs_y)

            for fname, feat, score, pid in zip(fnames, outputs, scores, pids):
                features[fname] = feat.cpu().data
                attentions[fname] = score.cpu().data
                labels[fname] = pid

            batch_time.update(time.time() - end)
            end = time.time()

            if print_summary:
                print(
                    'Extract Features: [{}/{}]\t'
                    'Time {:.3f} ({:.3f})\t'
                    'Data {:.3f} ({:.3f})\t'
                    .format(
                        i + 1, len(data_loader),
                        batch_time.val, batch_time.avg,
                        data_time.val, data_time.avg))

        return features, attentions, labels

    def evaluate(self, data_loader, print_summary=True):
        features, attentions, labels = self.extract_feature(data_loader)
        weights = np.array([attentions[key].numpy() for key in labels.keys()])
        mean_weight = weights.mean(axis=0)
        x_label = np.array(labels.values())
        y_label = np.array(labels.values())
        x = torch.cat([features[key].unsqueeze(0) for key in labels.keys()], 0)
        y = torch.cat([features[key].unsqueeze(0) for key in labels.keys()], 0)
        query_size = x.size(0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1).numpy()
        y = y.view(n, -1).numpy()
        affinitymat = x.dot(y.T)
        for i in range(query_size):
            affinitymat[i, i] = -99
        x_result = y_label[np.argmax(affinitymat, axis=1)]
        top1 = (x_result == x_label).sum() / float(x_label.shape[0])

        if print_summary:
            print(
                '\nquery_size: {:d}\t'
                'mean weights: {:.4f}  {:.4f}  {:.4f} {:.4f}\t'
                'top1: {:.3f}\n'
                .format(
                    m, mean_weight[0], mean_weight[1], mean_weight[2], mean_weight[3], top1
                )
            )
        return top1

    def test(self, data_loader, query, gallery, print_summary=True):
        features, attentions, labels = self.extract_feature(data_loader)
        weights = np.array([attentions[values[0]].numpy() for values in query])
        mean_weight = weights.mean(axis=0)
        x_label = np.array([labels[values[0]] for values in query])
        y_label = np.array([labels[values[0]] for values in gallery])
        x = torch.cat([features[values[0]].unsqueeze(0) for values in query], 0)
        y = torch.cat([features[values[0]].unsqueeze(0) for values in gallery], 0)
        m, n = x.size(0), y.size(0)
        x = x.view(m, -1).numpy()
        y = y.view(n, -1).numpy()
        dismat = x.dot(y.T)
        x_result = y_label[np.argmax(dismat, axis=1)]
        top1 = (x_result == x_label).sum() / float(x_label.shape[0])

        if print_summary:
            print(
                '\nquery_size: {:d}\t'
                'mean weights: {:.4f}  {:.4f}  {:.4f} {:.4f}\t'
                'top1: {:.3f}\n'
                .format(
                    m, mean_weight[0], mean_weight[1], mean_weight[2], mean_weight[3], top1
                )
            )
        return top1
