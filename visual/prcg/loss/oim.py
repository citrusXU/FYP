from __future__ import absolute_import

import torch
import torch.nn.functional as F
from torch import nn, autograd


class OIM(autograd.Function):
    def __init__(self, lut, momentum=0.5):
        super(OIM, self).__init__()
        self.lut = lut
        self.momentum = momentum

    def forward(self, features, targets):
        self.save_for_backward(features, targets)
        outputs = features.mm(self.lut.t())
        return outputs

    def backward(self, grad_outputs):
        features, targets = self.saved_tensors
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(self.lut)
        for x, y in zip(features, targets):
            self.lut[y] = self.momentum * self.lut[y] + (1.0-self.momentum) * x
            self.lut[y] /= self.lut[y].norm()
        return grad_inputs, None


def oim(features, targets, lut, momentum=0.5):
    return OIM(lut, momentum=momentum)(features, targets)


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut', torch.zeros(num_classes, num_features))

    def forward(self, features, targets):
        inputs = oim(features, targets, self.lut, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs


class OIM4b(autograd.Function):
    def __init__(self, lut_b1, lut_b2, lut_b3, lut_b4, momentum=0.5):
        super(OIM4b, self).__init__()
        self.lut_b1 = lut_b1
        self.lut_b2 = lut_b2
        self.lut_b3 = lut_b3
        self.lut_b4 = lut_b4
        self.momentum = momentum

    def forward(self, features, scores, targets, flags):
        self.save_for_backward(features, scores, targets, flags)
        # print inputs.size()
        features_b1, features_b2, features_b3, features_b4 = torch.split(features, 1, 1)
        features_b1 = features_b1.squeeze()
        features_b2 = features_b2.squeeze()
        features_b3 = features_b3.squeeze()
        features_b4 = features_b4.squeeze()
        outputs_b1 = features_b1.mm(self.lut_b1.t())
        outputs_b2 = features_b2.mm(self.lut_b2.t())
        outputs_b3 = features_b3.mm(self.lut_b3.t())
        outputs_b4 = features_b4.mm(self.lut_b4.t())
        outputs = outputs_b1 + outputs_b2 + outputs_b3 + outputs_b4
        # print('output', outputs.size())
        return outputs

    def backward(self, grad_outputs):
        features, scores, targets, flags = self.saved_tensors
        features_b1, features_b2, features_b3, features_b4 = torch.split(features, 1, 1)
        features_b1 = features_b1.squeeze()
        features_b2 = features_b2.squeeze()
        features_b3 = features_b3.squeeze()
        features_b4 = features_b4.squeeze()
        scores_b1, scores_b2, scores_b3, scores_b4 = torch.split(scores, 1, 1)
        flags_b1, flags_b2, flags_b3, flags_b4 = torch.split(flags, 1, 1)
        grad_inputs = None
        if self.needs_input_grad[0]:
            grad_inputs_b1 = grad_outputs.mm(self.lut_b1)
            grad_inputs_b2 = grad_outputs.mm(self.lut_b2)
            grad_inputs_b3 = grad_outputs.mm(self.lut_b3)
            grad_inputs_b4 = grad_outputs.mm(self.lut_b4)
            grad_inputs = torch.cat((grad_inputs_b1.unsqueeze(1), grad_inputs_b2.unsqueeze(1), grad_inputs_b3.unsqueeze(1), grad_inputs_b4.unsqueeze(1)), 1)
        for x, score, y, flag in zip(features_b1, scores_b1, targets, flags_b1):
            score = score.cpu()[0]
            flag = flag.cpu()[0]
            if flag > 0:
                self.lut_b1[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b1[y] + (1.0 - self.momentum) * x
                self.lut_b1[y] /= self.lut_b1[y].norm()
        for x, score, y, flag in zip(features_b2, scores_b2, targets, flags_b2):
            score = score.cpu()[0]
            flag = flag.cpu()[0]
            if flag > 0:
                self.lut_b2[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b2[y] + (1.0 - self.momentum) * x
                self.lut_b2[y] /= self.lut_b2[y].norm()
        for x, score, y, flag in zip(features_b3, scores_b3, targets, flags_b3):
            score = score.cpu()[0]
            flag = flag.cpu()[0]
            if flag > 0:
                self.lut_b3[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b3[y] + (1.0 - self.momentum) * x
                self.lut_b3[y] /= self.lut_b3[y].norm()
        for x, score, y, flag in zip(features_b4, scores_b4, targets, flags_b4):
            score = score.cpu()[0]
            flag = flag.cpu()[0]
            if flag > 0:
                self.lut_b4[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b4[y] + (1.0 - self.momentum) * x
                self.lut_b4[y] /= self.lut_b4[y].norm()
        # print('grad_input', grad_inputs.size())
        return grad_inputs, None, None, None

    # def forward(self, features, scores, targets):
    #     self.save_for_backward(features, scores, targets)
    #     # print inputs.size()
    #     features_b1, features_b2, features_b3, features_b4 = torch.split(features, 1, 1)
    #     features_b1 = features_b1.squeeze()
    #     features_b2 = features_b2.squeeze()
    #     features_b3 = features_b3.squeeze()
    #     features_b4 = features_b4.squeeze()
    #     outputs_b1 = features_b1.mm(self.lut_b1.t())
    #     outputs_b2 = features_b2.mm(self.lut_b2.t())
    #     outputs_b3 = features_b3.mm(self.lut_b3.t())
    #     outputs_b4 = features_b4.mm(self.lut_b4.t())
    #     outputs = outputs_b1 + outputs_b2 + outputs_b3 + outputs_b4
    #     # print('output', outputs.size())
    #     return outputs

    # def backward(self, grad_outputs):
    #     features, scores, targets = self.saved_tensors
    #     features_b1, features_b2, features_b3, features_b4 = torch.split(features, 1, 1)
    #     features_b1 = features_b1.squeeze()
    #     features_b2 = features_b2.squeeze()
    #     features_b3 = features_b3.squeeze()
    #     features_b4 = features_b4.squeeze()
    #     scores_b1, scores_b2, scores_b3, scores_b4 = torch.split(scores, 1, 1)
    #     grad_inputs = None
    #     if self.needs_input_grad[0]:
    #         grad_inputs_b1 = grad_outputs.mm(self.lut_b1)
    #         grad_inputs_b2 = grad_outputs.mm(self.lut_b2)
    #         grad_inputs_b3 = grad_outputs.mm(self.lut_b3)
    #         grad_inputs_b4 = grad_outputs.mm(self.lut_b4)
    #         grad_inputs = torch.cat((grad_inputs_b1.unsqueeze(1), grad_inputs_b2.unsqueeze(1), grad_inputs_b3.unsqueeze(1), grad_inputs_b4.unsqueeze(1)), 1)
    #     for x, score, y in zip(features_b1, scores_b1, targets):
    #         score = score.cpu()[0]
    #         self.lut_b1[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b1[y] + (1.0 - self.momentum) * x
    #         self.lut_b1[y] /= self.lut_b1[y].norm()
    #     for x, score, y in zip(features_b2, scores_b2, targets):
    #         score = score.cpu()[0]
    #         self.lut_b2[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b2[y] + (1.0 - self.momentum) * x
    #         self.lut_b2[y] /= self.lut_b2[y].norm()
    #     for x, score, y in zip(features_b3, scores_b3, targets):
    #         score = score.cpu()[0]
    #         self.lut_b3[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b3[y] + (1.0 - self.momentum) * x
    #         self.lut_b3[y] /= self.lut_b3[y].norm()
    #     for x, score, y in zip(features_b4, scores_b4, targets):
    #         score = score.cpu()[0]
    #         self.lut_b4[y] = (1.0 - (1.0 - self.momentum) * score) * self.lut_b4[y] + (1.0 - self.momentum) * x
    #         self.lut_b4[y] /= self.lut_b4[y].norm()
    #     # print('grad_input', grad_inputs.size())
    #     return grad_inputs, None, None


def oim4b(features, scores, targets, flags, lut_b1, lut_b2, lut_b3, lut_b4, momentum=0.5):
    return OIM4b(lut_b1, lut_b2, lut_b3, lut_b4, momentum=momentum)(features, scores, targets, flags)

# def oim4b(features, scores, targets, lut_b1, lut_b2, lut_b3, lut_b4, momentum=0.5):
#     return OIM4b(lut_b1, lut_b2, lut_b3, lut_b4, momentum=momentum)(features, scores, targets)


class OIM4bLoss(nn.Module):
    def __init__(self, num_features, num_classes, scalar=1.0, momentum=0.5,
                 weight=None, size_average=True):
        super(OIM4bLoss, self).__init__()
        self.num_features = num_features
        self.num_classes = num_classes
        # print('num_class', num_classes)
        self.momentum = momentum
        self.scalar = scalar
        self.weight = weight
        self.size_average = size_average

        self.register_buffer('lut_b1', torch.zeros(num_classes, num_features))
        self.register_buffer('lut_b2', torch.zeros(num_classes, num_features))
        self.register_buffer('lut_b3', torch.zeros(num_classes, num_features))
        self.register_buffer('lut_b4', torch.zeros(num_classes, num_features))

    def init_lut(self, data_loader):
        for i, inputs in enumerate(data_loader):
            _, feat_ys, _, pids, flags = inputs
            for feat_y, pid, flag in zip(feat_ys, pids, flags):
                if flag[0]:
                    self.lut_b1[pid] += feat_y[0]
                if flag[1]:
                    self.lut_b2[pid] += feat_y[1]
                if flag[2]:
                    self.lut_b3[pid] += feat_y[2]
                if flag[3]:
                    self.lut_b4[pid] += feat_y[3]
        for i in range(self.num_classes):
            if self.lut_b1[i].norm() != 0:
                self.lut_b1[i] /= self.lut_b1[i].norm()
            if self.lut_b2[i].norm() != 0:
                self.lut_b2[i] /= self.lut_b2[i].norm()
            if self.lut_b3[i].norm() != 0:
                self.lut_b3[i] /= self.lut_b3[i].norm()
            if self.lut_b4[i].norm() != 0:
                self.lut_b4[i] /= self.lut_b4[i].norm()

    # def forward(self, features, scores, targets):
    #     inputs = oim4b(features, scores, targets, self.lut_b1, self.lut_b2, self.lut_b3, self.lut_b4, momentum=self.momentum)
    #     inputs *= self.scalar
    #     loss = F.cross_entropy(inputs, targets, weight=self.weight,
    #                            size_average=self.size_average)
    #     return loss, inputs
    def forward(self, features, scores, targets, flags):
        inputs = oim4b(features, scores, targets, flags, self.lut_b1, self.lut_b2, self.lut_b3, self.lut_b4, momentum=self.momentum)
        inputs *= self.scalar
        loss = F.cross_entropy(inputs, targets, weight=self.weight,
                               size_average=self.size_average)
        return loss, inputs
