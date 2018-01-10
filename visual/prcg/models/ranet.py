from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torch.nn import Parameter


class RANet(nn.Module):
    def __init__(self, num_branch, num_features=2048, weighted_feat=True):
        super(RANet, self).__init__()

        self.num_branch = num_branch
        self.num_features = num_features
        self.weighted_feat = weighted_feat

        self.fc_in = self.num_branch*8
        self.conv_1 = nn.Conv1d(self.num_features, self.fc_in, self.num_branch, 1)
        self.bn_1 = nn.BatchNorm1d(self.fc_in)
        self.relu_1 = nn.ReLU(inplace=True)
        self.fc = nn.Linear(self.fc_in, self.num_branch)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, y):
        x = self.conv_1(x)
        x = self.bn_1(x)
        x = self.relu_1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        score = self.sigmoid(x)

        if self.weighted_feat:
            feature = y * score.unsqueeze(2).expand_as(y)
        else:
            feature = y

        return feature, score


if __name__ == '__main__':
    model = RANet(4, num_features=2048)
    x = Variable(torch.zeros(30, 2048, 4), requires_grad=False)
    y = Variable(torch.zeros(30, 4, 256), requires_grad=False)
    feat, score = model(x, y)
    print feat.data.size(), score.data.size()
