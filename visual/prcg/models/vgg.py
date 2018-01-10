from __future__ import absolute_import

import torch.nn.functional as F
import torch.nn.init as init
from torch import nn
from torch.autograd import Variable
import torch
from torchvision.models import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn
from torch.nn import Parameter


class VGGNet(nn.Module):
    __factory = {
        '11': vgg11,
        '13': vgg13,
        '16': vgg16,
        '19': vgg19,
        '11bn': vgg11,
        '13bn': vgg13,
        '16bn': vgg16,
        '19bn': vgg19
    }

    def __init__(
        self, depth, with_bn=True, pretrained=True,
            num_features=0, norm=False, embedding=True, dropout=0,
            input_size=(256, 256)):
        super(VGGNet, self).__init__()

        self.depth = depth
        self.with_bn = with_bn
        self.pretrained = pretrained
        self.embedding = embedding

        # Construct base (pretrained) InceptionNet
        if self.with_bn:
            self.base = VGGNet.__factory['{:d}bn'.format(depth)](pretrained=pretrained)
        else:
            self.base = VGGNet.__factory['{:d}'.format(depth)](pretrained=pretrained)

        self.num_features = num_features
        self.norm = norm
        self.dropout = dropout

        # out_planes = self.base.classifier[0].in_features
        out_planes = 512*(input_size[0]/32)*(input_size[1]/32)

        # Append new layers
        self.feat = nn.Linear(out_planes, self.num_features)
        self.feat_bn = nn.BatchNorm1d(self.num_features)
        init.kaiming_normal(self.feat.weight, mode='fan_out')
        init.constant(self.feat.bias, 0)
        init.constant(self.feat_bn.weight, 1)
        init.constant(self.feat_bn.bias, 0)

        if self.dropout > 0:
            self.drop = nn.Dropout(self.dropout)

        if not self.pretrained:
            self.reset_params()

    def forward(self, x):
        for name, module in self.base.features._modules.items():
            x = module(x)
        feature = x.view(x.size(0), -1)

        if self.embedding:
            feature = self.feat(feature)
            feature = self.feat_bn(feature)
            if self.norm:
                feature = feature / feature.norm(2, 1).unsqueeze(1).expand_as(feature)
            if self.dropout > 0:
                feature = self.drop(feature)
        return feature

    def reset_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant(m.weight, 1)
                init.constant(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant(m.bias, 0)

    def init_from_pretrained_model(self, pretrained_model):
        checkpoint = torch.load(pretrained_model)
        state_dict = checkpoint['state_dict']
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                raise KeyError('unexpected key "{}" in state_dict'
                               .format(name))
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, own_state[name].size(), param.size()))
                raise
        missing = set(own_state.keys()) - set(state_dict.keys())
        if len(missing) > 0:
            print('missing keys in state_dict: "{}"'.format(missing))


if __name__ == '__main__':
    model = VGGNet(16, with_bn=False, num_features=256, norm=True, input_size=(128, 256))
    x = Variable(torch.zeros(30, 3, 128, 256), requires_grad=False)
    feat = model(x)
    print(feat.data.size())
