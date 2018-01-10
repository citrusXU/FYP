from __future__ import absolute_import
import os.path as osp

from PIL import Image
import torch
import os
import numpy as np


class Preprocessor(object):
    def __init__(self, dataset, images_dir, default_size, data_type='face', transform=None):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.images_dir = images_dir
        self.data_type = data_type
        self.transform = transform
        self.default_size = default_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname_prefix, pid, _, _, _, _ = self.dataset[index]
        fpath = osp.join(self.images_dir, fname_prefix + '_{:s}.jpg'.format(self.data_type))
        if os.path.isfile(fpath):
            img = Image.open(fpath).convert('RGB')
        else:
            img = Image.new('RGB', self.default_size)
        if self.transform is not None:
            img = self.transform(img)
        return img, fname_prefix, pid


class FeatPreprocessor4b(object):
    def __init__(
        self, dataset,
            head_featx, face_featx, body_featx, upperbody_featx,
            head_featy, face_featy, body_featy, upperbody_featy,
            default_xsize=256, default_ysize=256):
        super(FeatPreprocessor4b, self).__init__()
        self.dataset = dataset
        self.default_xsize = default_xsize
        self.default_ysize = default_ysize
        self.head_featx = head_featx
        self.face_featx = face_featx
        self.body_featx = body_featx
        self.upperbody_featx = upperbody_featx
        self.head_featy = head_featy
        self.face_featy = face_featy
        self.body_featy = body_featy
        self.upperbody_featy = upperbody_featy

        self.eps = 0.0000001

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        if isinstance(indices, (tuple, list)):
            return [self._get_single_item(index) for index in indices]
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname_prefix, pid, face_flag, head_flag, upperbody_flag, body_flag = self.dataset[index]
        featx_b1 = self.face_featx[fname_prefix]
        featy_b1 = self.face_featy[fname_prefix]
        featx_b2 = self.head_featx[fname_prefix]
        featy_b2 = self.head_featy[fname_prefix]
        featx_b3 = self.upperbody_featx[fname_prefix]
        featy_b3 = self.upperbody_featy[fname_prefix]
        featx_b4 = self.body_featx[fname_prefix]
        featy_b4 = self.body_featy[fname_prefix]
        # if face_flag:
        #     featx_b1 = self.face_featx[fname_prefix]
        #     featy_b1 = self.face_featy[fname_prefix]
        # else:
        #     featx_b1 = torch.ones(self.default_xsize)*self.eps
        #     featy_b1 = torch.ones(self.default_ysize)*self.eps
        # if head_flag:
        #     featx_b2 = self.head_featx[fname_prefix]
        #     featy_b2 = self.head_featy[fname_prefix]
        # else:
        #     featx_b2 = torch.ones(self.default_xsize)*self.eps
        #     featy_b2 = torch.ones(self.default_ysize)*self.eps
        # if upperbody_flag:
        #     featx_b3 = self.upperbody_featx[fname_prefix]
        #     featy_b3 = self.upperbody_featy[fname_prefix]
        # else:
        #     featx_b3 = torch.ones(self.default_xsize)*self.eps
        #     featy_b3 = torch.ones(self.default_ysize)*self.eps
        # if body_flag:
        #     featx_b4 = self.body_featx[fname_prefix]
        #     featy_b4 = self.body_featy[fname_prefix]
        # else:
        #     featx_b4 = torch.ones(self.default_xsize)*self.eps
        #     featy_b4 = torch.ones(self.default_ysize)*self.eps

        featx = torch.cat((featx_b1.unsqueeze(0), featx_b2.unsqueeze(0), featx_b3.unsqueeze(0), featx_b4.unsqueeze(0)), dim=0).view(-1, 4)
        featy = torch.cat((featy_b1.unsqueeze(0), featy_b2.unsqueeze(0), featy_b3.unsqueeze(0), featy_b4.unsqueeze(0)), dim=0)

        return featx, featy, fname_prefix, pid, torch.FloatTensor((face_flag, head_flag, upperbody_flag, body_flag))
