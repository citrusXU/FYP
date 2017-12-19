from __future__ import print_function, absolute_import
import os.path as osp

from ..utils.dataset import Dataset


class PIPA(Dataset):

    def __init__(self, root, data_type='all', split_type='original'):
        super(PIPA, self).__init__(root, data_type, split_type)
        self.meta_file = osp.join(self.root, 'meta.json')
        self.load()
        self.print_summary()

    @property
    def images_dir(self):
        return osp.join(self.root, 'crop_image')
