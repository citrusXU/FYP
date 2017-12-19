from __future__ import print_function
import os.path as osp

import numpy as np

from .serialization import read_json


class Dataset(object):
    def __init__(self, root, data_type, split_type):
        self.root = root
        self.split_type = split_type
        self.data_type = data_type
        self.train, self.val, self.trainval, self.test = [], [], [], []
        self.query, self.gallery = [], []
        self.num_train_ids, self.num_val_ids, self.num_trainval_ids = 0, 0, 0
        self.num_query_ids, self.num_gallery_ids = 0, 0

    def load(self):
        # read sample list
        sample_list = read_json(self.meta_file)

        # count identities
        pid_sets = self.get_pid_sets(sample_list)
        self.num_train_ids = len(pid_sets['train'])
        self.num_val_ids = len(pid_sets['val'])
        self.num_trainval_ids = len(pid_sets['trainval'])
        self.num_query_ids = len(pid_sets['query'])
        self.num_gallery_ids = len(pid_sets['gallery'])

        # create id->label map
        pid_maps = self.get_pid_maps(pid_sets)

        # create data
        for sample in sample_list:
            gid = sample['group_id']
            img_id = sample['img_id']
            pid = sample['person_id']
            location = sample['location']
            fname_prefix = '{:s}_{:s}_{:s}_{:s}'.format(gid, img_id, self.bbox2str(location), pid)
            subset = sample['subset']
            region_flag = sample['region_flag']
            face_flag = region_flag['face']
            head_flag = region_flag['head']
            upperbody_flag = region_flag['upperbody']
            body_flag = region_flag['body']
            label = pid_maps[subset][pid]
            if not self.check_valid(region_flag):
                continue
            if subset == 'train':
                self.train.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
                self.trainval.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
            elif subset == 'val':
                self.val.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
                self.trainval.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
            else:
                split = sample['split'][self.split_type]
                if split == 0:
                    self.query.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
                elif split == 1:
                    self.gallery.append((fname_prefix, label, face_flag, head_flag, upperbody_flag, body_flag))
        self.test = list(set(self.query) | set(self.gallery))

    def check_valid(self, region_flag):
        if self.data_type == 'all':
            return True
        elif region_flag[self.data_type]:
            return True
        else:
            return False

    def bbox2str(self, bbox):
        string = ''
        for x in bbox:
            string += str(x)+'-'
        string = string[:-1]
        return string

    def get_pid_sets(self, sample_list):
        """
        create identities set for each subset
        """
        pid_sets = {
            'train': set(),
            'val': set(),
            'trainval': set(),
            'test': set(),
            'gallery': set(),
            'query': set()
        }
        for sample in sample_list:
            pid = sample['person_id']
            subset = sample['subset']
            pid_sets[subset].add(pid)
            if subset == 'test':
                split = sample['split'][self.split_type]
                if split == 0:
                    pid_sets['gallery'].add(pid)
                elif split == 1:
                    pid_sets['query'].add(pid)
        pid_sets['trainval'] = pid_sets['train'].copy().union(pid_sets['val'].copy())
        return pid_sets

    def get_pid_maps(self, pid_sets):
        """
        create id->label map
        """
        pid_maps = {
            'train': {},
            'val': {},
            'trainval': {},
            'test': {}
        }
        for i, pid in enumerate(list(pid_sets['train'])):
            pid_maps['train'][pid] = i
        for i, pid in enumerate(list(pid_sets['val'])):
            pid_maps['val'][pid] = i
        for i, pid in enumerate(list(pid_sets['trainval'])):
            pid_maps['trainval'][pid] = i
        for i, pid in enumerate(list(pid_sets['test'])):
            pid_maps['test'][pid] = i
        return pid_maps

    def print_summary(self):
        print(self.__class__.__name__, "dataset loaded")
        print('     subset   | # ids | # samples')
        print('---------------------------------')
        print('    train     | {:5d} | {:8d}'.format(self.num_train_ids, len(self.train)))
        print('    val       | {:5d} | {:8d}'.format(self.num_val_ids, len(self.val)))
        print('    trainval  | {:5d} | {:8d}'.format(self.num_trainval_ids, len(self.trainval)))
        print('    query     | {:5d} | {:8d}'.format(self.num_query_ids, len(self.query)))
        print('    gallery   | {:5d} | {:8d}'.format(self.num_gallery_ids, len(self.gallery)))
