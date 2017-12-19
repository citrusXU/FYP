import json
import numpy as np
import torch


def bbox2str(bbox):
        string = ''
        for x in bbox:
            string += str(x)+'-'
        string = string[:-1]
        return string


def read_test_feat(feat_file, meta_file, split_type='original', flip=0, clear_only=False):
    data = torch.load(feat_file)
    test_feat = data['test_feature']
    test_label = data['test_label']

    with open(meta_file, 'r') as f:
        sample_list = json.load(f)
    split_set = {}
    for sample in sample_list:
        gid = sample['group_id']
        img_id = sample['img_id']
        pid = sample['person_id']
        location = sample['location']
        key = '{:s}_{:s}_{:s}_{:s}'.format(gid, img_id, bbox2str(location), pid)
        value = sample['split'][split_type]
        if value is None:
            split_set[key] = -1
        else:
            split_set[key] = value

    feat_gallery = []
    feat_query = []
    label_gallery = []
    label_query = []

    namelist_gallery = []
    namelist_query = []
    for key, value in test_feat.items():
        feat = value.numpy()
        label = test_label[key]

        if split_set[key] < 0:
            split = -1
        else:
            split = (split_set[key]+flip) % 2

        if split == 1:
            feat_gallery.append(feat)
            label_gallery.append(label)
            namelist_gallery.append(key)
        elif split == 0:
            feat_query.append(feat)
            label_query.append(label)
            namelist_query.append(key)
        else:
            pass

    num_query_ids = len(set(label_query))
    num_gallery_ids = len(set(label_gallery))
    print("     subset   | # ids | # images")
    print("  ---------------------------")
    print(
        "    query     | {:5d} | {:8d}"
        .format(num_query_ids, len(label_query)))
    print(
        "    gallery   | {:5d} | {:8d}"
        .format(num_gallery_ids, len(label_gallery)))

    namelist = namelist_gallery + namelist_query
    return np.array(feat_gallery), np.array(label_gallery), \
        np.array(feat_query), np.array(label_query), \
        namelist
