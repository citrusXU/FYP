from sklearn import svm
import numpy as np
from feat_reader import read_test_feat
import os
from datetime import datetime
import argparse


def create_namelabel_map(namelist, label_gallery, label_query):
    namelabel_map = {}
    tmp = label_gallery.shape[0]
    for i in range(tmp):
        name = namelist[i]
        namelabel_map[name] = label_gallery[i]
    tmp2 = label_query.shape[0]
    for i in range(tmp2):
        name = namelist[i+tmp]
        namelabel_map[name] = label_query[i]
    return namelabel_map


def main(args):
    feat_file = args.feat_file
    meta_file = args.meta_file
    split_type = args.split_type
    clear_only = args.clear_only

    feat_gallery, label_gallery, \
        feat_query, label_query, namelist = read_test_feat(
            feat_file, meta_file, split_type=split_type, clear_only=clear_only
        )
    num_gallery = label_gallery.shape[0]
    num_query = label_query.shape[0]
    namegt_map = create_namelabel_map(namelist, label_gallery, label_query)
    namelist_gallery = namelist[:num_gallery]
    namelist_query = namelist[num_gallery:]

    acc = 0
    affinity_mat = feat_gallery.dot(feat_query.T)
    result_query = label_gallery[np.argmax(affinity_mat, axis=0)]
    acc += (result_query == label_query).sum()
    affinity_mat = feat_query.dot(feat_gallery.T)
    result_gallery = label_query[np.argmax(affinity_mat, axis=0)]
    acc += (result_gallery == label_gallery).sum()
    acc /= float(num_query+num_gallery)

    print 'Testing Accuracy on split split_type: {:.4f}'.format(acc)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('feat_file', type=str)
    parser.add_argument('meta_file', type=str)
    parser.add_argument('--split_type', type=str, default='original')
    parser.add_argument('--clear_only', action='store_true')
    args = parser.parse_args()

    main(args)
