from utils import *
import cv2
import json
import os
from datetime import datetime
from multiprocessing import Pool
import argparse


def crop_location(sample, img_dir, save_dir, dataset='cim', save_size=(256, 512)):
    gid = sample['group_id']
    img_id = sample['img_id']
    person_id = sample['person_id']
    location = sample['location']
    if location[2] - location[0] <= 0 or location[3] - location[1] <= 0:
        return False
    try:
        src_img_path = os.path.join(img_dir, '{:s}_{:s}.jpg'.format(gid, img_id))
        src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
        crop_img = safe_crop(src_img, location)
        crop_img = cv2.resize(crop_img, save_size)
        if dataset == 'cim':
            save_name = '{:s}_{:s}_{:s}_{:s}_body.jpg'.format(gid, img_id, bbox2str(location), person_id)
        elif dataset == 'pipa':
            save_name = '{:s}_{:s}_{:s}_{:s}_head.jpg'.format(gid, img_id, bbox2str(location), person_id)
        else:
            raise ValueError('No such dataset: {:s}'.format(dataset))
        save_path = os.path.join(save_dir, save_name)
        if not os.path.isfile(save_path):
            cv2.imwrite(save_path, crop_img)
    except Exception as e:
        print('Error! {:s} {:s}'.format(src_img_path, str(e)))
        return False
    return True


def crop_face(sample, img_dir, save_dir, save_size=(256, 256)):
    gid = sample['group_id']
    img_id = sample['img_id']
    person_id = sample['person_id']
    location = sample['location']
    save_name = '{:s}_{:s}_{:s}_{:s}_face.jpg'.format(gid, img_id, bbox2str(location), person_id)
    save_path = os.path.join(save_dir, save_name)

    face_landmarks = sample['face']
    if check_face(face_landmarks) == 0:
        return False
    else:
        try:
            src_img_path = os.path.join(img_dir, '{:s}_{:s}.jpg'.format(gid, img_id))
            src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
            face_landmarks = np.array(face_landmarks).astype(np.float)
            rotate_img = align_face(src_img, face_landmarks)
            crop_size = [110, 110]
            center_x = rotate_img.shape[0]/2 + 15
            center_y = rotate_img.shape[1]/2
            w = crop_size[0]
            h = crop_size[1]
            rotate_img = rotate_img[center_x-w/2:center_x+w/2, center_y-h/2:center_y+h/2, :]
            rotate_img = cv2.resize(rotate_img, save_size)
            cv2.imwrite(save_path, rotate_img)
        except Exception as e:
            print('Error! {:s} {:s}'.format(src_img_path, str(e)))
            return False
    return True


def crop_body(sample, img_dir, save_dir, save_size=(256, 512)):
    gid = sample['group_id']
    img_id = sample['img_id']
    person_id = sample['person_id']
    location = sample['location']
    save_name = '{:s}_{:s}_{:s}_{:s}_body.jpg'.format(gid, img_id, bbox2str(location), person_id)
    save_path = os.path.join(save_dir, save_name)

    pose = sample['pose']
    pose_conf = sample['pose_conf']

    if check_body(pose_conf) == 0:
        return False
    else:
        try:
            src_img_path = os.path.join(img_dir, '{:s}_{:s}.jpg'.format(gid, img_id))
            src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
            pose = np.array(pose).astype(np.float)
            rotate_img = align_body(src_img, pose)
            rotate_img = cv2.resize(rotate_img, save_size)
            cv2.imwrite(save_path, rotate_img)
        except Exception as e:
            print('Error! {:s} {:s}'.format(src_img_path, str(e)))
            return False
    return True


def crop_upperbody(sample, img_dir, save_dir, save_size=(256, 384)):
    gid = sample['group_id']
    img_id = sample['img_id']
    person_id = sample['person_id']
    location = sample['location']
    save_name = '{:s}_{:s}_{:s}_{:s}_upperbody.jpg'.format(gid, img_id, bbox2str(location), person_id)
    save_path = os.path.join(save_dir, save_name)

    pose = sample['pose']
    pose_conf = sample['pose_conf']

    upperbody_flag = check_upperbody(pose_conf)
    if upperbody_flag == 0:
        return False
    else:
        try:
            src_img_path = os.path.join(img_dir, '{:s}_{:s}.jpg'.format(gid, img_id))
            src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
            pose = np.array(pose).astype(np.float)
            if upperbody_flag == 1:
                rotate_img = align_upperbody1(src_img, pose)
            else:
                rotate_img = align_upperbody2(src_img, pose)
            rotate_img = cv2.resize(rotate_img, save_size)
            cv2.imwrite(save_path, rotate_img)
        except Exception as e:
            print('Error! {:s} {:s}'.format(src_img_path, str(e)))
            return False
    return True


def crop_head(sample, img_dir, save_dir, save_size=(256, 256)):
    gid = sample['group_id']
    img_id = sample['img_id']
    person_id = sample['person_id']
    location = sample['location']
    save_name = '{:s}_{:s}_{:s}_{:s}_head.jpg'.format(gid, img_id, bbox2str(location), person_id)
    save_path = os.path.join(save_dir, save_name)

    pose = sample['pose']
    pose_conf = sample['pose_conf']
    face_landmarks = sample['face']

    head_flag = check_head(face_landmarks, pose_conf)
    if head_flag == 0:
        return False
    else:
        try:
            src_img_path = os.path.join(img_dir, '{:s}_{:s}.jpg'.format(gid, img_id))
            src_img = cv2.imread(src_img_path, cv2.IMREAD_COLOR)
            pose = np.array(pose).astype(np.float)
            face_landmarks = np.array(face_landmarks).astype(np.float)
            if head_flag == 1:
                rotate_img = align_head1(src_img, face_landmarks)
            else:
                rotate_img = align_head2(src_img, pose)
            rotate_img = cv2.resize(rotate_img, save_size)
            cv2.imwrite(save_path, rotate_img)
        except Exception as e:
            print('Error! {:s} {:s}'.format(src_img_path, str(e)))
            return False
    return True


def crop_all(sample, img_dir, save_dir, dataset):
    flags = [False, False, False, False, False]
    flags[0] = crop_face(sample, img_dir, save_dir)
    flags[1] = crop_head(sample, img_dir, save_dir)
    flags[2] = crop_upperbody(sample, img_dir, save_dir)
    flags[3] = crop_body(sample, img_dir, save_dir)
    flags[4] = crop_location(sample, img_dir, save_dir, dataset)
    return flags


def call_back(rst):
    global count
    count[-1] += 1
    for i in range(5):
        count[i] += int(rst[i])
    if count[-1] % 5000 == 0:
        print(datetime.now())
        print('   #Total   | #Location  |   #Face    |   #Head    | #Upperbody |   #Body    |')
        print('------------------------------------------------------------------------------')
        print(
            ' {:10d} | {:10d} | {:10d} | {:10d} | {:10d} | {:10d} |'.format(
                count[-1], count[-2], count[0], count[1], count[2], count[3]))
        print('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='/mnt/SSD/person_data/')
    parser.add_argument('--dataset', type=str, default='cim', choices=['cim', 'pipa'])
    args = parser.parse_args()

    root = args.root
    dataset = args.dataset
    save_dir = os.path.join(root, dataset, 'crop_image')
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    img_dir = os.path.join(root, dataset, 'image')
    meta_file = os.path.join(root, dataset, 'meta_extra.json')
    with open(meta_file, 'r') as f:
        sample_list = json.load(f)

    print('{:s} \t Croping images on {:s}. Totally {:d} samples.\n'.format(str(datetime.now()), dataset, len(sample_list)))
    global count
    count = [0, 0, 0, 0, 0, 0]
    pool = Pool(40)
    for sample in sample_list:
        pool.apply_async(crop_all, args=(sample, img_dir, save_dir, dataset), callback=call_back)
    pool.close()
    pool.join()

    print('{:s} \t All done!'.format(str(datetime.now())))
    print('   #Total   | #Location  |   #Face    |   #Head    | #Upperbody |   #Body    |')
    print('------------------------------------------------------------------------------')
    print(
        ' {:10d} | {:10d} | {:10d} | {:10d} | {:10d} | {:10d} |'.format(
            count[-1], count[-2], count[0], count[1], count[2], count[3]))
    print('\n')
