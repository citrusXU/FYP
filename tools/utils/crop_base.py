import numpy as np
import cv2


def bbox2str(bbox):
    string = ''
    for x in bbox:
        string += str(x)+'-'
    string = string[:-1]
    return string


def safe_crop(img, bbox):
    h = img.shape[0]
    w = img.shape[1]
    l = max(bbox[0], 0)
    r = min(bbox[2], w)
    t = max(bbox[1], 0)
    b = min(bbox[3], h)
    crop_img = img[t:b, l:r, :]
    return crop_img


def get_similarity_matrix(src_pts, dst_pts):
        """
        get a similarity matrix (totation, scale and translation)
        """
        num = src_pts.shape[0]
        A = np.zeros([num*2, 4])
        b = np.zeros([num*2])
        for i in range(num):
            A[2*i, 0] = src_pts[i, 0]
            A[2*i, 1] = - src_pts[i, 1]
            A[2*i, 2] = 1
            A[2*i + 1, 0] = src_pts[i, 1]
            A[2*i + 1, 1] = src_pts[i, 0]
            A[2*i + 1, 3] = 1
            b[2*i] = dst_pts[i, 0]
            b[2*i + 1] = dst_pts[i, 1]
        # c = np.linalg.pinv(A.T.dot(A)).dot(A.T).dot(b)
        c = np.linalg.pinv(A).dot(b)
        T = np.zeros([2, 3])
        T[0, 0] = c[0]
        T[0, 1] = -c[1]
        T[0, 2] = c[2]
        T[1, 0] = c[1]
        T[1, 1] = c[0]
        T[1, 2] = c[3]
        return T


def check_face(face_landmarks):
    """
    if the face region is valid: 0 invalud, 1 valid
    """
    if face_landmarks is not None:
        return 1
    else:
        return 0


def check_body(pose_conf):
    """
    if the body region is valid: 0 invalud, 1 valid
    """
    if pose_conf is None:
        return 0
    elif pose_conf[1] > 0 and pose_conf[8] > 0 and pose_conf[11] > 0:
        return 1
    else:
        return 0


def check_upperbody(pose_conf):
    """
    if the upperbody region is valid: 0 invalud, 1 valid (ref align_upperbody1), 2 valid (ref align_upperbody2)
    """
    if pose_conf is None:
        return 0
    elif pose_conf[1] > 0 and pose_conf[8] > 0 and pose_conf[11] > 0:
        return 1
    elif pose_conf[1] > 0 and pose_conf[2] > 0 and pose_conf[5] > 0:
        return 2
    else:
        return 0


def check_head(face_landmarks, pose_conf):
    """
    if the head region is valid: 0 invalud, 1 valid (ref align_head1), 2 valid (ref align_head2)
    """
    if face_landmarks is not None:
        return 1
    elif pose_conf is None:
        return 0
    elif pose_conf[1] > 0 and pose_conf[16] > 0 and pose_conf[17] > 0:
        return 2
    else:
        return 0


def align_face(image, face_landmarks, crop_size=(178, 218)):
    """
    algin face by: left eye, right eye, mouse
    """
    mean_pose = np.array(
        [(70.7450, 112.0000),
            (108.2370, 112.0000),
            (89.4324, 153.5140)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    src = np.array(
        [[face_landmarks[0, 0], face_landmarks[0, 1]],
            [face_landmarks[1, 0], face_landmarks[1, 1]],
            [(face_landmarks[4, 0]+face_landmarks[3, 0])/2,
                (face_landmarks[4, 1]+face_landmarks[3, 1])/2]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate


def align_body(image, pose, crop_size=(256, 512)):
    """
    align body by: neck, left hip, right hip
    """
    mean_pose = np.array(
        [(128.0, 150.0),
            (85.0, 340.0),
            (170.0, 340.0)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left = pose[8]
    right = pose[11]
    if left[0] > right[0]:
        left = pose[11]
        right = pose[8]
    src = np.array(
        [[pose[1, 0], pose[1, 1]],
            [left[0], left[1]],
            [right[0], right[1]]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate


def align_upperbody1(image, pose, crop_size=(256, 384)):
    """
    align upperbody by: neck, left hip, right hip
    """
    mean_pose = np.array(
        [(128.0, 150.0),
            (65.0, 340.0),
            (190.0, 340.0)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left = pose[8]
    right = pose[11]
    if left[0] > right[0]:
        left = pose[11]
        right = pose[8]
    src = np.array(
        [[pose[1, 0], pose[1, 1]],
            [left[0], left[1]],
            [right[0], right[1]]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate


def align_upperbody2(image, pose, crop_size=(256, 384)):
    """
    align upperbody by: neck, left shoulder, right shoulder
    """
    mean_pose = np.array(
        [(128.0, 150.0),
            (55.0, 170.0),
            (200.0, 170.0)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left = pose[2]
    right = pose[5]
    if left[0] > right[0]:
        left = pose[5]
        right = pose[2]
    src = np.array(
        [[pose[1, 0], pose[1, 1]],
            [left[0], left[1]],
            [right[0], right[1]]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate


def align_head1(image, face_landmarks, crop_size=(256, 256)):
    """
    for frontal face
    align head by: left eye, right eye, mouse
    """
    mean_pose = np.array(
        [(82.0, 138.0),
            (174.0, 138.0),
            (128.0, 180.0)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    src = np.array(
        [[face_landmarks[0, 0], face_landmarks[0, 1]],
            [face_landmarks[1, 0], face_landmarks[1, 1]],
            [(face_landmarks[4, 0]+face_landmarks[3, 0])/2,
                (face_landmarks[4, 1]+face_landmarks[3, 1])/2]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate


def align_head2(image, pose, crop_size=(256, 256)):
    """
    for non-frontal face
    align head by: left ear, right ear, neck
    """
    mean_pose = np.array(
        [(32.0, 148.0),
            (224.0, 148.0),
            (128.0, 260.0)],
        np.float32)
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    left = pose[16]
    right = pose[17]
    if left[0] > right[0]:
        left = pose[17]
        right = pose[16]
    src = np.array(
        [[left[0], left[1]],
            [right[0], right[1]],
            [pose[1, 0], pose[1, 1]]],
        np.float32)
    matrix = get_similarity_matrix(src, mean_pose)
    image_rotate = cv2.warpAffine(
        image, matrix, crop_size,
        borderMode=cv2.BORDER_REPLICATE)
    return image_rotate
