import os
import sys
import cv2
import numpy as np

category = input('category: ')
data_type = input('data_type: ')

root_path = '../data/CorrData/imagepart/'
image_path = os.path.join(root_path, category, 'images', data_type)
anno_path = os.path.join(root_path, category, 'annotations', data_type)
save_img_path = os.path.join(root_path, category, 'crop_images', data_type)
save_anno_path = os.path.join(root_path, category, 'crop_annotations', data_type)
visual_path = os.path.join(root_path, category, 'visual', data_type)

if not os.path.exists(save_img_path):
    os.makedirs(save_img_path)
if not os.path.exists(save_anno_path):
    os.makedirs(save_anno_path)
if not os.path.exists(visual_path):
    os.makedirs(visual_path)

out_size = 512
for image_name in os.listdir(image_path):
    image_fn = os.path.join(image_path, image_name)
    anno_fn = os.path.join(anno_path, image_name.replace('.JPEG', '.png'))
    image = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    anno = cv2.imread(anno_fn, cv2.IMREAD_UNCHANGED)
    # print('anno: ', anno.min(), anno.max())
    vis_anno = anno * 255 / 3
    vis_anno = vis_anno.astype(np.uint8)
    cv2.imwrite(os.path.join(visual_path, f'anno_{image_name.split(".")[0]}.png'), vis_anno)

    # compute center and scale
    image_h, image_w = image.shape[:2]
    horizontal = np.min(anno, axis=0)
    vertical = np.min(anno, axis=1)
    up = min(np.argwhere(vertical < 3))
    bottom = max(np.argwhere(vertical < 3))
    left = min(np.argwhere(horizontal < 3))
    right = max(np.argwhere(horizontal < 3))
    # print('image_h: ', image_h, 'image_w: ', image_w)
    # print('up: ', up, 'bottom: ', bottom, 'left: ', left, 'right: ', right)
    obj_center = np.array([(up + bottom) // 2, (left + right) // 2])
    obj_size = max(right - left, bottom - up) + 40
    # print('obj_size: ', obj_size, 'obj_center: ', obj_center)
    scale = out_size / obj_size

    # resize and crop
    resize = int(image_h * scale), int(image_w * scale)
    image = cv2.resize(image, (resize[1], resize[0]))
    anno = cv2.resize(anno, resize, interpolation=cv2.INTER_NEAREST)
    out_obj_center = obj_center * scale
    out_obj_center = out_obj_center.astype(np.int32)
    out_left = int(out_obj_center[1] - out_size // 2)
    out_right = int(out_obj_center[1] + out_size // 2)
    out_up = int(out_obj_center[0] - out_size // 2)
    out_bottom = int(out_obj_center[0] + out_size // 2)
    print('111')
    print('out_left: ', out_left, 'out_right: ', out_right, 'out_up: ', out_up, 'out_bottom: ', out_bottom)

    if out_right >= resize[1]:
        image = cv2.copyMakeBorder(image, 0, 0, 0, out_right - resize[1], cv2.BORDER_CONSTANT)
        anno = cv2.copyMakeBorder(anno, 0, 0, 0, out_right - resize[1], cv2.BORDER_CONSTANT, value=3)
    if out_left < 0:
        # or out_right > resize[1] or out_up < 0 or out_bottom > resize[0]:
        image = cv2.copyMakeBorder(image, 0, 0, -out_left, 0, cv2.BORDER_CONSTANT)
        anno = cv2.copyMakeBorder(anno, 0, 0, -out_left, 0, cv2.BORDER_CONSTANT, value=3)
        out_right -= out_left
        out_left -= out_left
    if out_bottom >= resize[0]:
        image = cv2.copyMakeBorder(image, 0, out_bottom - resize[0], 0, 0, cv2.BORDER_CONSTANT)
        anno = cv2.copyMakeBorder(anno, 0, out_bottom - resize[0], 0, 0, cv2.BORDER_CONSTANT, value=3)
    if out_up < 0:
        image = cv2.copyMakeBorder(image, -out_up, 0, 0, 0, cv2.BORDER_CONSTANT)
        anno = cv2.copyMakeBorder(anno, -out_up, 0, 0, 0, cv2.BORDER_CONSTANT, value=3)
        out_bottom -= out_up
        out_up -= out_up

    print('out_left: ', out_left, 'out_right: ', out_right, 'out_up: ', out_up, 'out_bottom: ', out_bottom)
    image = image[out_up: out_bottom + 1, out_left: out_right + 1, :]
    anno = anno[out_up: out_bottom + 1, out_left: out_right + 1]

    image = cv2.resize(image, (out_size, out_size))
    anno = cv2.resize(anno, (out_size, out_size))
    cv2.imwrite(os.path.join(save_img_path, f'{image_name.split(".")[0]}.png'), image)
    cv2.imwrite(os.path.join(save_anno_path, f'{image_name.split(".")[0]}.png'), anno)
