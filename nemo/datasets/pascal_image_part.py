import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

from nemo.utils import get_abs_path

class PosePredLoader():
    def __init__(self, cate, base_path='eval/pose_estimation_3d_nemo_%s'):
        base_path = base_path % cate
        self.preds = torch.load(os.path.join(base_path, 'pascal3d_occ0_%s_val.pth' % cate))
        self.kkeys = {t.split('/')[1]:t for t in self.preds.keys()}

    def __getitem__(self, key):
        if key in self.kkeys:
            return self.preds[self.kkeys[key]]['final'][0]
        else:
            return None

class PascalImagePart(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        root_path = os.path.join(root_path, 'pascalimagepart')
        self.root_path = get_abs_path(root_path)

        data_path = os.path.join(root_path, 'images', category)
        anno_path = os.path.join(root_path, 'annotations', category)
        pose_path = 'eval/pose_estimation_3d_nemo_%s' % category
        pose_path_train = 'eval1/pose_estimation_3d_nemo_%s_training' % category
        self.preds = torch.load(os.path.join(pose_path, 'pascal3d_occ0_%s_val.pth' % category))
        # self.preds.update(torch.load(os.path.join(pose_path_train, 'pascal3d_occ0_%s_val.pth' % category)))
        self.kkeys = {t.split('/')[1]: t for t in self.preds.keys()}
        print('len: ', len(self.kkeys))
        # print(self.kkeys)

        self.save_path = get_abs_path(data_path)

        self.images = []
        self.segs = []
        self.annos = []
        self.names = []
        for image_name in os.listdir(data_path):
            if 'seg' in image_name:
                continue
            image_fn = os.path.join(data_path, image_name)
            seg_fn = os.path.join(data_path, image_name.replace('.JPEG', '_seg.png'))
            image = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            seg = cv2.imread(seg_fn, cv2.IMREAD_UNCHANGED)
            anno_fn = os.path.join(anno_path, image_name.replace('.JPEG', '.npz'))
            anno = np.load(anno_fn)
            self.images.append(image)
            self.segs.append(seg)
            self.annos.append(anno)
            self.names.append(image_name.split('.')[0])

    def __getitem__(self, item):
        sample = dict()
        ori_img = self.images[item]
        img = ori_img / 255.
        seg = self.segs[item]
        anno = self.annos[item]
        name = self.names[item]
        pose_pred = 0.
        if name in self.kkeys:
            pose_pred = self.preds[self.kkeys[name]]['final'][0]
            if self.category == 'bicycle':
                pose_pred['azimuth'] = pose_pred['azimuth'] + np.pi / 2
            pose_pred['elevation'] = np.pi - pose_pred['elevation']

        distance = float(anno['distance'])
        print(distance)
        elevation = float(anno['elevation'])
        azimuth = float(anno['azimuth'])
        theta = float(anno['theta'])

        img = img.transpose(2, 0, 1)

        sample['img'] = np.ascontiguousarray(img).astype(np.float32)
        sample['img_ori'] = np.ascontiguousarray(ori_img).astype(np.float32)
        sample['seg'] = np.ascontiguousarray(seg).astype(np.float32)
        sample['distance'] = distance
        sample['elevation'] = elevation
        sample['azimuth'] = azimuth
        sample['theta'] = theta
        sample['save_path'] = os.path.join(self.save_path, name)
        sample['pose_pred'] = pose_pred

        return sample

    def __len__(self):
        return len(self.images)