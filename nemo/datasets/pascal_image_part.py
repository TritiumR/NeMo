import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

from nemo.utils import get_abs_path


class PascalImagePart(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        root_path = os.path.join(root_path, 'pascalimagepart')
        self.root_path = get_abs_path(root_path)

        data_path = os.path.join(root_path, 'train', 'images', category)
        anno_path = os.path.join(root_path, 'train', 'annotations', category)

        self.images = []
        self.segs = []
        self.annos = []
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

    def __getitem__(self, item):
        sample = dict()
        ori_img = self.images[item]
        img = ori_img / 255.
        seg = self.segs[item]
        anno = self.annos[item]

        distance = float(anno['distance'])
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

        return sample

    def __len__(self):
        return len(self.images)