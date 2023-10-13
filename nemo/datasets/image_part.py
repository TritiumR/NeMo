import os
import sys
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision

from nemo.utils import get_abs_path


class ImagePart(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        root_path = os.path.join(root_path, 'imagepart')
        self.root_path = get_abs_path(root_path)

        image_path = os.path.join(root_path, category, 'crop_images', data_type)
        anno_path = os.path.join(root_path, category, 'crop_annotations', data_type)

        self.images = []
        self.annos = []
        for image_name in os.listdir(image_path):
            image_fn = os.path.join(image_path, image_name)
            anno_fn = os.path.join(anno_path, image_name)
            image = cv2.imread(image_fn, cv2.IMREAD_UNCHANGED)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            anno = cv2.imread(anno_fn, cv2.IMREAD_UNCHANGED)
            # vis_anno = (3 - anno) * 255 / 3
            # vis_anno = vis_anno.astype(np.uint8)
            # cv2.imwrite(get_abs_path(f'visual/image_{image_name}.png'), image)
            # cv2.imwrite(get_abs_path(f'visual/anno_{image_name}.png'), vis_anno)
            self.images.append(image)
            self.annos.append(anno)

    def __getitem__(self, item):
        sample = dict()
        ori_img = self.images[item]
        img = ori_img / 255.
        anno = self.annos[item]

        img = img.transpose(2, 0, 1)

        sample['img'] = np.ascontiguousarray(img).astype(np.float32)
        sample['img_ori'] = np.ascontiguousarray(ori_img).astype(np.float32)
        sample['anno'] = np.ascontiguousarray(anno).astype(np.float32)

        return sample

    def __len__(self):
        return len(self.images)