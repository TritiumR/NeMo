import os

import cv2
import numpy as np
from torch.utils.data import Dataset

from nemo.utils import get_abs_path


class SyntheticShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, shapenet_path, data_camera_mode='shapnet_car', **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        self.root_path = get_abs_path(root_path)
        self.data_camera_mode = data_camera_mode
        self.shape_path = os.path.join(shapenet_path, 'ShapeNetCore.v2')
        self.transforms = transforms

        self.img_path = os.path.join(self.root_path, 'Data_simple_512x512', self.data_type, self.category)
        self.anno_path = os.path.join(self.root_path, 'Annotations', self.data_type, self.category)
        self.mesh_path = os.path.join(self.shape_path, self.category)

        self.instance_list = [x for x in os.listdir(self.img_path) if '.' not in x]

        self.img_fns = []
        self.anno_fns = []
        for instance_id in self.instance_list:
            img_list = os.listdir(os.path.join(self.img_path, instance_id))
            self.img_fns += [os.path.join(self.img_path, x) for x in img_list]
            anno_list = os.listdir(os.path.join(self.anno_path, instance_id))
            self.anno_fns += [os.path.join(self.anno_path, x) for x in anno_list]

        print('len: ', len(self.img_fns))

    def __getitem__(self, item):
        ori_img = cv2.imread(self.img_fns[item], cv2.IMREAD_UNCHANGED)
        anno = np.load(self.anno_fns[item], allow_pickle=True)[()]

        instance_id = self.img_fns[item].split('/')[-2]
        verts, faces, _, _ = pcu.load_mesh_vfnc(os.path.join(self.mesh_path, instance_id, 'recon_mesh.ply'))

        img = ori_img[:, :, :3][:, :, ::-1]
        mask = ori_img[:, :, 3:4]

        condinfo = np.array([
            anno['rotation'],
            np.pi / 2.0 - anno['elevation']
        ], dtype=np.float32)

        sample = {}
        sample['img'] = np.ascontiguousarray(img)
        sample['mask'] = np.ascontiguousarray(mask)
        sample['verts'] = verts
        sample['faces'] = faces
        sample['condinfo'] = condinfo
        return sample

    def __len__(self):
        return len(self.img_fns)
