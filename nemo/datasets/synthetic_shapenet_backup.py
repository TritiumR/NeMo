import os
import sys
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import point_cloud_utils as pcu

from nemo.utils import get_abs_path
import os


class MeshLoader():
    def __init__(self, dataset_config):
        self.mesh_path = os.path.join(self.dataset_config['root_path'], 'mesh', cate)
        self.mesh_name_list = [t.split('_recon_mesh')[0] for t in os.listdir(self.mesh_path)]


class SyntheticShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, data_camera_mode='shapnet_car', **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        self.root_path = get_abs_path(root_path)
        self.data_camera_mode = data_camera_mode

        cat_off = dict()
        cat_off['car'] = "02958343"
        if self.category == 'car':
            self.skip_list = ['17c32e15723ed6e0cd0bf4a0e76b8df5']
            self.ray_list = ['85f6145747a203becc08ff8f1f541268', '5343e944a7753108aa69dfdc5532bb13',
                             '67a3dfa9fb2d1f2bbda733a39f84326d']

        self.img_path = os.path.join(self.root_path, 'image', self.data_type, self.category)
        self.render_img_path = os.path.join(self.root_path, 'render_img', self.data_type, self.category)
        if self.data_type == 'train':
            self.index_path = os.path.join(self.root_path, 'index', self.category, '4d22bfe3097f63236436916a86a90ed7')
        self.angle_path = os.path.join(self.root_path, 'angle', self.data_type, self.category)
        self.mesh_path = os.path.join(self.root_path, 'mesh', self.category)
        self.ori_mesh = os.path.join(self.root_path, 'ori_mesh', cat_off[self.category])

        self.instance_list = [x for x in os.listdir(self.img_path) if '.' not in x]

        self.img_fns = []
        self.angle_fns = []
        self.render_img_fns = []
        max_verts = 0
        max_faces = 0
        lambda_fn = lambda x: int(x[:3])
        # max_count = 2
        for instance_id in self.instance_list:
            if instance_id in self.skip_list:
                continue
            # if max_count == 0:
            #     break
            # max_count -= 1
            img_list = os.listdir(os.path.join(self.img_path, instance_id))
            img_list = sorted(img_list, key=lambda_fn)
            self.img_fns += [os.path.join(self.img_path, instance_id, x) for x in img_list]
            angle_list = os.listdir(os.path.join(self.angle_path, instance_id))
            angle_list = sorted(angle_list, key=lambda_fn)
            self.angle_fns += [os.path.join(self.angle_path, instance_id, x) for x in angle_list]
            render_img_list = os.listdir(os.path.join(self.render_img_path, instance_id))
            render_img_list = sorted(render_img_list, key=lambda_fn)
            self.render_img_fns += [os.path.join(self.render_img_path, instance_id, x) for x in render_img_list]

            # print('img_fns: ', self.img_fns[-1])
            # print('angle_fns: ', self.angle_fns[-1])
            # print('render_img_fns: ', self.render_img_fns[-1])

            assert len(self.img_fns) == len(self.angle_fns) == len(self.render_img_fns), \
                f'{len(self.img_fns)}, {len(self.angle_fns)}, {len(self.render_img_fns)}'

            verts, faces, _, _ = pcu.load_mesh_vfnc(os.path.join(self.mesh_path, f'{instance_id}_recon_mesh.ply'))
            max_verts = max(max_verts, verts.shape[0])
            max_faces = max(max_faces, faces.shape[0])
        # print('max_verts: ', max_verts)
        # print('max_faces: ', max_faces)
        self.max_verts = max_verts
        self.max_faces = max_faces

    def get_meshes(self, instance_id, ):
        verts, faces, _, _ = pcu.load_mesh_vfnc(os.path.join(self.mesh_path, f'{instance_id}_recon_mesh.ply'))
        # offset = np.load(os.path.join(self.index_path, instance_id, 'offset.npy'), allow_pickle=True)[()]

        # faces
        faces = torch.from_numpy(faces.astype(np.int32))

        # normalize
        vert_middle = (verts.max(axis=0) + verts.min(axis=0)) / 2
        if instance_id in self.ray_list:
            vert_middle[1] += 0.05
        vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
        verts = verts - vert_middle
        verts = verts / vert_scale
        verts = torch.from_numpy(verts.astype(np.float32))
        
        return verts, faces

    def __getitem__(self, item):
        ori_img = cv2.imread(self.img_fns[item], cv2.IMREAD_UNCHANGED)
        render_img = cv2.imread(self.render_img_fns[item], cv2.IMREAD_UNCHANGED)
        # ori_img = Image.open(self.img_fns[item]).convert('RGBA')
        ori_img = np.array(ori_img)
        render_img = np.array(render_img)
        # print('ori_img.shape: ', ori_img.shape)
        # print('render_img.shape: ', render_img.shape)
        angle = np.load(self.angle_fns[item], allow_pickle=True)[()]

        instance_id = self.img_fns[item].split('/')[-2]

        verts, faces = self.get_meshes()

        faces_pad = torch.zeros((self.max_faces, 3), dtype=torch.int32)
        faces_pad[:faces.shape[0], :] = faces

        verts_pad = torch.zeros((self.max_verts, 3), dtype=torch.float32)
        verts_pad[:verts.shape[0], :] = verts

        img = ori_img.transpose(2, 0, 1)
        mask = render_img[:, :, 3]
        mask = cv2.resize(mask, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1
        # vis_mask = mask * 255
        # cv2.imwrite(get_abs_path('visual/mask.png'), vis_mask)

        distance = angle['dist']
        elevation = np.pi / 2 - angle['phi']
        azimuth = angle['theta'] + np.pi / 2
        theta = angle['camera_rotation']

        sample = dict()

        img = img / 255.0
        sample['img'] = np.ascontiguousarray(img).astype(np.float32)
        sample['obj_mask'] = np.ascontiguousarray(mask).astype(np.float32)

        sample['verts'] = verts_pad
        sample['verts_len'] = verts.shape[0]
        sample['faces'] = faces_pad
        sample['faces_len'] = faces.shape[0]
        sample['distance'] = distance
        sample['elevation'] = elevation
        sample['azimuth'] = azimuth
        sample['theta'] = theta

        if self.data_type == 'train':
            index = np.load(os.path.join(self.index_path, instance_id, 'index.npy'), allow_pickle=True)[()]
            sample['index'] = index

        return sample

    def __len__(self):
        return len(self.img_fns)


class Normalize:
    def __init__(self):
        self.trans = torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    def __call__(self, sample):
        sample["img"] = self.trans(sample["img"])
        return sample
