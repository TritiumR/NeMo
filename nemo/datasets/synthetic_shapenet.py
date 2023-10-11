import os
import sys
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision
import point_cloud_utils as pcu
from sklearn.neighbors import KDTree
from pytorch3d.io import load_obj

from nemo.utils import get_abs_path
import os


class MeshLoader():
    def __init__(self, dataset_config, cate='car', type='train'):
        if cate == 'car':
            self.skip_list = ['17c32e15723ed6e0cd0bf4a0e76b8df5']
            self.ray_list = ['85f6145747a203becc08ff8f1f541268', '5343e944a7753108aa69dfdc5532bb13',
                             '67a3dfa9fb2d1f2bbda733a39f84326d']
            cate_id = '02958343'
            chosen_id = '4d22bfe3097f63236436916a86a90ed7'
        elif cate == 'aeroplane':
            self.skip_list = []
            self.ray_list = []
            cate_id = '02691156'
            chosen_id = '3cb63efff711cfc035fc197bbabcd5bd'
        else:
            raise NotImplementedError

        index_path = os.path.join(dataset_config['root_path'], 'index', cate, chosen_id)

        self.mesh_path = os.path.join(dataset_config['root_path'], 'mesh', cate)
        self.mesh_name_dict = dict()
        for name in os.listdir(self.mesh_path):
            name = name.split('_')[0]
            if name in self.skip_list:
                continue
            self.mesh_name_dict[name] = len(self.mesh_name_dict)
        if chosen_id not in self.mesh_name_dict:
            self.mesh_name_dict[chosen_id] = len(self.mesh_name_dict)
        self.mesh_list = [self.get_meshes(name_) for name_ in self.mesh_name_dict.keys()]

        self.index_list = [np.load(os.path.join(index_path, t, 'index.npy'), allow_pickle=True)[()] for t in self.mesh_name_dict.keys()]
        # print('index_list: ', self.index_list[0].shape)

        if dataset_config.get('ori_mesh', False):
            self.ori_mesh_path = os.path.join(dataset_config['root_path'], 'ori_mesh', cate_id)
            self.ori_mesh_list = [self.get_ori_meshes(name_) for name_ in self.mesh_name_dict.keys()]

            # nearst point
            for idx, index in enumerate(self.index_list):
                sample_verts = self.mesh_list[idx][0][index]
                ori_verts = self.ori_mesh_list[idx][0]
                kdtree = KDTree(ori_verts)
                _, nearest_idx = kdtree.query(sample_verts, k=1)
                # print('nearest_idx: ', nearest_idx.shape)
                self.index_list[idx] = nearest_idx[:, 0]

    def get_mesh_listed(self):
        return [t[0].numpy() for t in self.mesh_list], [t[1].numpy() for t in self.mesh_list]

    def get_ori_mesh_listed(self):
        return [t[0].numpy() for t in self.ori_mesh_list], [t[1].numpy() for t in self.ori_mesh_list]

    def get_index_list(self, indexs=None):
        if indexs is not None:
            return torch.from_numpy(np.array([self.index_list[t] for t in indexs]))
        return torch.from_numpy(np.array(self.index_list))

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

    def get_ori_meshes(self, instance_id, ):
        mesh_fn = os.path.join(self.ori_mesh_path, instance_id, 'models', 'model_normalized.obj')
        verts, faces_idx, _ = load_obj(mesh_fn)
        faces = faces_idx.verts_idx

        # normalize
        vert_middle = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2
        verts = verts - vert_middle

        return verts, faces


class PartLoader():
    def __init__(self, dataset_config, cate='car'):
        if cate == 'car':
            chosen_id = '4d22bfe3097f63236436916a86a90ed7'
            cate_id = '02958343'
        elif cate == 'aeroplane':
            cate_id = '02691156'
            chosen_id = '3cb63efff711cfc035fc197bbabcd5bd'
        else:
            raise NotImplementedError

        # load chosen mesh
        ori_mesh_path = os.path.join(dataset_config['root_path'], 'ori_mesh', cate_id, chosen_id, 'models', 'model_normalized.obj')
        chosen_verts, _, _ = load_obj(ori_mesh_path)
        vert_middle = (chosen_verts.max(axis=0)[0] + chosen_verts.min(axis=0)[0]) / 2

        # load annotated parts
        self.part_meshes = []
        self.part_names = []
        self.offsets = []
        part_path = os.path.join(dataset_config['root_path'], 'part', cate)
        for name in os.listdir(part_path):
            if '.obj' not in name:
                continue
            part_fn = os.path.join(part_path, name)
            part_verts, faces_idx, _ = load_obj(part_fn)
            part_faces = faces_idx.verts_idx
            part_verts = part_verts - vert_middle
            part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
            part_verts = part_verts - part_middle
            self.offsets.append(np.array(part_middle))
            self.part_meshes.append((part_verts, part_faces))
            self.part_names.append(name.split('.')[0])

    def get_name_listed(self):
        return self.part_names

    def get_part_mesh(self, name=None):
        if name is None:
            return [mesh[0].numpy() for mesh in self.part_meshes], [mesh[1].numpy() for mesh in self.part_meshes]
        return self.part_meshes[self.part_names.index(name)]

    def get_offset(self, name=None):
        if name is None:
            return self.offsets
        return self.offsets[self.part_names.index(name)]


class PartsLoader():
    def __init__(self, dataset_config, cate='car', chosen_ids=None):
        if cate == 'car':
            cate_id = '02958343'
        elif cate == 'aeroplane':
            cate_id = '02691156'
        else:
            raise NotImplementedError

        self.parts_meshes = dict()
        self.parts_offsets = dict()
        self.part_names = None
        self.dataset_config = dataset_config
        self.cate = cate

        for chosen_id in chosen_ids:
            # load chosen mesh
            ori_mesh_path = os.path.join(dataset_config['root_path'], 'ori_mesh', cate_id, chosen_id, 'models', 'model_normalized.obj')
            chosen_verts, _, _ = load_obj(ori_mesh_path)
            vert_middle = (chosen_verts.max(axis=0)[0] + chosen_verts.min(axis=0)[0]) / 2

            # load annotated parts
            part_meshes = []
            offsets = []
            if dataset_config.get('ori_mesh', False):
                part_path = os.path.join(dataset_config['root_path'], 'ori_parts', cate, chosen_id)
            else:
                part_path = os.path.join(dataset_config['root_path'], cate, chosen_id)

            if self.part_names is None:
                self.part_names = []
                for name in os.listdir(part_path):
                    if '.obj' not in name:
                        continue
                    part_fn = os.path.join(part_path, name)
                    part_verts, faces_idx, _ = load_obj(part_fn)
                    part_faces = faces_idx.verts_idx
                    part_verts = part_verts - vert_middle
                    part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
                    part_verts = part_verts - part_middle
                    offsets.append(np.array(part_middle))
                    part_meshes.append((part_verts, part_faces))
                    self.part_names.append(name.split('.')[0])
            else:
                for name in self.part_names:
                    part_fn = os.path.join(part_path, f'{name}.obj')
                    part_verts, faces_idx, _ = load_obj(part_fn)
                    part_faces = faces_idx.verts_idx
                    part_verts = part_verts - vert_middle
                    part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
                    part_verts = part_verts - part_middle
                    offsets.append(np.array(part_middle))
                    part_meshes.append((part_verts, part_faces))

            self.parts_meshes[chosen_id] = part_meshes
            self.parts_offsets[chosen_id] = offsets

    def get_name_listed(self):
        return self.part_names

    def get_part_mesh(self, id=None, name=None):
        part_meshes = self.parts_meshes[id]
        if name is None:
            return [mesh[0].numpy() for mesh in part_meshes], [mesh[1].numpy() for mesh in part_meshes]
        return part_meshes[self.part_names.index(name)]

    def get_offset(self, id=None, name=None):
        offsets = self.parts_offsets[id]
        if name is None:
            return offsets
        return offsets[self.part_names.index(name)]

    def get_ori_part(self, id, name):
        part_path = os.path.join(self.dataset_config['root_path'], 'ori_parts', self.cate, id)
        part_fn = os.path.join(part_path, f'{name}.obj')
        part_vert, faces_idx, _ = load_obj(part_fn)
        part_face = faces_idx.verts_idx
        part_middle = (part_vert.max(axis=0)[0] + part_vert.min(axis=0)[0]) / 2
        part_vert = part_vert - part_middle

        return part_vert.numpy(), part_face.numpy()


class SyntheticShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, data_camera_mode='shapnet_car', **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        self.root_path = get_abs_path(root_path)
        self.data_camera_mode = data_camera_mode

        if self.category == 'car':
            self.skip_list = ['17c32e15723ed6e0cd0bf4a0e76b8df5']
            self.ray_list = ['85f6145747a203becc08ff8f1f541268', '5343e944a7753108aa69dfdc5532bb13',
                             '67a3dfa9fb2d1f2bbda733a39f84326d']
            cate_id = '02958343'
        elif self.category == 'aeroplane':
            self.skip_list= []
            self.ray_list = []
            cate_id = '02691156'
        else:
            raise NotImplementedError

        self.img_path = os.path.join(self.root_path, 'image', self.data_type, self.category)
        self.render_img_path = os.path.join(self.root_path, 'render_img', self.data_type, self.category)
        self.angle_path = os.path.join(self.root_path, 'angle', self.data_type, self.category)
        self.mesh_path = os.path.join(self.root_path, 'mesh', self.category)
        self.ori_mesh = os.path.join(self.root_path, 'ori_mesh', cate_id)

        self.instance_list = [x for x in os.listdir(self.img_path) if '.' not in x]
        # self.instance_list = ['4d22bfe3097f63236436916a86a90ed7']

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

    def __getitem__(self, item):
        ori_img = cv2.imread(self.img_fns[item], cv2.IMREAD_UNCHANGED)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        render_img = cv2.imread(self.render_img_fns[item], cv2.IMREAD_UNCHANGED)
        # ori_img = Image.open(self.img_fns[item]).convert('RGBA')
        ori_img = np.array(ori_img)
        render_img = np.array(render_img)
        # print('ori_img.shape: ', ori_img.shape)
        # print('render_img.shape: ', render_img.shape)
        angle = np.load(self.angle_fns[item], allow_pickle=True)[()]

        instance_id = self.img_fns[item].split('/')[-2]

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
        sample['img_ori'] = np.ascontiguousarray(img).astype(np.float32)
        sample['obj_mask'] = np.ascontiguousarray(mask).astype(np.float32)

        sample['distance'] = distance
        sample['elevation'] = elevation
        sample['azimuth'] = azimuth
        sample['theta'] = theta
        sample['instance_id'] = instance_id
        sample['this_name'] = item

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
