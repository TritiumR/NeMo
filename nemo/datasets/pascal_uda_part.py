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
    def __init__(self, dataset_config, cate='car', type=None):
        skip_list = ['511962626501e4abf500cc506a763c18']
        if cate == 'car_uda':
            cate_id = '02958343'
            chosen_id = '4d22bfe3097f63236436916a86a90ed7'
            self.anno_parts = ['body', 'wheel', 'mirror']
            cate = 'car'
        else:
            raise NotImplementedError

        index_path = os.path.join(dataset_config['root_path'], 'index', cate, chosen_id)

        self.mesh_path = os.path.join(dataset_config['root_path'], 'mesh', cate)
        self.mesh_name_dict = dict()
        for name in os.listdir(self.mesh_path):
            name = name.split('_')[0]
            if name in skip_list:
                continue
            self.mesh_name_dict[name] = len(self.mesh_name_dict)
        if chosen_id not in self.mesh_name_dict:
            self.mesh_name_dict[chosen_id] = len(self.mesh_name_dict)
        self.mesh_list = [self.get_meshes(name_) for name_ in self.mesh_name_dict.keys()]

        self.index_list = [np.load(os.path.join(index_path, t, 'index.npy'), allow_pickle=True)[()] for t in
                           self.mesh_name_dict.keys()]

    def get_mesh_listed(self):
        return [t[0].numpy() for t in self.mesh_list], [t[1].numpy() for t in self.mesh_list]

    def get_index_list(self, indexs=None):
        if indexs is not None:
            return torch.from_numpy(np.array([self.index_list[t] for t in indexs]))
        return torch.from_numpy(np.array(self.index_list))

    def get_meshes(self, instance_id, ):
        verts, faces, _, _ = pcu.load_mesh_vfnc(os.path.join(self.mesh_path, f'{instance_id}_recon_mesh.ply'))

        # faces
        faces = torch.from_numpy(faces.astype(np.int32))

        # normalize
        vert_middle = (verts.max(axis=0) + verts.min(axis=0)) / 2
        vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
        verts = verts - vert_middle
        verts = verts / vert_scale
        verts = torch.from_numpy(verts.astype(np.float32))

        return verts, faces


class PartsLoader():
    def __init__(self, dataset_config, cate='car', chosen_ids=None):
        if cate == 'car_uda':
            cate_id = '02958343'
            cate = 'car'
        else:
            raise NotImplementedError

        self.parts_meshes = dict()
        self.parts_offsets = dict()
        self.part_names = None
        self.dataset_config = dataset_config
        self.cate = cate

        for chosen_id in chosen_ids:
            if cate in ['bicycle', 'boat']:
                recon_mesh_path = os.path.join(dataset_config['root_path'], 'mesh', cate, f'{chosen_id}_recon_mesh.ply')
                chosen_verts, _, _, _ = pcu.load_mesh_vfnc(recon_mesh_path)
                vert_scale = ((chosen_verts.max(axis=0) - chosen_verts.min(axis=0)) ** 2).sum() ** 0.5
                chosen_verts = torch.from_numpy(chosen_verts.astype(np.float32))
            else:
                # load chosen mesh
                ori_mesh_path = os.path.join(dataset_config['root_path'], 'ori_mesh', cate_id, chosen_id, 'models', 'model_normalized.obj')
                chosen_verts, _, _ = load_obj(ori_mesh_path)
                vert_scale = 1
            vert_middle = (chosen_verts.max(axis=0)[0] + chosen_verts.min(axis=0)[0]) / 2
            if chosen_id in ['1d63eb2b1f78aa88acf77e718d93f3e1', '3cb63efff711cfc035fc197bbabcd5bd']:
                vert_middle[1] -= 0.08

            # load annotated parts
            part_meshes = []
            offsets = []
            if dataset_config.get('ori_mesh', False):
                part_path = os.path.join(dataset_config['root_path'], 'ori_parts', cate, chosen_id)
            else:
                part_path = os.path.join(dataset_config['root_path'], f'{cate}', chosen_id)

            if cate in ['bicycle', 'boat']:
                if self.part_names is None:
                    self.part_names = []
                    for name in os.listdir(part_path):
                        if '.ply' not in name:
                            continue
                        part_fn = os.path.join(part_path, name)
                        part_verts, part_faces, _, _ = pcu.load_mesh_vfnc(part_fn)
                        part_verts = torch.from_numpy(part_verts.astype(np.float32))
                        part_faces = torch.from_numpy(part_faces.astype(np.int32))
                        part_verts = part_verts - vert_middle
                        part_verts = part_verts / vert_scale
                        part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
                        part_verts = part_verts - part_middle
                        offsets.append(np.array(part_middle))
                        part_meshes.append((part_verts, part_faces))
                        self.part_names.append(name.split('.')[0].split('_')[0])
                else:
                    for name in self.part_names:
                        part_fn = os.path.join(part_path, f'{name}_recon.ply')
                        if not os.path.exists(part_fn):
                            part_meshes.append((torch.zeros(1, 3), torch.zeros(1, 3)))
                            offsets.append(np.array([[0., 0., 0.]]))
                            continue
                        part_verts, part_faces, _, _ = pcu.load_mesh_vfnc(part_fn)
                        part_verts = torch.from_numpy(part_verts.astype(np.float32))
                        part_faces = torch.from_numpy(part_faces.astype(np.int32))
                        part_verts = part_verts - vert_middle
                        part_verts = part_verts / vert_scale
                        part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
                        part_verts = part_verts - part_middle
                        offsets.append(np.array(part_middle))
                        part_meshes.append((part_verts, part_faces))
            else:
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
                        self.part_names.append(name.split('.')[0].split('_')[0])
                else:
                    for name in self.part_names:
                        if dataset_config.get('ori_mesh', False):
                            part_fn = os.path.join(part_path, f'{name}.obj')
                        else:
                            part_fn = os.path.join(part_path, f'{name}_recon.obj')
                        if not os.path.exists(part_fn):
                            # print('no part ', name)
                            part_meshes.append((torch.zeros(1, 3), torch.zeros(1, 3)))
                            offsets.append(np.array([[0., 0., 0.]]))
                            continue
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


class PascalUDAPart(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.data_type = data_type
        self.category = category
        root_path = os.path.join(root_path, 'pascalUDApart')
        self.root_path = get_abs_path(root_path)

        if category == 'car_uda':
            category = 'car'

        img_path = os.path.join(root_path, 'images', category)
        seg_path = os.path.join(root_path, 'images', category + '_merge')
        anno_path = os.path.join(root_path, 'annotations', category)
        pose_path = 'eval/pose_estimation_3d_nemo_%s' % category
        pose_path_train = 'eval1/pose_estimation_3d_nemo_%s_training' % category
        self.preds = torch.load(os.path.join(pose_path, 'pascal3d_occ0_%s_val.pth' % category))
        # self.preds.update(torch.load(os.path.join(pose_path_train, 'pascal3d_occ0_%s_val.pth' % category)))
        self.kkeys = {t.split('/')[1]: t for t in self.preds.keys()}
        print('len: ', len(self.kkeys))
        # print(self.kkeys)

        self.images = []
        self.segs = []
        self.annos = []
        self.names = []
        for image_name in os.listdir(img_path):
            if 'seg' in image_name:
                continue
            image_fn = os.path.join(img_path, image_name)
            seg_fn = os.path.join(seg_path, image_name.replace('.JPEG', '_seg.png'))
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
        # print(distance)
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
        sample['pose_pred'] = pose_pred

        return sample

    def __len__(self):
        return len(self.images)