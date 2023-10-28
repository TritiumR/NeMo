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
    def __init__(self, dataset_config, cate='aeroplane', type=None):
        if cate == 'jeep':
            cate_id = 'n03594945'  # .glb
        elif cate == 'aeroplane':
            cate_id = 'n02690373'  # .obj
            chosen_id = '22831bc32bd744d3f06dea205edf9704'
        elif cate == 'sailboat':
            cate_id = 'n02981792'  # .glb
        elif cate == 'bicycle':
            cate_id = 'n02835271'  # .glb .obj
        elif cate == 'police':
            cate_id = 'n03977966'
            chosen_id = '372ceb40210589f8f500cc506a763c18'
        else:
            raise NotImplementedError

        index_path = os.path.join(dataset_config['root_path'], 'index', cate_id, chosen_id)

        self.mesh_path = os.path.join(dataset_config['root_path'], 'recon', cate_id)
        self.mesh_name_dict = dict()
        for name in os.listdir(self.mesh_path):
            name = name.split('_')[0]
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
        # offset = np.load(os.path.join(self.index_path, instance_id, 'offset.npy'), allow_pickle=True)[()]

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
        if cate == 'car':
            cate_id = '02958343'
        elif cate == 'aeroplane':
            cate_id = '02691156'
        elif cate == 'boat':
            cate_id = '04530566'
        elif cate == 'bicycle':
            cate_id = '02834778'
        elif cate == 'chair':
            cate_id = '03001627'
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
                chosen_verts = torch.from_numpy(chosen_verts.astype(np.float32))
            else:
                # load chosen mesh
                ori_mesh_path = os.path.join(dataset_config['root_path'], 'ori_mesh', cate_id, chosen_id, 'models',
                                             'model_normalized.obj')
                chosen_verts, _, _ = load_obj(ori_mesh_path)
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

    def get_ori_part(self, id, name):
        part_path = os.path.join(self.dataset_config['root_path'], 'ori_parts', self.cate, id)
        part_fn = os.path.join(part_path, f'{name}.obj')
        part_vert, faces_idx, _ = load_obj(part_fn)
        part_face = faces_idx.verts_idx
        part_middle = (part_vert.max(axis=0)[0] + part_vert.min(axis=0)[0]) / 2
        part_vert = part_vert - part_middle

        return part_vert.numpy(), part_face.numpy()


class DSTPartShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.category = category
        self.root_path = get_abs_path(root_path)

        if self.category == 'aeroplane':
            cate_id = 'n02690373'
            part_list = ['1c27d282735f81211063b9885ddcbb1', '1d96d1c7cfb1085e61f1ef59130c405d',
                          '1de008320c90274d366b1ebd023111a8', '4ad92be763c2ded8fca1f1143bb6bc17',
                          '4fbdfec0f9ee078dc1ccec171a275967', '7f2d03635180db2137678474be485ca',
                          '7f4a0b23f1256c879a6e43b878d5b335', '8adc6a0f45a1ef2e71d03b466c72ce41',
                          '48bcce07b0baf689d9e6f00e848ea18', '66a32714d2344d1bf52a658ce0ec2c1']
        elif self.category == 'police':
            cate_id = 'n03977966'
            part_list = ['1a7125aefa9af6b6597505fd7d99b613', '45e69263902d77304dde7b6e74a2cede',
                         '275df71b6258e818597505fd7d99b613', '479f89af38e88bc9715e04edb8af9c53']
            part_list1 = ['45186c083231f2207b5338996083748c', '511962626501e4abf500cc506a763c18',
                          '498e4295b3aeca9fefddc097534e4553', '5389c96e84e9e0582b1e8dc2f1faa8cb',
                          '7492ced6cb6289c556de8db8652eec4e', '9511b5ded804a33f597505fd7d99b613',
                          'a5d32daf96820ca5f63ee8a34069b7c5', 'e90a136270c03eebaaafd94b9f216ef6']
        else:
            raise NotImplementedError

        self.cate_path = os.path.join(self.root_path, 'train', cate_id)
        if data_type == 'train':
            self.instance_list = [x for x in os.listdir(self.cate_path) if '.' not in x and x not in part_list]
        if data_type == 'test':
            self.instance_list = part_list

        print('instance_used: ', len(self.instance_list))

        self.img_fns = []
        self.angle_fns = []
        self.render_img_fns = []
        lambda_fn = lambda x: int(x[:3])
        for instance_id in self.instance_list:
            img_path = os.path.join(self.cate_path, instance_id, 'image_minigpt4_1008')
            render_img_path = os.path.join(self.cate_path, instance_id, 'image_render')
            anno_path = os.path.join(self.cate_path, instance_id, 'annotation')
            img_list = os.listdir(img_path)
            img_list = sorted(img_list, key=lambda_fn)
            self.img_fns += [os.path.join(img_path, x) for x in img_list]
            angle_list = os.listdir(anno_path)
            angle_list = sorted(angle_list, key=lambda_fn)
            self.angle_fns += [os.path.join(anno_path, x) for x in angle_list]
            render_img_list = os.listdir(render_img_path)
            render_img_list = sorted(render_img_list, key=lambda_fn)
            self.render_img_fns += [os.path.join(render_img_path, x) for x in render_img_list]

            assert len(self.img_fns) == len(self.angle_fns) == len(self.render_img_fns), \
                f'{len(self.img_fns)}, {len(self.angle_fns)}, {len(self.render_img_fns)}'

    def __getitem__(self, item):
        ori_img = cv2.imread(self.img_fns[item], cv2.IMREAD_UNCHANGED)
        ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        render_img = cv2.imread(self.render_img_fns[item], cv2.IMREAD_UNCHANGED)
        ori_img = np.array(ori_img)
        render_img = np.array(render_img)
        angle = np.load(self.angle_fns[item], allow_pickle=True)[()]

        instance_id = self.img_fns[item].split('/')[-3]

        img = ori_img.transpose(2, 0, 1)
        mask = render_img[:, :, 3]
        mask = cv2.resize(mask, (img.shape[2], img.shape[1]), interpolation=cv2.INTER_NEAREST)
        mask[mask > 0] = 1
        vis_mask = mask * 255
        cv2.imwrite(get_abs_path('visual/mask.png'), vis_mask)

        distance = angle['dist']
        # print('distence: ', distance)
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
