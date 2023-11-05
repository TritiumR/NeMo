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
import trimesh

from nemo.utils import get_abs_path
import os


class MeshLoader():
    def __init__(self, dataset_config, cate='airliner', type=None):
        if cate == 'jeep':
            cate_id = 'n03594945'  # .glb
            chosen_id = '178f22467bae4c729bdcc15dbc7e445d'
        elif cate == 'airliner':
            cate_id = 'n02690373'  # .obj
            chosen_id = '22831bc32bd744d3f06dea205edf9704'
            self.anno_parts = ['engine', 'fuselarge', 'wing', 'vertical_stabilizer', 'wheel', 'horizontal_stabilizer']
        elif cate == 'sailboat':
            cate_id = 'n02981792'  # .glb
            chosen_id = '246335e0dfc3a0ea834ac3b5e36b95c'
            self.anno_parts = ['sail', 'body']
        elif cate == 'bike':
            cate_id = 'n02835271'  # .glb .obj
            chosen_id = '91k7HKqdM9'
            self.anno_parts = ['wheel', 'frame']
        elif cate == 'police':
            cate_id = 'n03977966'
            chosen_id = '372ceb40210589f8f500cc506a763c18'
            self.anno_parts = ['wheel', 'front_trunk', 'body']
        elif cate == 'police1':
            cate_id = 'n03977966'
            chosen_id = '372ceb40210589f8f500cc506a763c18'
            self.anno_parts = ['wheel', 'door', 'front_trunk', 'back_trunk', 'frame', 'mirror']
        elif cate == 'bench':
            cate_id = 'n03891251'
            chosen_id = '1b0463c11f3cc1b3601104cd2d998272'
            # self.anno_parts = ['arm', 'backrest', 'beam', 'seat', 'leg']
            self.anno_parts = ['backrest', 'seat', 'side']
        else:
            raise NotImplementedError

        index_path = os.path.join(dataset_config['root_path'], 'index', cate_id, chosen_id)

        self.mesh_name_dict = dict()

        if dataset_config['mesh_type'] == 'ply':
            self.mesh_path = os.path.join(dataset_config['root_path'], 'recon', cate_id)
            for name in os.listdir(self.mesh_path):
                name = name.split('_')[0]
                self.mesh_name_dict[name] = len(self.mesh_name_dict)
        else:
            self.mesh_path = os.path.join(dataset_config['root_path'], 'cad', cate_id)
            for name in os.listdir(self.mesh_path):
                name = name.split('.')[0]
                self.mesh_name_dict[name] = len(self.mesh_name_dict)

        if chosen_id not in self.mesh_name_dict:
            self.mesh_name_dict[chosen_id] = len(self.mesh_name_dict)

        # # careful!!!
        # self.mesh_name_dict = dict()
        # self.mesh_name_dict['246335e0dfc3a0ea834ac3b5e36b95c'] = len(self.mesh_name_dict)

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
        ply_fn = os.path.join(self.mesh_path, f'{instance_id}_recon_mesh.ply')
        obj_fn = os.path.join(self.mesh_path, f'{instance_id}.obj')
        glb_fn = os.path.join(self.mesh_path, f'{instance_id}.glb')
        # print('glb_fn: ', glb_fn)
        if os.path.exists(ply_fn):
            verts, faces, _, _ = pcu.load_mesh_vfnc(ply_fn)

            # faces
            faces = torch.from_numpy(faces.astype(np.int32))

            # normalize
            vert_middle = (verts.max(axis=0) + verts.min(axis=0)) / 2
            vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
            verts = verts - vert_middle
            verts = verts / vert_scale
            verts = torch.from_numpy(verts.astype(np.float32))
        elif os.path.exists(obj_fn):
            verts, faces_idx, _ = load_obj(obj_fn)
            faces = faces_idx.verts_idx

            # normalize
            vert_middle = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2
            verts = verts - vert_middle
        elif os.path.exists(glb_fn):
            mesh = trimesh.load_mesh(glb_fn)
            if isinstance(mesh, trimesh.Scene):
                # Convert the scene to a single mesh
                mesh = trimesh.util.concatenate(mesh.dump())

            # Extract vertices and faces
            verts = mesh.vertices
            faces = mesh.faces

            vert_middle = (verts.max(axis=0) + verts.min(axis=0)) / 2
            vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
            verts = verts - vert_middle
            verts = verts / vert_scale
            verts = torch.from_numpy(verts.astype(np.float32))
            faces = torch.from_numpy(faces.astype(np.int32))
            # print('verts: ', verts.shape)
            # print('faces: ', faces.shape)
        else:
            print('no mesh!')
            verts = None
            faces = None

        return verts, faces


class PartsLoader():
    def __init__(self, dataset_config, cate='car', chosen_ids=None):
        if cate == 'airliner':
            cate_id = 'n02690373'
        elif cate == 'bench':
            cate_id = 'n03891251'
        elif cate == 'police' or cate == 'police1':
            cate_id = 'n03977966'
        elif cate == 'bike':
            cate_id = 'n02835271'
        elif cate == 'sailboat':
            cate_id = 'n02981792'
        else:
            raise NotImplementedError

        self.parts_meshes = dict()
        self.parts_offsets = dict()
        self.part_names = None
        self.dataset_config = dataset_config
        self.cate = cate

        for chosen_id in chosen_ids:
            if dataset_config['mesh_type'] == 'ply':
                recon_mesh_path = os.path.join(dataset_config['root_path'], 'recon', cate_id, f'{chosen_id}_recon_mesh.ply')
                chosen_verts, _, _, _ = pcu.load_mesh_vfnc(recon_mesh_path)
                chosen_verts = torch.from_numpy(chosen_verts.astype(np.float32))
            else:
                mesh_path = os.path.join(dataset_config['root_path'], 'cad', cate_id)
                obj_fn = os.path.join(mesh_path, f'{chosen_id}.obj')
                glb_fn = os.path.join(mesh_path, f'{chosen_id}.glb')
                if os.path.exists(obj_fn):
                    chosen_verts, _, _ = load_obj(obj_fn)
                elif os.path.exists(glb_fn):
                    mesh = trimesh.load_mesh(glb_fn)
                    if isinstance(mesh, trimesh.Scene):
                        # Convert the scene to a single mesh
                        mesh = trimesh.util.concatenate(mesh.dump())

                    # Extract vertices and faces
                    chosen_verts = mesh.vertices
                else:
                    chosen_verts = None

            vert_scale = ((chosen_verts.max(axis=0)[0] - chosen_verts.min(axis=0)[0]) ** 2).sum() ** 0.5
            vert_middle = (chosen_verts.max(axis=0)[0] + chosen_verts.min(axis=0)[0]) / 2

            # load annotated parts
            part_meshes = []
            offsets = []

            part_path = os.path.join(dataset_config['root_path'], cate_id, chosen_id)

            if self.part_names is None:
                self.part_names = []
                for name in os.listdir(part_path):
                    if '.ply' not in name:
                        continue
                    part_fn = os.path.join(part_path, name)
                    # print(part_fn)
                    part_verts, part_faces, _, _ = pcu.load_mesh_vfnc(part_fn)
                    part_verts = torch.from_numpy(part_verts.astype(np.float32))
                    part_faces = torch.from_numpy(part_faces.astype(np.int32))
                    part_verts = part_verts - vert_middle
                    part_verts = part_verts / vert_scale

                    part_middle = (part_verts.max(axis=0)[0] + part_verts.min(axis=0)[0]) / 2
                    part_verts = part_verts - part_middle
                    offsets.append(np.array(part_middle))
                    part_meshes.append((part_verts, part_faces))
                    self.part_names.append(name.split('.')[0])
            else:
                for name in self.part_names:
                    part_fn = os.path.join(part_path, f'{name}.ply')
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


class DSTPartShapeNet(Dataset):
    def __init__(self, data_type, category, root_path, **kwargs):
        super().__init__()
        self.category = category
        self.root_path = get_abs_path(root_path)

        skip_list = ['2a80c18fc2b4732bfb7c76304cb719f8',
                     '1d0dae2db37fcb6ab078c101ed808ecf', '1d9fc51fa296bac9a1770107888e7eb8',
                     '1f239e55de52bb63eefaf3e79e3e3454', '1a55e418c61e9dab6f4a86fe50d4c8f0',
                     '1bb14f4633ad52e3ae944a46a2846086', '1ffcb829779ad5942056b4bd5d870b47',
                     '2c78148669d842209411c9f61b90503e', 'f3750246b6564607afbefc61cb1683b1',]
        if self.category == 'airliner':
            cate_id = 'n02690373'
            part_list = ['1c27d282735f81211063b9885ddcbb1', '1d96d1c7cfb1085e61f1ef59130c405d',
                          '1de008320c90274d366b1ebd023111a8', '4ad92be763c2ded8fca1f1143bb6bc17',
                          '4fbdfec0f9ee078dc1ccec171a275967', '7f2d03635180db2137678474be485ca',
                          '7f4a0b23f1256c879a6e43b878d5b335', '8adc6a0f45a1ef2e71d03b466c72ce41',
                          '48bcce07b0baf689d9e6f00e848ea18', '66a32714d2344d1bf52a658ce0ec2c1']
        elif self.category == 'police' or self.category == 'police1':
            cate_id = 'n03977966'
            part_list = ['1a7125aefa9af6b6597505fd7d99b613', '45e69263902d77304dde7b6e74a2cede',
                         '275df71b6258e818597505fd7d99b613',]
            # part_list = ['275df71b6258e818597505fd7d99b613',]
            part_list1 = ['45186c083231f2207b5338996083748c', '511962626501e4abf500cc506a763c18',
                          '498e4295b3aeca9fefddc097534e4553', '5389c96e84e9e0582b1e8dc2f1faa8cb',
                          '7492ced6cb6289c556de8db8652eec4e', '9511b5ded804a33f597505fd7d99b613',
                          'a5d32daf96820ca5f63ee8a34069b7c5', 'e90a136270c03eebaaafd94b9f216ef6']
        elif self.category == 'bench':
            cate_id = 'n03891251'
            part_list = ['1a40eaf5919b1b3f3eaa2b95b99dae6', '1aa15011153c5d6aa64b59533813e6d6',
                         '1b1cffcc9435c559f155d75bbf62b80', '1b9ddee986099bb78880edc6251fa529',
                         '1b80cd42474a990ccd8655d05e2f9e02', '1b78416210cbdcf1b184e775cf66758c',
                         '1be83cdaf803fa3b827358da75ee7655', '1bf5b1fa54aeec162701a18243b45d3',
                         '1c310698c57a3d378fd6b27be619556b', '1dfcce44c2f98d2c81e7c6cfefba0d68']
        elif self.category == 'jeep':
            cate_id = 'n03594945'
            part_list1 = ['7c1f9f951ac5432584f0c36199c8d6fc', '8dd9a0b3faa84aaa813431b44c716a6d',
                         '178f22467bae4c729bdcc15dbc7e445d', '192f7e81dce84c6780434f692a0f96c5',
                         'f3750246b6564607afbefc61cb1683b1', '7886c8713f39495f88b84e882592e0a5',
                         'd6e73ec5537e49269ec92b8ab78230c2']
            part_list = ['a8c75ce1d4704e55bfecd1e81c60a373', '531f120594d946be8cd9d87b5095f856',
                         'c7b8a665fd6549668e27cf659f81f6db', 'cbc7c5da7fa94e1ba2bb2db376389c3f']
        elif self.category == 'bike':
            cate_id = 'n02835271'
            # part_list = ['3d7fc7394cde43298de89a28c0afbaff', '4saPxqRTOLETE9TlL5iZkKd9zI1',
            #              '8a9586b8eff9f3a6932abeb6e2fd4072', 'd54e0f22d4604d741f24b86004bccc4d',
            #              'efe1456e80e2a8646350c50e0a456641']
            part_list = ['4saPxqRTOLETE9TlL5iZkKd9zI1']
        elif self.category == 'sailboat':
            cate_id = 'n02981792'
            part_list = ['048ca9ab42b0453ab344a0691fbd5058', '3c7ae2bae2474175afe466345182da05',
                         '76d0b1e24be14d2f9a524bfce3001aeb', '8a7d855f005d4d0b8c17d10b3a2edf2c']
        else:
            raise NotImplementedError

        self.cate_path = os.path.join(self.root_path, 'train', cate_id)
        if data_type == 'train':
            self.instance_list = [x for x in os.listdir(self.cate_path) if '.' not in x and x not in part_list]
        if data_type == 'test':
            self.instance_list = part_list

        print('instance_used: ', len(self.instance_list))

        if data_type == 'test':
            img_per_instance = int(25 / len(part_list))
        else:
            img_per_instance = -1
        self.img_fns = []
        self.angle_fns = []
        self.render_img_fns = []
        self.segment_fns = []
        lambda_fn = lambda x: int(x[:3])
        for instance_id in self.instance_list:
            if instance_id in skip_list:
                continue
            img_path = os.path.join(self.cate_path, instance_id, 'image_minigpt4_1008')
            render_img_path = os.path.join(self.cate_path, instance_id, 'image_render')
            anno_path = os.path.join(self.cate_path, instance_id, 'annotation')
            img_list = os.listdir(img_path)
            img_list = sorted(img_list, key=lambda_fn)
            self.img_fns += [os.path.join(img_path, x) for x in img_list[:img_per_instance]]
            angle_list = os.listdir(anno_path)
            angle_list = sorted(angle_list, key=lambda_fn)
            self.angle_fns += [os.path.join(anno_path, x) for x in angle_list[:img_per_instance]]
            render_img_list = os.listdir(render_img_path)
            render_img_list = sorted(render_img_list, key=lambda_fn)
            self.render_img_fns += [os.path.join(render_img_path, x) for x in render_img_list[:img_per_instance]]
            if data_type == 'test':
                segment_path = os.path.join(self.root_path, 'segment1', cate_id, instance_id)
                segment_list = os.listdir(segment_path)
                segment_list = sorted(segment_list, key=lambda_fn)
                self.segment_fns += [os.path.join(segment_path, x) for x in segment_list[:img_per_instance]]

            assert len(self.img_fns) == len(self.angle_fns) == len(self.render_img_fns), \
                f'{len(self.img_fns)}, {len(self.angle_fns)}, {len(self.render_img_fns)}'

            if data_type == 'test':
                assert len(self.img_fns) == len(self.segment_fns), \
                    f'{len(self.img_fns)}, {len(self.segment_fns)}'

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

        if len(self.segment_fns) > 0:
            seg = cv2.imread(self.segment_fns[item], cv2.IMREAD_UNCHANGED)
            # print(seg.shape)
            sample['seg'] = np.ascontiguousarray(seg).astype(np.float32)

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
