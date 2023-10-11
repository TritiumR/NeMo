import sys
sys.path.append('../nemo/lib')
sys.path.append('../nemo')
sys.path.append('../')
import numpy as np
import os
from MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import load_objs_as_meshes, load_obj
from sklearn.neighbors import KDTree
import point_cloud_utils as pcu
import pytorch3d


device = 'cuda:0'
cat_type = input('category type: ')
use_ori = input('use ori mesh: ')
if use_ori == 'y':
    use_ori = True
else:
    use_ori = False

if cat_type == 'car':
    imgs_path = '../data/CorrData/image/train/car'
    recon_path = '../data/CorrData/mesh/car'
    mesh_path = '../data/CorrData/ori_mesh'
    save_path = '../data/CorrData/car'
    ref_path = '../data/CorrData/part/car'
    cate_id = '02958343'
    chosen_id = '4d22bfe3097f63236436916a86a90ed7'
    index_path = os.path.join('../data/CorrData/index/car', chosen_id)
elif cat_type == 'plane':
    annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/aeroplane'
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/aeroplane'
    recon_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/plane'
    mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02691156'
    save_path = '../data/CorrData/aeroplane'
    ref_path = '../data/CorrData/part/aeroplane'
    cate_id = '02691156'
    chosen_id = '1a32f10b20170883663e90eaf6b4ca52'
    index_path = os.path.join('../data/CorrData/index/aeroplane', chosen_id)
else:
    raise ValueError('Wrong type')

if not os.path.exists(save_path):
    os.makedirs(save_path)

# load chosen mesh
ori_mesh_path = os.path.join(mesh_path, cate_id, chosen_id, 'models', 'model_normalized.obj')
chosen_verts, _, _ = load_obj(ori_mesh_path)
vert_middle = (chosen_verts.max(axis=0)[0] + chosen_verts.min(axis=0)[0]) / 2
chosen_verts = chosen_verts - vert_middle
# load parts
parts_verts = []
part_names = []
part_path = os.path.join(ref_path)
for name in os.listdir(part_path):
    if '.obj' not in name:
        continue
    if 'mirror' in name:
        continue
    part_fn = os.path.join(part_path, name)
    part_verts, _, _ = load_obj(part_fn)
    part_verts = part_verts - vert_middle
    parts_verts.append(part_verts)
    part_names.append(name.split('.')[0])
print(part_names)

# load reconstructed mesh
recon_fn = os.path.join(recon_path, f"{chosen_id}_recon_mesh.ply")
chosen_recon_verts, _, _, _ = pcu.load_mesh_vfnc(recon_fn)
vert_middle = (chosen_recon_verts.max(axis=0) + chosen_recon_verts.min(axis=0)) / 2
vert_scale = ((chosen_recon_verts.max(axis=0) - chosen_recon_verts.min(axis=0)) ** 2).sum() ** 0.5
chosen_recon_verts = chosen_recon_verts - vert_middle
chosen_recon_verts = chosen_recon_verts / vert_scale
# load index
chosen_index = np.load(os.path.join(index_path, chosen_id, 'index.npy'), allow_pickle=True)[()]
chosen_indexed_verts = chosen_recon_verts[chosen_index]
# nearest part as label
min_dist = None
min_idx = None
for part_id, part_verts in enumerate(parts_verts):
    kdtree = KDTree(part_verts)
    nearest_dist, _ = kdtree.query(chosen_indexed_verts, k=1)
    nearest_dist = nearest_dist[:, 0]
    # print('nearest_dist: ', nearest_dist.shape)
    if min_dist is None:
        min_dist = nearest_dist
        min_idx = np.ones(len(min_dist)) * part_id
    else:
        min_idx[nearest_dist < min_dist] = part_id
        min_dist = np.minimum(min_dist, nearest_dist)

# transfer to different instances
instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))
for instance_id in instance_ids:
    if '.' in instance_id:
        continue
    instance_path = os.path.join(save_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # using processed mesh
    recon_fn = os.path.join(recon_path, f"{instance_id}_recon_mesh.ply")
    recon_verts, recon_faces, _, _ = pcu.load_mesh_vfnc(recon_fn)
    # normalize
    vert_middle = (recon_verts.max(axis=0) + recon_verts.min(axis=0)) / 2
    vert_scale = ((recon_verts.max(axis=0) - recon_verts.min(axis=0)) ** 2).sum() ** 0.5
    recon_verts = recon_verts - vert_middle
    recon_verts = recon_verts / vert_scale

    index = np.load(os.path.join(index_path, instance_id, 'index.npy'))
    indexed_verts = recon_verts[index]

    if use_ori:
        # load ori mesh
        ori_fn = os.path.join(mesh_path, cate_id, instance_id, 'models', 'model_normalized.obj')
        ori_verts, ori_faces, _ = load_obj(ori_fn)
        # normalize
        vert_middle = (ori_verts.max(axis=0)[0] + ori_verts.min(axis=0)[0]) / 2
        verts = ori_verts - vert_middle
        faces = ori_faces.verts_idx
    else:
        verts = recon_verts
        faces = recon_faces

    # nearest indexed_verts as label
    kdtree = KDTree(indexed_verts)
    _, nearest_idx = kdtree.query(verts, k=1)
    nearest_idx = nearest_idx[:, 0]
    label = min_idx[nearest_idx]

    for part_id in range(len(parts_verts)):
        print('part_id: ', part_id)
        part_list = dict()
        for idx in range(len(verts)):
            if label[idx] == part_id:
                part_list[idx] = len(part_list)
        # print('part_list: ', len(part_list))
        part_verts = verts[label == part_id]

        part_faces = []
        for face in faces:
            v1, v2, v3 = face
            v1 = v1.item()
            v2 = v2.item()
            v3 = v3.item()
            # print(v1, v2, v3)
            if v1 in part_list and v2 in part_list and v3 in part_list:
                face[0] = part_list[v1]
                face[1] = part_list[v2]
                face[2] = part_list[v3]
                part_faces.append(face)
        part_faces = np.array(part_faces, dtype=np.int32)
        print('part_faces: ', part_faces.shape)

        # save mesh as obj file
        if use_ori:
            save_fn = os.path.join(instance_path, f'{part_names[part_id]}_ori.obj')
            pytorch3d.io.save_obj(save_fn, part_verts, torch.from_numpy(part_faces))
        else:
            save_fn = os.path.join(instance_path, f'{part_names[part_id]}_recon.obj')
            pytorch3d.io.save_obj(save_fn, torch.from_numpy(part_verts), torch.from_numpy(part_faces))
        print('saved')

