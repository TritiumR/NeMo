import sys
sys.path.append('../nemo/lib')
sys.path.append('../nemo')
import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.cameras import look_at_view_transform
from config import get_config, print_usage
from capsule import Network
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import point_cloud_utils as pcu


def fps(points, n_samples):
    """
    points: [N, 3] array containing the whole point cloud
    n_samples: samples you want in the sampled point cloud typically << N
    """
    points = np.array(points)

    # Represent the points by their indices in points
    points_left = np.arange(len(points))  # [P]

    # Initialise an array for the sampled indices
    sample_inds = np.zeros(n_samples, dtype='int')  # [S]

    # Initialise distances to inf
    dists = np.ones_like(points_left) * float('inf')  # [P]

    # Select a point from points by its index, save it
    selected = 0
    sample_inds[0] = points_left[selected]

    # Delete selected
    points_left = np.delete(points_left, selected)  # [P - 1]

    # Iteratively select points for a maximum of n_samples
    for i in range(1, n_samples):
        # Find the distance to the last added point in selected
        # and all the others
        last_added = sample_inds[i - 1]

        dist_to_last_added_point = (
                (points[last_added] - points[points_left]) ** 2).sum(-1)  # [P - i]

        # If closer, updated distances
        dists[points_left] = np.minimum(dist_to_last_added_point,
                                        dists[points_left])  # [P - i]

        # We want to pick the one that has the largest nearest neighbour
        # distance to the sampled points
        selected = np.argmax(dists[points_left])
        sample_inds[i] = points_left[selected]

        # Update points_left
        points_left = np.delete(points_left, selected)

    return points[sample_inds], sample_inds


def vis_pts_att(pts, label_map, fn="temp.png", marker=".", alpha=0.9):
    # pts (n, d): numpy, d-dim point cloud
    # label_map (n, ): numpy or None
    # fn: filename of visualization
    assert pts.shape[1] == 3
    TH = 0.7
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_zlim(-TH,TH)
    ax.set_xlim(-TH,TH)
    ax.set_ylim(-TH,TH)
    xs = pts[:, 0]
    ys = pts[:, 1]
    zs = pts[:, 2]
    if label_map is not None:
        ax.scatter(xs, ys, zs, c=label_map, cmap="jet", marker=marker, alpha=alpha)
    else:
        ax.scatter(xs, ys, zs, marker=marker, alpha=alpha, edgecolor="none")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    plt.savefig(
        fn,
        bbox_inches='tight',
        pad_inches=0,
        dpi=300,)
    plt.close()


device = 'cuda:0'

# prepare network
config, unparsed = get_config()
if len(unparsed) > 0:
    print_usage()
    exit(1)

num_precess_point = int(input('num_precess_point: '))
if num_precess_point > config.num_pts:
    print('no more than num_pts')
    exit(1)

config.res_dir = '../../canonical-capsules/logs'

network = Network(config)
network.model.eval()
network.load_checkpoint()


def capsule_decompose(pc, R=None, T=None):
    with torch.no_grad():
        _x = pc[None]
        _labels, _feats = network.model.decompose_one_pc(_x, R, T)
    return _labels, _feats


if config.cat_type == 'car':
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/test/car'
    recon_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/car'
    ori_mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02958343'
    save_path = '../data/ShapeNet/Preprocess/car'
    chosen_instance = '4d22bfe3097f63236436916a86a90ed7'
elif config.cat_type == 'plane':
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/test/aeroplane'
    recon_path = '../data/CorrData/mesh/aeroplane'
    ori_mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02691156'
    save_path = '../data/ShapeNet/Preprocess/aeroplane'
    chosen_instance = '3cb63efff711cfc035fc197bbabcd5bd'
elif config.cat_type == 'boat':
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/boat'
    recon_path = '../data/CorrData/mesh/boat'
    ori_mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/04530566'
    save_path = '../data/ShapeNet/Preprocess/boat'
    # chosen_instance = '2340319ec4d93ae8c1df6b0203ecb359'
    chosen_instance = '246335e0dfc3a0ea834ac3b5e36b95c'
elif config.cat_type == 'bicycle':
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/test/bicycle'
    recon_path = '../data/CorrData/mesh/bicycle'
    ori_mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02834778'
    save_path = '../data/ShapeNet/Preprocess/bicycle'
    chosen_instance = '91k7HKqdM9'
elif config.cat_type == 'chair':
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/chair'
    recon_path = '../data/CorrData/mesh/chair'
    ori_mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/03001627'
    save_path = '../data/ShapeNet/Preprocess/chair'
    chosen_instance = '10de9af4e91682851e5f7bff98fb8d02'
else:
    raise NotImplementedError

save_path = os.path.join(save_path, chosen_instance)
if not os.path.exists(save_path):
    os.makedirs(save_path)

instance_path = os.path.join(save_path, f'{chosen_instance}')
if not os.path.exists(instance_path):
    os.makedirs(instance_path)

recon_fn = os.path.join(recon_path, f"{chosen_instance}_recon_mesh.ply")
verts, faces, _, _ = pcu.load_mesh_vfnc(recon_fn)

# normalize (scale: recon, offset:ori)
vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
if config.cat_type == 'bicycle':
    ori_fn = os.path.join(ori_mesh_path, chosen_instance, 'model.obj')
else:
    ori_fn = os.path.join(ori_mesh_path, chosen_instance, 'models', 'model_normalized.obj')
ori_verts, _, _ = load_obj(ori_fn, device=device)
vert_middle = (ori_verts.max(dim=0)[0] + ori_verts.min(dim=0)[0]) / 2 * vert_scale
np.save(os.path.join(instance_path, 'offset.npy'), vert_middle.cpu().numpy())

# decompose using point cloud
v = verts
if os.path.exists(os.path.join(save_path, f'{chosen_instance}', 'index.npy')):
    prev_idx = np.load(os.path.join(save_path, f'{chosen_instance}', 'index.npy'), allow_pickle=True)[()]
    print('yes')
else:
    prev_idx = np.random.choice(len(v), config.num_pts, replace=False)
prev_x = torch.from_numpy(v[prev_idx]).to(device, dtype=torch.float32)
R_can = torch.tensor([[[0.3456, 0.5633, 0.7505],
                       [-0.9333, 0.2898, 0.2122],
                       [-0.0980, -0.7737, 0.6259]]]).to(device)
T_can = torch.tensor([[[-0.0161], [-0.0014], [-0.0346]]]).to(device)
_, prev_feats = capsule_decompose(prev_x, R_can, T_can)

np.save(os.path.join(save_path, f'{chosen_instance}', 'index.npy'), prev_idx)

instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))

for instance_id in instance_ids:
    if '.' in instance_id or instance_id == chosen_instance:
        continue
    instance_path = os.path.join(save_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # using processed mesh
    recon_fn = os.path.join(recon_path, f"{instance_id}_recon_mesh.ply")
    verts, _, _, _ = pcu.load_mesh_vfnc(recon_fn)

    # normalize (scale: recon, offset:ori)
    vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
    if config.cat_type == 'bicycle':
        ori_fn = os.path.join(ori_mesh_path, instance_id, 'model.obj')
    else:
        ori_fn = os.path.join(ori_mesh_path, instance_id, 'models', 'model_normalized.obj')
    ori_verts, _, _ = load_obj(ori_fn, device=device)
    vert_middle = (ori_verts.max(dim=0)[0] + ori_verts.min(dim=0)[0]) / 2 * vert_scale
    # ori_scale = ((ori_verts.max(dim=0)[0] - ori_verts.min(dim=0)[0]) ** 2).sum() ** 0.5
    # print('ori_scale: ', ori_scale)
    # np.save(os.path.join(instance_path, 'offset.npy'), vert_middle.cpu().numpy())

    # decompose using point cloud
    v = verts
    idx = np.random.choice(len(v), config.num_pts, replace=False)
    x = torch.from_numpy(v[idx]).to(device, dtype=torch.float32)
    R_can = torch.tensor([[[0.3456, 0.5633, 0.7505],
                           [-0.9333, 0.2898, 0.2122],
                           [-0.0980, -0.7737, 0.6259]]]).to(device)
    T_can = torch.tensor([[[-0.0161], [-0.0014], [-0.0346]]]).to(device)
    _, feats = capsule_decompose(x, R_can, T_can)

    # nearest neighbor
    kdtree = KDTree(x.cpu().numpy())
    _, nearest_idx = kdtree.query(verts, k=1)

    idx_list = []
    visual_list = []
    for point_idx in range(num_precess_point):
        point_feat = prev_feats[0, :, 0, point_idx, 0]
        sim_feats = feats[0, :, 0, :, 0]
        similarity = torch.matmul(point_feat, sim_feats) / torch.norm(point_feat) / torch.norm(sim_feats, dim=0)

        max_sim = torch.max(similarity).item()

        # map similarity to original vertices
        vert_similarity = similarity[nearest_idx]

        # avoid choosing the same point
        vert_similarity[idx_list] = -1.1

        max_point = torch.argmax(vert_similarity)
        idx_list.append(max_point.item())
        # print("max_point:", max_point.item(), "max_sim: ", max_sim)

    # # visualize last point correspondence
    # prev_label = np.zeros(config.num_pts)
    # prev_label[config.num_pts - 1] = 1
    # vis_pts_att(verts[idx_list], similarity[nearest_idx][idx_list].cpu(), os.path.join(instance_path, f'curr_{max_sim}.png'))
    # vis_pts_att(prev_x.cpu(), prev_label, os.path.join(instance_path, f'prev_{max_sim}.png'))

    # visualize first 10 points correspondence
    visual_label = np.zeros(num_precess_point)
    for i in range(10):
        visual_label[i] = i

    prev_x = prev_x[:num_precess_point]
    idx_list = idx_list[:num_precess_point]
    vis_pts_att(prev_x.cpu(), visual_label, os.path.join(instance_path, f'prev.png'))
    vis_pts_att(verts[idx_list], visual_label, os.path.join(instance_path, f'curr.png'))

    idx_list = np.array(idx_list)

    np.save(os.path.join(instance_path, 'index.npy'), idx_list)
