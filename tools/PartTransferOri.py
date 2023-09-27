import sys

sys.path.append('../nemo/lib')
sys.path.append('../nemo')
sys.path.append('../')
import torch
import numpy as np
import os
from PIL import Image, ImageDraw
from MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.cameras import look_at_view_transform
from pytorch3d import _C
from config import get_config, print_usage
from capsule import Network
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
import point_cloud_utils as pcu
from nemo.utils import load_off


def PointFaceDistance(points, points_first_idx, tris, tris_first_idx, max_points, min_triangle_area=5e-3):
    """
    Args:
        points: FloatTensor of shape `(P, 3)`
        points_first_idx: LongTensor of shape `(N,)` indicating the first point
            index in each example in the batch
        tris: FloatTensor of shape `(T, 3, 3)` of triangular faces. The `t`-th
            triangular face is spanned by `(tris[t, 0], tris[t, 1], tris[t, 2])`
        tris_first_idx: LongTensor of shape `(N,)` indicating the first face
            index in each example in the batch
        max_points: Scalar equal to maximum number of points in the batch
        min_triangle_area: (float, defaulted) Triangles of area less than this
            will be treated as points/lines.
    Returns:
        dists: FloatTensor of shape `(P,)`, where `dists[p]` is the squared
            euclidean distance of `p`-th point to the closest triangular face
            in the corresponding example in the batch
        idxs: LongTensor of shape `(P,)` indicating the closest triangular face
            in the corresponding example in the batch.

        `dists[p]` is
        `d(points[p], tris[idxs[p], 0], tris[idxs[p], 1], tris[idxs[p], 2])`
        where `d(u, v0, v1, v2)` is the distance of point `u` from the triangular
        face `(v0, v1, v2)`

    """
    dists, idxs = _C.point_face_dist_forward(
        points,
        points_first_idx,
        tris,
        tris_first_idx,
        max_points,
        min_triangle_area,
    )
    return dists, idxs

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

config.res_dir = '../../canonical-capsules/logs'

network = Network(config)
network.model.eval()
network.load_checkpoint()


def capsule_decompose(pc, R=None, T=None):
    with torch.no_grad():
        _x = pc[None]
        _labels, _feats = network.model.decompose_one_pc(_x, R, T)
    return _labels, _feats


# render config
render_image_size = (512, 512)
image_size = (512, 512)

blend_params = BlendParams(sigma=1e-4, gamma=1e-4)
raster_settings = RasterizationSettings(
    image_size=render_image_size,
    blur_radius=0.0,
    faces_per_pixel=1,
    bin_size=0
)
# We can add a point light in front of the object.
lights = PointLights(device=device, location=((2.0, 2.0, -2.0),))

# prepare camera
cameras = PerspectiveCameras(focal_length=1.0 * 3200,
                             principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                             image_size=(render_image_size,), device=device, in_ndc=False)

phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights, cameras=cameras),
)

if config.cat_type == 'car':
    annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/car'
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple/train/car'
    recon_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/car'
    mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02958343'
    save_path = '../visual/SegmentTransfer/car'
    ref_path = '../data/CorrData/segment_part/car'
    chosen_instance = '4d22bfe3097f63236436916a86a90ed7'

    R_can = torch.tensor([[[0.3456, 0.5633, 0.7505],
                           [-0.9333, 0.2898, 0.2122],
                           [-0.0980, -0.7737, 0.6259]]]).to(device)
    T_can = torch.tensor([[[-0.0161], [-0.0014], [-0.0346]]]).to(device)
elif config.cat_type == 'plane':
    annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/aeroplane'
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/aeroplane'
    recon_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/plane'
    mesh_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02691156'
    save_path = '../visual/SegmentTransfer/plane'
    ref_path = '../data/CorrData/segment_part/plane'
    # chosen_instance = '1a888c2c86248bbcf2b0736dd4d8afe0'
    chosen_instance = '1a32f10b20170883663e90eaf6b4ca52'
    R_can = torch.tensor([[[-0.1290, 0.8888, 0.4397],
                           [-0.5100, 0.3208, -0.7981],
                           [-0.8505, -0.3272, 0.4119]]]).to(device)
    T_can = torch.tensor([[[0.0], [0.0], [0.0]]]).to(device)
else:
    raise ValueError('Wrong type')

save_path = os.path.join(save_path, chosen_instance)
if not os.path.exists(save_path):
    os.makedirs(save_path)

# # load mesh from .off
# ref_mesh_fn = os.path.join(ref_path, f"{chosen_instance}.off")
# verts, faces, = load_off(ref_mesh_fn, to_torch=False)
# # change axis
# x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
# verts = np.stack([z, y, x], axis=1) * 2.3
#
# # load label from .txt
# ref_label_fn = os.path.join(ref_path, f"{chosen_instance}_labels.txt")
# file_handle = open(ref_label_fn)
# file_list = file_handle.readlines()
# part_label = np.zeros(len(verts))
# line = 0
# while line < len(file_list):
#     assert 'label' in file_list[line]
#     line += 1
#     start_id = int(file_list[line].split(' ')[0])
#     end_id = int(file_list[line].split(' ')[-1])
#     part_label[faces[start_id: end_id + 1]] = line // 3
#     line += 2

# load point cloud and label from .txt
ref_fn = os.path.join(ref_path, f"{chosen_instance}.txt")
file_handle = open(ref_fn)
file_list = file_handle.readlines()
all_strings = "".join(file_list)
info = np.fromstring(all_strings, dtype=np.float32, sep="\n")
info = info.reshape((-1, 7))
x, y, z = info[:, 0], info[:, 1], info[:, 2]
verts = np.stack([z, y, -x], axis=1) * 2.3
norms = info[:, 3:6]
part_label = info[:, 6].astype(np.int32)

vis_pts_att(verts, part_label / np.max(part_label), os.path.join(save_path, f'ref_segment.png'))

# decompose using point cloud
v = verts
if len(v) > config.num_pts:
    prev_idx = np.random.choice(len(v), config.num_pts, replace=False)
else:
    prev_idx = np.random.choice(len(v), config.num_pts, replace=True)
prev_x = torch.from_numpy(v[prev_idx]).to(device, dtype=torch.float32)
x_label = part_label[prev_idx]

label, prev_feats = capsule_decompose(prev_x, R_can, T_can)
prev_feats = prev_feats[0, :, 0, :, 0]
prev_feats = prev_feats / torch.norm(prev_feats, dim=0)


vis_pts_att(prev_x.cpu(), label.cpu(), os.path.join(save_path, f'cap_decompose.png'))

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

    # decompose using point cloud
    v = recon_verts
    # times = len(v) // config.num_pts
    times = 1
    idx = np.random.choice(len(v), config.num_pts * times, replace=False)
    points = v[idx]
    trans_part_label = np.zeros(len(idx))
    for time in range(times):
        x = torch.from_numpy(points[time * config.num_pts: (time + 1) * config.num_pts]).to(device, dtype=torch.float32)
        label, feats = capsule_decompose(x, R_can, T_can)
        feats = feats[0, :, 0, :, 0]
        feats = feats / torch.norm(feats, dim=0)
        similarity = torch.matmul(prev_feats.transpose(0, 1), feats)
        max_point = torch.argmax(similarity, dim=0)
        trans_part_label[time * config.num_pts: (time + 1) * config.num_pts] = x_label[max_point.cpu().numpy()]

    vis_pts_att(x.cpu(), label.cpu(), os.path.join(instance_path, f'decompose.png'))
    vis_pts_att(points, trans_part_label, os.path.join(instance_path, f'transfer_part.png'))

    if config.ori_mesh:
        # load original mesh
        mesh_fn = os.path.join(mesh_path, instance_id, 'models', 'model_normalized.obj')
        verts, faces_idx, _ = load_obj(mesh_fn, device=device)
        faces = faces_idx.verts_idx
    else:
        verts = torch.from_numpy(recon_verts.astype(np.float32)).to(device)
        faces = torch.from_numpy(recon_faces.astype(np.int32)).to(device)

    # normalize
    vert_middle = (verts.max(axis=0)[0] + verts.min(axis=0)[0]) / 2
    vert_scale = ((verts.max(axis=0)[0] - verts.min(axis=0)[0]) ** 2).sum() ** 0.5
    verts = (verts - vert_middle) / vert_scale
    point_middle = (points.max(axis=0) + points.min(axis=0)) / 2
    point_scale = ((points.max(axis=0) - points.min(axis=0)) ** 2).sum() ** 0.5
    points = (points - point_middle) / point_scale

    # nearest neighbor
    kdtree = KDTree(points)
    _, nearest_idx = kdtree.query(verts.cpu(), k=1)
    trans_part_label = trans_part_label[nearest_idx][:, 0]

    # construct mesh
    cmap = plt.get_cmap('jet')
    colors = cmap((trans_part_label - trans_part_label.min()) / (trans_part_label.max() - trans_part_label.min()))
    verts_features = torch.tensor(colors[:, :3], dtype=torch.float32)[None]  # (1, V, 3)
    textures = Textures(verts_features=verts_features.to(device))

    meshes = Meshes(
        verts=[verts],
        faces=[faces],
        textures=textures
    )

    img_path = os.path.join(imgs_path, instance_id)
    img_fns = os.listdir(img_path)

    print('image number: ', len(img_fns))
    for idx, img_fn in enumerate(img_fns[:10]):
        img = np.array(Image.open(os.path.join(img_path, img_fn)))

        count_id = img_fn[:-7]
        anno_fn = os.path.join(annos_path, instance_id, count_id + '.npy')
        print(anno_fn)
        anno = np.load(anno_fn, allow_pickle=True).item()
        distance = anno['dist']
        elevation = np.pi / 2 - anno['phi']
        azimuth = anno['theta'] + np.pi / 2
        camera_rotation = anno['camera_rotation']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=device, degrees=False)
        R = torch.bmm(R, rotation_theta(float(camera_rotation), device_=device))

        image = phong_renderer(meshes_world=meshes.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()

        image = np.array((image / image.max()) * 255).astype(np.uint8)

        mixed_image = (image * 1. + img * 0.).astype(np.uint8)
        Image.fromarray(mixed_image).save(os.path.join(instance_path, f'{count_id}.jpg'))
