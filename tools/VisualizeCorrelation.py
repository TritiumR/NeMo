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

annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/car'
imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/train/car'
meshs_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02958343'
save_path = '../visual/CapCorrelation'

if not os.path.exists(save_path):
    os.makedirs(save_path)

instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))

prev_feats = None

for instance_id in instance_ids:
    if '.' in instance_id:
        continue
    instance_path = os.path.join(save_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)

    # load mesh
    mesh_fn = os.path.join(meshs_path, instance_id, 'models', 'model_normalized.obj')

    verts, faces_idx, _ = load_obj(mesh_fn, device=device)
    faces = faces_idx.verts_idx

    # normalize
    vert_middle = verts.max(dim=0)[0] + verts.min(dim=0)[0]
    verts -= vert_middle / 2

    cmap = plt.get_cmap('plasma')

    # decompose using mesh
    v = verts.cpu().numpy()
    v, index = fps(v, config.num_pts)
    x = torch.from_numpy(v).to(device)
    input_x = x * 2.4
    R_can = torch.tensor([[[0.3456, 0.5633, 0.7505],
                           [-0.9333, 0.2898, 0.2122],
                           [-0.0980, -0.7737, 0.6259]]]).to(device)
    T_can = torch.tensor([[[-0.0161], [-0.0014], [-0.0346]]]).to(device)
    _, feats = capsule_decompose(input_x, R_can, T_can)

    if prev_feats is not None:
        for point_idx in range(2):
            point_id = np.random.randint(0, config.num_pts)
            point_feat = prev_feats[0, :, 0, point_id, 0]
            sim_feats = feats[0, :, 0, :, 0]
            similarity = torch.matmul(point_feat, sim_feats) / torch.norm(point_feat) / torch.norm(sim_feats, dim=0)

            if config.only_highest:
                max_point = torch.argmax(similarity)
                similarity = torch.zeros(config.num_pts)
                similarity[max_point] = 1.0

            vis_pts_att(input_x.cpu(), similarity.cpu(), os.path.join(instance_path, f'curr_{point_idx}.png'))

            # nearest neighbor
            kdtree = KDTree(x.cpu().numpy())
            _, nearest_idx = kdtree.query(verts.cpu().numpy(), k=1)
            similarity = similarity[nearest_idx][:, 0]
            similarity = (similarity - similarity.min()) / (similarity.max() - similarity.min())
            # print('sim_min: ', similarity.min(), 'sim_max: ', similarity.max())

            colors = cmap(similarity.cpu().numpy())
            verts_features = torch.tensor(colors[:, :3], dtype=torch.float32)[None]  # (1, V, 3)
            textures = Textures(verts_features=verts_features.to(device))

            meshes = Meshes(
                verts=[verts.to(device)],
                faces=[faces.to(device)],
                textures=textures
            )

            prev_similarity = torch.zeros(config.num_pts)
            prev_similarity[point_id] = 1.0

            vis_pts_att(prev_input_x.cpu(), prev_similarity.cpu(), os.path.join(instance_path, f'prev_{point_idx}.png'))

            # nearest neighbor
            kdtree = KDTree(prev_x.cpu().numpy())
            _, nearest_idx = kdtree.query(prev_verts.cpu().numpy(), k=1)
            prev_similarity = prev_similarity[nearest_idx][:, 0]
            prev_similarity = (prev_similarity - prev_similarity.min()) / (prev_similarity.max() - prev_similarity.min())

            prev_colors = cmap(prev_similarity.cpu().numpy())
            prev_verts_features = torch.tensor(prev_colors[:, :3], dtype=torch.float32)[None]  # (1, V, 3)
            prev_textures = Textures(verts_features=prev_verts_features.to(device))

            prev_meshes = Meshes(
                verts=[prev_verts.to(device)],
                faces=[prev_faces.to(device)],
                textures=prev_textures
            )

            img_path = os.path.join(imgs_path, instance_id)
            img_fns = os.listdir(img_path)

            print('image number: ', len(img_fns))
            for idx, img_fn in enumerate(img_fns):
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

                mixed_image = (image * 0.9 + img * 0.1).astype(np.uint8)
                Image.fromarray(mixed_image).save(os.path.join(instance_path, f'curr_{point_idx}_{count_id}.jpg'))

            img_path = os.path.join(imgs_path, prev_instance_id)
            img_fns = os.listdir(img_path)

            print('image number: ', len(img_fns))
            for idx, img_fn in enumerate(img_fns):
                img = np.array(Image.open(os.path.join(img_path, img_fn)))

                count_id = img_fn[:-7]
                anno_fn = os.path.join(annos_path, prev_instance_id, count_id + '.npy')
                print(anno_fn)
                anno = np.load(anno_fn, allow_pickle=True).item()
                distance = anno['dist']
                elevation = np.pi / 2 - anno['phi']
                azimuth = anno['theta'] + np.pi / 2
                camera_rotation = anno['camera_rotation']

                R, T = look_at_view_transform(distance, elevation, azimuth, device=device, degrees=False)
                R = torch.bmm(R, rotation_theta(float(camera_rotation), device_=device))

                image = phong_renderer(meshes_world=prev_meshes.clone(), R=R, T=T)
                image = image[0, ..., :3].detach().squeeze().cpu().numpy()

                image = np.array((image / image.max()) * 255).astype(np.uint8)

                mixed_image = (image * 0.9 + img * 0.1).astype(np.uint8)
                Image.fromarray(mixed_image).save(os.path.join(instance_path, f'prev_{point_idx}_{count_id}.jpg'))

    prev_x = x
    prev_input_x = input_x
    prev_feats = feats
    prev_verts = verts
    prev_faces = faces
    prev_instance_id = instance_id
    # exit(0)

