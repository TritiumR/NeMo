# need "blender" env
import os
import open3d as o3d
import point_cloud_utils as pcu
import numpy as np

cat_type = input('category type: ')

if cat_type == 'aeroplane':
    imgs_path = '../data/CorrData/DST_part3d/train/n02690373'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/02691156/ply'
    save_path = '../data/CorrData/DST_part3d/recon/n02690373'
if cat_type == 'jeep':
    imgs_path = '../data/CorrData/DST_part3d/train/n03594945'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/02958343/ply'
    save_path = '../data/CorrData/DST_part3d/recon/n03594945'
if cat_type == 'sailboat':
    imgs_path = '../data/CorrData/DST_part3d/train/n02981792'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/04530566/ply'
    save_path = '../data/CorrData/DST_part3d/recon/n02981792'
if cat_type == 'police':
    imgs_path = '../data/CorrData/DST_part3d/train/n03977966'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/02958343/ply'
    save_path = '../data/CorrData/DST_part3d/recon/n03977966'
else:
    raise NotImplementedError

if not os.path.exists(save_path):
    os.makedirs(save_path)

instance_ids = os.listdir(imgs_path)

# min_verts = 100000
# for instance_id in instance_ids:
#     if '.' in instance_id:
#         continue
#     point_fn = os.path.join(points_path, f'{instance_id}.points.ply')
#     v, _, n, c = pcu.load_mesh_vfnc(point_fn)
#     print('v.shape: ', v.shape)
#     min_verts = min(min_verts, v.shape[0])
#
# print('min_verts: ', min_verts)
# exit(0)

idx = 0
for instance_id in instance_ids:
    if cat_type == 'bicycle':
        point_fn = os.path.join(points_path, f'{instance_id}.points.ply.npy')
        v = np.load(point_fn)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(v)
        alpha = 0.03
        alpha_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
        o3d.io.write_triangle_mesh(os.path.join(save_path, f"{instance_id}_recon_mesh.ply"), alpha_mesh)
        continue
    point_fn = os.path.join(points_path, f'{instance_id}.points.ply')
    if not os.path.exists(point_fn):
        print('no')
        continue
    v, _, n, c = pcu.load_mesh_vfnc(point_fn)
    print(v.shape, n.shape)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    pcd.normals = o3d.utility.Vector3dVector(n)

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    o3d.io.write_triangle_mesh(os.path.join(save_path, f"{instance_id}_recon_mesh.ply"), poisson_mesh)

    idx += 1
    print(idx, 'done')
