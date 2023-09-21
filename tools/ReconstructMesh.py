# need "blender" env
import os
import open3d as o3d
import point_cloud_utils as pcu

cat_type = input('category type: ')
mode = input('mode: ')

if cat_type == 'car':
    imgs_path = f'/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/{mode}/car'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/02958343/ply'
    save_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/car'
elif cat_type == 'plane':
    imgs_path = f'/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple_512x512/{mode}/aeroplane'
    points_path = '/home/chuanruo/canonical-capsules/data/customShapeNet/02691156/ply'
    save_path = '/mnt/sde/angtian/data/ShapeNet/Reconstruct/aeroplane'
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
    if '.' in instance_id:
        continue
    point_fn = os.path.join(points_path, f'{instance_id}.points.ply')
    v, _, n, c = pcu.load_mesh_vfnc(point_fn)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(v)
    pcd.normals = o3d.utility.Vector3dVector(n)

    poisson_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    o3d.io.write_triangle_mesh(os.path.join(save_path, f"{instance_id}_recon_mesh.ply"), poisson_mesh)

    idx += 1
    print(idx, 'done')
