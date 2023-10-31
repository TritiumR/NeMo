import sys

sys.path.append('../nemo/lib')
import torch
import numpy as np
import os
import json
from PIL import Image, ImageDraw
import BboxTools as bbt
from MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.renderer import AmbientLights
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.cameras import look_at_view_transform
import point_cloud_utils as pcu
import trimesh


def get_meshes_ply(fn):
    verts, faces, _, _ = pcu.load_mesh_vfnc(fn)

    # faces
    faces = torch.from_numpy(faces.astype(np.int32))

    # normalize
    vert_middle = (verts.max(axis=0) + verts.min(axis=0)) / 2
    if instance_id in self.ray_list:
        vert_middle[1] += 0.05
    if instance_id in self.up_list:
        vert_middle[1] -= 0.08
    vert_scale = ((verts.max(axis=0) - verts.min(axis=0)) ** 2).sum() ** 0.5
    verts = verts - vert_middle
    verts = verts / vert_scale
    verts = torch.from_numpy(verts.astype(np.float32))

    return verts, faces


def get_meshes_ori(fn):
    verts, faces_idx, _ = load_obj(fn)
    faces = faces_idx.verts_idx

    # normalize
    vert_middle = (verts.max(dim=0)[0] + verts.min(dim=0)[0]) / 2
    vert_scale = ((verts.max(dim=0)[0] - verts.min(dim=0)[0]) ** 2).sum() ** 0.5
    verts = verts - vert_middle
    verts = verts / vert_scale

    return verts, faces


def get_meshes_glb(fn):
    scene = trimesh.load(fn)
    geometries = list(scene.geometry.values())
    geometry = geometries[0]
    print(geometry)

    return verts, faces


device = 'cuda:0'
category = input('category: ')

if category == 'airliner':
    focal_length = 800
elif category == 'police':
    focal_length = 660
elif category == 'bench':
    focal_length = 640
else:
    raise NotImplementedError

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
lights = AmbientLights(device=device)

# prepare camera
cameras = PerspectiveCameras(focal_length=focal_length,
                             principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                             image_size=(render_image_size,), device=device, in_ndc=False)

rasterizer = MeshRasterizer(
    cameras=cameras,
    raster_settings=raster_settings
)

phong_renderer = MeshRenderer(
    rasterizer=MeshRasterizer(
        cameras=cameras,
        raster_settings=raster_settings
    ),
    shader=HardPhongShader(device=device, lights=lights, cameras=cameras),
)


if category == 'airliner':
    cate_id = 'n02690373'
    part_list = ['1c27d282735f81211063b9885ddcbb1', '1d96d1c7cfb1085e61f1ef59130c405d',
                 '1de008320c90274d366b1ebd023111a8', '4ad92be763c2ded8fca1f1143bb6bc17',
                 '4fbdfec0f9ee078dc1ccec171a275967', '7f2d03635180db2137678474be485ca',
                 '7f4a0b23f1256c879a6e43b878d5b335', '8adc6a0f45a1ef2e71d03b466c72ce41',
                 '48bcce07b0baf689d9e6f00e848ea18', '66a32714d2344d1bf52a658ce0ec2c1']
    part_names = ['engine', 'fuselarge', 'wing', 'vertical_stabilizer', 'wheel', 'horizontal_stabilizer']
elif category == 'police':
    cate_id = 'n03977966'
    part_list = ['1a7125aefa9af6b6597505fd7d99b613', '45e69263902d77304dde7b6e74a2cede',
                 '275df71b6258e818597505fd7d99b613', '479f89af38e88bc9715e04edb8af9c53'
                 '45186c083231f2207b5338996083748c', '511962626501e4abf500cc506a763c18',
                  '498e4295b3aeca9fefddc097534e4553', '5389c96e84e9e0582b1e8dc2f1faa8cb',
                  '7492ced6cb6289c556de8db8652eec4e', '9511b5ded804a33f597505fd7d99b613',
                  'a5d32daf96820ca5f63ee8a34069b7c5', 'e90a136270c03eebaaafd94b9f216ef6']
    part_names = ['wheel', 'door', 'front_trunk', 'back_trunk', 'frame', 'rearview']
elif category == 'bench':
    cate_id = 'n03891251'
    part_list = ['1a40eaf5919b1b3f3eaa2b95b99dae6', '1aa15011153c5d6aa64b59533813e6d6',
                 '1b1cffcc9435c559f155d75bbf62b80', '1b9ddee986099bb78880edc6251fa529',
                 '1b80cd42474a990ccd8655d05e2f9e02', '1b78416210cbdcf1b184e775cf66758c',
                 '1be83cdaf803fa3b827358da75ee7655', '1bf5b1fa54aeec162701a18243b45d3',
                 '1c310698c57a3d378fd6b27be619556b', '1dfcce44c2f98d2c81e7c6cfefba0d68']
    part_names = ['arm', 'backrest', 'beam', 'seat', 'leg']
else:
    raise NotImplementedError

print('part_names: ', part_names)
root_path = '../data/CorrData/DST_part3d'
mesh_path = os.path.join(root_path, 'cad', cate_id)
imgs_path = os.path.join(root_path, 'train', cate_id)
annos_path = os.path.join(root_path, 'train', cate_id)
part_path = os.path.join(root_path, 'part', cate_id)
save_path = os.path.join(root_path, 'segment', cate_id)
vis_path = os.path.join(root_path, 'vis_segment', cate_id)

for instance_id in part_list:
    # load mesh
    mesh_fn = os.path.join(mesh_path, instance_id + '.obj')
    # print('mesh_fn: ', mesh_fn)
    verts = None
    faces = None
    if os.path.exists(mesh_fn):
        verts, faces = get_meshes_ori(mesh_fn)
    mesh_fn = os.path.join(mesh_path, instance_id + '.ply')
    if os.path.exists(mesh_fn):
        verts, faces = get_meshes_ply(mesh_fn)
    mesh_fn = os.path.join(mesh_path, instance_id + '.glb')
    if os.path.exists(mesh_fn):
        verts, faces = get_meshes_glb(mesh_fn)
    if verts is None:
        print('no mesh')
        continue

    # make mesh
    verts_features = torch.ones_like(verts)[None]
    textures = Textures(verts_features=verts_features.to(device))
    meshes = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
    )

    # load part annotation
    part_mesh_list = []
    part_fn = os.path.join(part_path, instance_id + '.json')
    with open(part_fn, 'r') as f:
        part_anno = json.load(f)
        for part_id, part_name in enumerate(part_names):
            if part_name not in part_anno or len(part_anno[part_name]) == 0:
                part_mesh_list.append(None)
                continue
            part_idx = part_anno[part_name]
            print('len(part_idx): ', len(part_idx))

            # make part verts
            part_verts = verts[part_idx]

            # make part faces
            vert_idx = dict()
            for idx in part_idx:
                vert_idx[idx] = len(vert_idx)
            part_faces = []
            for face in faces:
                v1, v2, v3 = face
                v1 = v1.item()
                v2 = v2.item()
                v3 = v3.item()
                if v1 in part_idx and v2 in part_idx and v3 in part_idx:
                    face[0] = vert_idx[v1]
                    face[1] = vert_idx[v2]
                    face[2] = vert_idx[v3]
                    part_faces.append(face)
            part_faces = torch.from_numpy(np.array(part_faces, dtype=np.int32))

            # make part mesh
            part_features = torch.zeros_like(part_verts)[None]
            textures = Textures(verts_features=part_features.to(device))
            part_mesh = Meshes(
                verts=[part_verts.to(device)],
                faces=[part_faces.to(device)],
                textures=textures
            )

            part_mesh_list.append(part_mesh)

    p_num = len(part_names)

    img_path = os.path.join(imgs_path, instance_id, 'image_minigpt4_1008')
    anno_path = os.path.join(annos_path, instance_id, 'annotation')
    segment_path = os.path.join(save_path, instance_id)
    vis_segment_path = os.path.join(vis_path, instance_id)
    if not os.path.exists(segment_path):
        os.makedirs(segment_path)
    if not os.path.exists(vis_segment_path):
        os.makedirs(vis_segment_path)

    for img_name in os.listdir(img_path):
        name = img_name.split('_')[0]

        anno_fn = os.path.join(anno_path, name + '.npy')
        anno = np.load(anno_fn, allow_pickle=True)[()]
        distance = anno['dist']
        elevation = np.pi / 2 - anno['phi']
        azimuth = anno['theta'] + np.pi / 2
        theta = anno['camera_rotation']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=device, degrees=False)
        R = torch.bmm(R, rotation_theta(float(theta), device_=device))

        depth_whole = rasterizer(meshes.clone(), R=R, T=T).zbuf
        depth_whole = depth_whole[0, ..., 0].detach().squeeze().cpu().numpy()
        depth_image_list = []
        for part_id in range(p_num):
            depth_part = np.ones_like(depth_whole) * np.inf
            part_meshes = part_mesh_list[part_id]
            if part_meshes is None:
                depth_image_list.append(depth_part)
            else:
                part_depth = rasterizer(meshes_world=part_meshes.clone(), R=R, T=T).zbuf

                part_depth_image = part_depth[0, ..., 0].detach().squeeze().cpu().numpy()
                part_depth_image = np.where(part_depth_image == -1, np.inf, part_depth_image)
                depth_part = np.minimum(depth_part, part_depth_image)
                depth_image_list.append(depth_part)

        depth_image_list.append(depth_whole)
        seg_mask = np.argmin(depth_image_list, axis=0)
        print('seg_mask: ', seg_mask.shape, seg_mask.min(), seg_mask.max())

        vis_seg_mask = seg_mask * 255 / seg_mask.max()

        seg_mask = seg_mask.astype(np.uint8)
        vis_seg_mask = vis_seg_mask.astype(np.uint8)

        save_segment_path = os.path.join(segment_path, name + '.png')
        save_vis_segment_path = os.path.join(vis_segment_path, name + '_vis.png')
        Image.fromarray(seg_mask).save(save_segment_path)
        Image.fromarray(vis_seg_mask).save(save_vis_segment_path)
        print('saved to ', save_segment_path)

