import sys
sys.path.append('../nemo/lib')
import torch
import numpy as np
import os
from PIL import Image, ImageDraw
import BboxTools as bbt
from MeshUtils import *
from pytorch3d.renderer import PerspectiveCameras
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.cameras import look_at_view_transform

cat_type = input('category type: ')
device = 'cuda:0'

if cat_type == 'car':
    annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/car'
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple/train/car'
    meshs_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02958343'
elif cat_type == 'plane':
    annos_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Annotations/train/aeroplane'
    imgs_path = '/ccvl/net/ccvl15/jiahao/DST/DST-pose-fix-distance/Data_simple/train/aeroplane'
    meshs_path = '/mnt/sde/angtian/data/ShapeNet/ShapeNetCore_v2/02691156'
else:
    raise NotImplementedError
save_path = '../visual/DiffusionQuality'

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

if not os.path.exists(save_path):
    os.makedirs(save_path)

instance_ids = os.listdir(imgs_path)
print('instance number: ', len(instance_ids))

for instance_id in instance_ids:
    instance_path = os.path.join(save_path, f'{instance_id}')
    if not os.path.exists(instance_path):
        os.makedirs(instance_path)
    # load mesh
    mesh_fn = os.path.join(meshs_path, instance_id, 'models', 'model_normalized.obj')

    verts, faces_idx, _ = load_obj(mesh_fn, device=device)
    # print(verts.shape)
    vert_middle = verts.max(dim=0)[0] + verts.min(dim=0)[0]
    vert_range = verts.max(dim=0)[0] - verts.min(dim=0)[0]
    scale = (vert_range ** 2).sum() ** 0.5
    # print('scale: ', scale, 'middle: ', vert_middle)
    faces = faces_idx.verts_idx
    verts_features = torch.ones_like(verts)[None]  # (1, V, 3)
    textures = Textures(verts_features=verts_features.to(device))

    meshes = Meshes(
        verts=[verts.to(device)],
        faces=[faces.to(device)],
        textures=textures
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
        offset = - vert_middle.cpu().numpy() / 2

        meshes_update = meshes
        meshes_update = meshes_update.update_padded(meshes.verts_padded() + torch.from_numpy(offset).to(device).float())

        R, T = look_at_view_transform(distance, elevation, azimuth, device=device, degrees=False)
        R = torch.bmm(R, rotation_theta(float(camera_rotation), device_=device))

        image = phong_renderer(meshes_world=meshes_update.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()

        image = np.array((image / image.max()) * 255).astype(np.uint8)

        mixed_image = (image * 0.6 + img * 0.4).astype(np.uint8)
        Image.fromarray(mixed_image).save(os.path.join(instance_path, f'{idx}.jpg'))
