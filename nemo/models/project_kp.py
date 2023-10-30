import torch
import numpy as np
from pytorch3d.renderer import RasterizationSettings, MeshRasterizer, PerspectiveCameras, look_at_view_transform, \
    camera_position_from_spherical_angles, PointLights, MeshRenderer, HardPhongShader
from nemo.utils import rotation_theta
from pytorch3d.structures import Meshes
from pytorch3d.ops.interp_face_attrs import interpolate_face_attributes
from pytorch3d.renderer import TexturesVertex as Textures
from pytorch3d.transforms import Transform3d
import os
from PIL import Image
# if True:
try:
    from VoGE.Renderer import GaussianRenderSettings, GaussianRenderer
    from VoGE.Meshes import GaussianMeshesNaive as GaussianMesh
    from VoGE.Converter.Converters import naive_vertices_converter
    from VoGE.Utils import ind_fill

    enable_voge = True
except:
    enable_voge = False


def to_tensor(val):
    if isinstance(val, torch.Tensor):
        return val[None] if len(val.shape) == 2 else val
    elif isinstance(val, list):
        return [(t if isinstance(val, torch.Tensor) else torch.from_numpy(t)) for t in val]
    else:
        get = torch.from_numpy(val)
        return get[None] if len(get.shape) == 2 else get


def func_single(meshes, **kwargs):
    return meshes, meshes.verts_padded()


def func_reselect(meshes, indexs, **kwargs):
    verts_ = [meshes._verts_list[i] for i in indexs]
    faces_ = [meshes._faces_list[i] for i in indexs]
    meshes_out = Meshes(verts=verts_, faces=faces_).to(meshes.device)
    return meshes_out, meshes_out.verts_padded()


def rotation_matrix(azimuth, elevation):
    # Create the azimuth rotation matrix
    Rz = torch.tensor([
        [torch.cos(azimuth), -torch.sin(azimuth), 0],
        [torch.sin(azimuth), torch.cos(azimuth), 0],
        [0, 0, 1]
    ])

    # Create the elevation rotation matrix
    Rx = torch.tensor([
        [1, 0, 0],
        [0, torch.cos(elevation), -torch.sin(elevation)],
        [0, torch.sin(elevation), torch.cos(elevation)]
    ])

    # Combine the rotations
    R = torch.mm(Rx, Rz)

    return R


class PackedRaster():
    def __init__(self, raster_configs, object_mesh, mesh_mode='single', device='cpu'):
        """
        raster_configs: dict, include:
                {
                    'type': 'near',
                    'use_degree': False,
                    'image_size',
                    'down_rate',
                    'focal_length': 3000,
                    'blur_radius': 0,
                    'kp_vis_thr': 0.25
                }
        mesh_mode: ['single', 'multi', 'deformable']
        """
        raster_type = raster_configs.get('type', 'near')

        self.raster_type = raster_type
        self.use_degree = raster_configs.get('use_degree', False)
        self.raster_configs = raster_configs

        self.mesh_mode = mesh_mode
        self.kwargs = raster_configs
        self.focal_length = 3200

        image_size = raster_configs.get('image_size')
        feature_size = (
        image_size[0] // raster_configs.get('down_rate'), image_size[1] // raster_configs.get('down_rate'))
        cameras = PerspectiveCameras(
            focal_length=raster_configs.get('focal_length', self.focal_length) / raster_configs.get('down_rate'),
            principal_point=((feature_size[1] // 2, feature_size[0] // 2,),), image_size=(feature_size,), in_ndc=False,
            device=device)
        self.cameras = cameras
        self.down_rate = raster_configs.get('down_rate')

        if raster_type == 'near' or raster_type == 'triangle':
            raster_setting = RasterizationSettings(image_size=feature_size,
                                                   blur_radius=raster_configs.get('blur_radius', 0.0)) # attention !!!
            self.raster = MeshRasterizer(raster_settings=raster_setting, cameras=cameras)

            if isinstance(object_mesh, Meshes):
                self.meshes = object_mesh.to(device)
            elif isinstance(object_mesh, dict):
                self.meshes = Meshes(verts=to_tensor(object_mesh['verts']), faces=to_tensor(object_mesh['faces'])).to(
                    device)
            else:
                self.meshes = Meshes(verts=to_tensor(object_mesh[0]), faces=to_tensor(object_mesh[1])).to(device)
        if raster_type == 'voge' or raster_type == 'vogew':
            assert enable_voge, 'VoGE must be install to utilize voge-nemo.'
            self.kp_vis_thr = raster_configs.get('kp_vis_thr', 0.25)
            render_setting = GaussianRenderSettings(image_size=feature_size, max_point_per_bin=-1,
                                                    max_assign=raster_configs.get('max_assign', 20))
            self.render = GaussianRenderer(render_settings=render_setting, cameras=cameras).to(device)
            self.meshes = GaussianMesh(*naive_vertices_converter(*object_mesh, percentage=0.5)).to(device)

    def step(self):
        if self.raster_type == 'voge' or self.raster_type == 'vogew':
            self.kp_vis_thr -= 0.001 / 5

    def get_verts_recent(self, ):
        if self.raster_type == 'voge' or self.raster_type == 'vogew':
            return self.meshes.verts[None]
        if self.mesh_mode == 'single':
            return self.meshes.verts_padded()

    def __call__(self, azim, elev, dist, theta, **kwargs):
        R, T = look_at_view_transform(dist=dist, azim=azim, elev=elev, degrees=self.use_degree,
                                      device=self.cameras.device)
        R = torch.bmm(R, rotation_theta(theta, device_=self.cameras.device))

        if self.mesh_mode == 'single' and self.raster_type == 'near':
            this_cameras = self.cameras.clone()
            this_cameras.R = R
            this_cameras.T = T

            if kwargs.get('principal', None) is not None:
                this_cameras._N = R.shape[0]
                this_cameras.principal_point = kwargs.get('principal', None).to(self.cameras.device) / self.down_rate

            # image = None
            # if kwargs.get('visual', None) is not None:
            #     # render config
            #     render_image_size = (512, 512)
            #
            #     raster_settings = RasterizationSettings(
            #         image_size=render_image_size,
            #         blur_radius=0.0,
            #         faces_per_pixel=1,
            #         bin_size=0
            #     )
            #     # We can add a point light in front of the object.
            #     lights = PointLights(device=self.cameras.device, location=((2.0, 2.0, -2.0),))
            #
            #     # prepare camera
            #     cameras = PerspectiveCameras(focal_length=1.0 * 3200,
            #                                  principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
            #                                  image_size=(render_image_size,), device=self.cameras.device, in_ndc=False)
            #
            #     phong_renderer = MeshRenderer(
            #         rasterizer=MeshRasterizer(
            #             cameras=cameras,
            #             raster_settings=raster_settings
            #         ),
            #         shader=HardPhongShader(device=self.cameras.device, lights=lights, cameras=cameras),
            #     )
            #
            #     image = phong_renderer(meshes_world=self.meshes.clone(), R=R, T=T)
            #     image = image[0, ..., :3].detach().squeeze().cpu().numpy()

            return get_one_standard(self.raster, this_cameras, self.meshes,  **kwargs,
                                    **self.kwargs) # + (image,)
        else:
            if kwargs.get('principal', None) is not None:
                self.render.cameras._N = R.shape[0]
                self.render.cameras.principal_point = kwargs.get('principal', None).to(
                    self.cameras.device) / self.down_rate

            if self.raster_type == 'voge':
                # Return voge.fragments
                frag = self.render(self.meshes, R=R, T=T)
                get_dict = frag.to_dict()
                get_dict['start_idx'] = torch.arange(frag.vert_index.shape[0]).to(frag.vert_index.device)
                # if torch.any( torch.nn.functional.relu(1 - frag.vert_weight.sum(3).view(R.shape[0], -1)).sum(1) < 1 ):
                return get_dict
            if self.raster_type == 'vogew':
                # Return voge.fragments
                frag = self.render(self.meshes, R=R, T=T)
                get_dict = frag.to_dict()
                get_dict['start_idx'] = torch.arange(frag.vert_index.shape[0]).to(frag.vert_index.device)
                get_weight = torch.zeros((*frag.vert_index.shape[0:-1], self.meshes.verts.shape[0] + 1),
                                         device=frag.vert_index.device)
                ind = frag.vert_index.long() - torch.arange(frag.vert_index.shape[0]).to(frag.vert_index.device)[:,
                                               None, None, None] * self.meshes.verts.shape[0]
                ind[ind < 0] = -1
                # weight_ = torch.cat((torch.zeros((*frag.vert_index.shape[0:-1], 1), device=frag.vert_index.device), frag.vert_weight, ), dim=-1)
                ind += 1
                get_weight = ind_fill(get_weight, ind, frag.vert_weight, dim=3)
                return get_weight[..., 1:]

    def visual_pose(self, mesh_idx, img, pose, folder, error):
        render_image_size = (512, 512)

        mesh, _ = func_reselect(self.meshes, [mesh_idx])

        verts_features = torch.ones_like(mesh.verts_padded())  # (1, V, 3)
        textures = Textures(verts_features=verts_features.to(self.cameras.device))

        mesh = Meshes(verts=mesh.verts_padded(), faces=mesh.faces_padded(), textures=textures).to(self.cameras.device)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=self.cameras.device, location=((2.0, 2.0, -2.0),))

        # prepare camera
        cameras = PerspectiveCameras(focal_length=self.focal_length,
                                     principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                                     image_size=(render_image_size,), device=self.cameras.device, in_ndc=False)

        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.cameras.device, lights=lights, cameras=cameras),
        )

        distance = pose['distance']
        elevation = pose['elevation']
        azimuth = pose['azimuth']
        theta = pose['theta']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.cameras.device, degrees=False)
        R = torch.bmm(R, rotation_theta(theta, device_=self.cameras.device))

        image = phong_renderer(meshes_world=mesh.clone(), R=R, T=T)
        image = image[0, ..., :3].detach().squeeze().cpu().numpy()

        if not img.shape[2] == 3:
            img = img.permute(1, 2, 0).numpy()
        else:
            img = (img.numpy() / 255)
        # print('img: ', img.shape, img.max(), img.min())

        # print('image: ', image.shape)
        # print('img: ', img.shape)
        # print('image: ', image.max(), image.min())
        # print('img: ', img.max(), img.min())

        saved_path = './visual/Pose/' + folder
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        mixed_image = ((image * 0.6 + img * 0.4) * 255).astype(np.uint8)
        mixed_image = Image.fromarray(mixed_image)
        mixed_image.save(os.path.join(saved_path, f'error{np.array(error).mean():.4f}.jpg'))

    def visual_part_pose(self, part_verts, part_faces, img, pose, part_pose, folder, error):
        render_image_size = (512, 512)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )
        # We can add a point light in front of the object.
        lights = PointLights(device=self.cameras.device, location=((2.0, 2.0, -2.0),))

        # prepare camera
        cameras = PerspectiveCameras(focal_length=self.focal_length,
                                     principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                                     image_size=(render_image_size,), device=self.cameras.device, in_ndc=False)

        phong_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=raster_settings
            ),
            shader=HardPhongShader(device=self.cameras.device, lights=lights, cameras=cameras),
        )

        distance = pose['distance']
        elevation = pose['elevation']
        azimuth = pose['azimuth']
        theta = pose['theta']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.cameras.device, degrees=False)
        R = torch.bmm(R, rotation_theta(theta, device_=self.cameras.device))

        device = self.cameras.device
        image_list = []
        for part_id, part_vert in enumerate(part_verts):
            offset = part_pose['offset'][part_id][None]
            azimuth = part_pose['azimuth'][part_id]
            elevation = part_pose['elevation'][part_id]
            xscale = part_pose['xscale'][part_id]
            yscale = part_pose['yscale'][part_id]
            zscale = part_pose['zscale'][part_id]

            rotate = rotation_matrix(azimuth, elevation)

            # # print('offset: ', offset)
            # # print('rotate: ', rotate)
            # # print('scale: ', scale)
            #
            # rotate = torch.cat([torch.cat([rotate, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
            transform = Transform3d(device=device)
            transform = transform.scale(x=xscale.to(device), y=yscale.to(device), z=zscale.to(device))
            transform = transform.rotate(rotate.to(device))
            transform = transform.translate(offset.to(device))

            part_vert = transform.transform_points(torch.from_numpy(part_vert).to(device))
            part_face = torch.from_numpy(part_faces[part_id]).to(device)

            verts_features = torch.ones_like(part_vert)[None]  # (1, V, 3)
            textures = Textures(verts_features=verts_features.to(device))

            mesh = Meshes(verts=[part_vert], faces=[part_face], textures=textures).to(device)

            image = phong_renderer(meshes_world=mesh.clone(), R=R, T=T)
            image = image[0, ..., :3].detach().squeeze().cpu().numpy()
            image_list.append(image)

        saved_path = './visual/PartPose/' + folder
        if not os.path.exists(saved_path):
            os.makedirs(saved_path)

        img = img.permute(1, 2, 0).numpy()

        mixed_image = np.zeros_like(img)
        weight = 1. / len(image_list)
        for image in image_list:
            mixed_image += image * 2 * weight

        mixed_image = ((mixed_image * (1 - weight) + img * weight) * 255).astype(np.uint8)
        # clip the rgb value
        mixed_image = np.clip(mixed_image, 0, 255)
        mixed_image = Image.fromarray(mixed_image)
        mixed_image.save(os.path.join(saved_path, f'error{np.array(error).mean():.4f}.jpg'))

    def get_segment(self, part_verts, part_faces, pose, part_pose, part_names):
        render_image_size = (512, 512)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        # prepare camera
        cameras = PerspectiveCameras(focal_length=self.focal_length,
                                     principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                                     image_size=(render_image_size,), device=self.cameras.device, in_ndc=False)

        distance = pose['distance']
        elevation = pose['elevation']
        azimuth = pose['azimuth']
        theta = pose['theta']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.cameras.device, degrees=False)
        R = torch.bmm(R, rotation_theta(theta, device_=self.cameras.device))

        device = self.cameras.device
        part_xverts = []
        part_xfaces = []
        part_xtexs = []
        for part_id, part_vert in enumerate(part_verts):
            offset = part_pose['offset'][part_id][None]
            azimuth = part_pose['azimuth'][part_id]
            elevation = part_pose['elevation'][part_id]
            xscale = part_pose['xscale'][part_id]
            yscale = part_pose['yscale'][part_id]
            zscale = part_pose['zscale'][part_id]


            rotate = rotation_matrix(azimuth, elevation)
            # rotate = torch.cat([torch.cat([rotate, torch.Tensor([0, 0, 0])[:, None]], dim=1), torch.Tensor([0, 0, 0, 1])[None]], dim=0)
            transform = Transform3d(device=device)
            transform = transform.scale(x=xscale.to(device), y=yscale.to(device), z=zscale.to(device))
            transform = transform.rotate(rotate.to(device))
            transform = transform.translate(offset.to(device))

            part_vert = transform.transform_points(torch.from_numpy(part_vert).to(device))
            part_face = torch.from_numpy(part_faces[part_id]).to(device)
            part_name = part_names[part_id]
            color = torch.tensor([1, 0, 0])
            verts_features = torch.ones_like(part_vert) * color.to(device) # (1, V, 3)

            part_xfaces.extend(part_face + len(part_xverts))
            part_xverts.extend(part_vert)
            part_xtexs.extend(verts_features)

        part_xverts = torch.stack(part_xverts, dim=0)
        part_xfaces = torch.stack(part_xfaces, dim=0)
        part_xtexs = torch.stack(part_xtexs, dim=0)[None]
        textures = Textures(verts_features=part_xtexs.to(device))

        mesh = Meshes(verts=[part_xverts], faces=[part_xfaces], textures=textures).to(device)

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        frag = rasterizer(meshes_world=mesh.clone(), R=R, T=T)
        face_attr = part_xtexs[0][mesh.faces_packed().long()]
        get = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords, face_attr).squeeze(0).squeeze(2)

        return get

    def get_segment_depth(self, part_verts, part_faces, pose, part_pose):
        render_image_size = (512, 512)

        raster_settings = RasterizationSettings(
            image_size=render_image_size,
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=0
        )

        cameras = PerspectiveCameras(focal_length=self.focal_length,
                                     principal_point=((render_image_size[1] // 2, render_image_size[0] // 2),),
                                     image_size=(render_image_size,), device=self.cameras.device, in_ndc=False)

        rasterizer = MeshRasterizer(
            cameras=cameras,
            raster_settings=raster_settings
        )
        if not isinstance(pose['distance'], float):
            distance = pose['distance'].float()
            elevation = pose['elevation'].float()
            azimuth = pose['azimuth'].float()
            theta = pose['theta'].float()
        else:
            distance = pose['distance']
            elevation = pose['elevation']
            azimuth = pose['azimuth']
            theta = pose['theta']

        R, T = look_at_view_transform(distance, elevation, azimuth, device=self.cameras.device, degrees=False)
        R = torch.bmm(R, rotation_theta(theta, device_=self.cameras.device))

        # get whole depth
        device = self.cameras.device
        part_xverts = []
        part_xfaces = []
        part_xverts_list = []
        part_xfaces_list = []
        for part_id, part_vert in enumerate(part_verts):
            offset = part_pose['offset'][part_id][None]
            azimuth = part_pose['azimuth'][part_id]
            elevation = part_pose['elevation'][part_id]
            xscale = part_pose['xscale'][part_id]
            yscale = part_pose['yscale'][part_id]
            zscale = part_pose['zscale'][part_id]

            rotate = rotation_matrix(azimuth, elevation)
            transform = Transform3d(device=device)
            transform = transform.scale(x=xscale.to(device), y=yscale.to(device), z=zscale.to(device))
            transform = transform.rotate(rotate.to(device))
            transform = transform.translate(offset.to(device))

            part_vert = transform.transform_points(torch.from_numpy(part_vert).to(device))
            part_face = torch.from_numpy(part_faces[part_id]).to(device)

            part_xverts_list.append(part_vert)
            part_xfaces_list.append(part_face)

            part_xfaces.extend(part_face + len(part_xverts))
            part_xverts.extend(part_vert)

        part_xverts = torch.stack(part_xverts, dim=0)
        part_xfaces = torch.stack(part_xfaces, dim=0)
        part_xtexs = torch.zeros_like(part_xverts)[None]
        textures = Textures(verts_features=part_xtexs.to(device))

        mesh = Meshes(verts=[part_xverts], faces=[part_xfaces], textures=textures).to(device)

        depth_whole = rasterizer(mesh.clone(), R=R, T=T).zbuf
        depth_whole = depth_whole[0, ..., 0].detach().squeeze().cpu().numpy()

        depth_image_list = []
        for part_id, part_vert in enumerate(part_verts):
            depth_part = np.ones_like(depth_whole) * np.inf

            part_vert = part_xverts_list[part_id]
            part_face = part_xfaces_list[part_id]
            part_verts_features = torch.zeros_like(part_vert)[None]
            part_textures = Textures(verts_features=part_verts_features.to(device))

            part_meshes = Meshes(
                verts=[part_vert.to(device)],
                faces=[part_face.to(device)],
                textures=part_textures
            )

            part_depth = rasterizer(meshes_world=part_meshes.clone(), R=R, T=T).zbuf

            part_depth_image = part_depth[0, ..., 0].detach().squeeze().cpu().numpy()
            part_depth_image = np.where(part_depth_image == -1, np.inf, part_depth_image)
            depth_part = np.minimum(depth_part, part_depth_image)
            depth_image_list.append(depth_part)

        depth_image_list.append(depth_whole)
        depth_image_list = np.array(depth_image_list)
        # print('depth_image_list: ', depth_image_list.shape)
        seg_mask = np.argmin(depth_image_list, axis=0)
        # print('seg_mask: ', seg_mask.shape)

        return seg_mask


def get_one_standard(raster, camera, mesh, func_of_mesh=func_single, restrict_to_boundary=True, dist_thr=1e-3,
                     **kwargs):
    # dist_thr => NeMo original repo: cal_occ_one_image: eps
    mesh_, verts_ = func_of_mesh(mesh, **kwargs)

    R = camera.R
    T = camera.T

    # Calculate the camera location
    cam_loc = -torch.matmul(torch.inverse(R), T[..., None])[:, :, 0]

    # (B, K, 2)
    project_verts = camera.transform_points(verts_)[..., 0:2].flip(-1)
    # Don't know why, hack. Checked by visualization
    project_verts = 2 * camera.principal_point[:, None].float().flip(-1) - project_verts

    # (B, K)
    inner_mask = torch.min(camera.image_size.unsqueeze(1) > torch.ones_like(project_verts), dim=-1)[0] & \
                 torch.min(0 < torch.ones_like(project_verts), dim=-1)[0]

    if restrict_to_boundary:
        # image_size -> (h, w)
        project_verts = torch.min(project_verts, (camera.image_size.unsqueeze(1) - 1) * torch.ones_like(project_verts))
        project_verts = torch.max(project_verts, torch.zeros_like(project_verts))

    raster.cameras = camera
    frag = raster(mesh_.extend(R.shape[0]) if mesh_._N == 1 else mesh_, R=R, T=T)
    true_dist_per_vert = (cam_loc[:, None] - verts_).pow(2).sum(-1).pow(.5)
    face_dist = torch.gather(true_dist_per_vert[:, None].expand(-1, mesh_.faces_padded().shape[1], -1), dim=2, index=mesh_.faces_padded().expand(true_dist_per_vert.shape[0], -1, -1).clamp(min=0))
    
    if func_of_mesh is func_reselect:
        face_dist = torch.cat([face_dist[i, :mesh_.num_faces_per_mesh()[i]] for i in range(R.shape[0])], dim=0)

    # (B, 1, H, W)
    # depth_ = frag.zbuf[..., 0][:, None]
    depth_ = interpolate_face_attributes(frag.pix_to_face, frag.bary_coords, face_dist.view(-1, 3, 1))[:, :, :, 0, 0][:, None]

    grid = project_verts[:, None] / torch.Tensor(list(depth_.shape[2:])).to(project_verts.device) * 2 - 1

    sampled_dist_per_vert = torch.nn.functional.grid_sample(depth_, grid.flip(-1), align_corners=False, mode='nearest')[:, 0, 0, :]

    vis_mask = torch.abs(sampled_dist_per_vert - true_dist_per_vert) < dist_thr


    if func_of_mesh is func_reselect:
        for i in range(R.shape[0]):
            vis_mask[i, mesh_._num_verts_per_mesh[i]:] = False

    return project_verts, vis_mask & inner_mask


