import torch
import torch.nn as nn
import numpy as np

try:
    from pytorch3d.structures import Meshes

    use_textures = True
except:
    from pytorch3d.structures import Meshes

    use_textures = False

try:
    from VoGE.Meshes import GaussianMeshesNaive as GaussianMesh
    from VoGE.Converter.Converters import naive_vertices_converter

    enable_voge = True
except:
    enable_voge = False

from pytorch3d.renderer import MeshRasterizer
from pytorch3d.transforms import Transform3d
from sklearn.neighbors import KDTree

from nemo.utils import (
    forward_interpolate,
    forward_interpolate_voge,
    pre_process_mesh_pascal,
    vertex_memory_to_face_memory,
    campos_to_R_T,
)


def func_reselect(meshes, indexs, **kwargs):
    verts_ = [meshes._verts_list[i] for i in indexs]
    faces_ = [meshes._faces_list[i] for i in indexs]
    meshes_out = Meshes(verts=verts_, faces=faces_).to(meshes.device)
    return meshes_out, meshes_out.verts_padded()


def MeshInterpolateModule(*args, **kwargs):
    rasterizer = kwargs.get('rasterizer')
    if isinstance(rasterizer, MeshRasterizer):
        return MeshInterpolateModuleMesh(*args, **kwargs)
    else:
        assert enable_voge
        return MeshInterpolateModuleVoGE(*args, **kwargs)


def rotation_matrix(azimuth, elevation):
    # Create the azimuth rotation matrix
    Rz = torch.stack([
        torch.cos(azimuth), -torch.sin(azimuth), torch.tensor(0.),
        torch.sin(azimuth), torch.cos(azimuth), torch.tensor(0.),
        torch.tensor(0.), torch.tensor(0.), torch.tensor(1.)
    ]).reshape(3, 3)

    # Create the elevation rotation matrix
    Rx = torch.stack([
        torch.tensor(1.), torch.tensor(0.), torch.tensor(0.),
        torch.tensor(0.), torch.cos(elevation), -torch.sin(elevation),
        torch.tensor(0.), torch.sin(elevation), torch.cos(elevation)
    ]).reshape(3, 3)

    # Combine the rotations
    R = torch.mm(Rx, Rz)

    return R


class MeshInterpolateModuleVoGE(nn.Module):
    def __init__(self, vertices, faces, memory_bank, rasterizer, post_process=None, off_set_mesh=False):
        super(MeshInterpolateModuleVoGE, self).__init__()

        # Convert memory features of vertices to faces
        self.memory = None
        self.update_memory(memory_bank=memory_bank,)

        # Preprocess convert meshes in PASCAL3d+ standard to Pytorch3D
        verts = pre_process_mesh_pascal(vertices)

        self.meshes = GaussianMesh(*naive_vertices_converter(verts, faces, percentage=0.5))

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, ):
        self.memory = memory_bank

    def to(self, *args, **kwargs):
        if 'device' in kwargs.keys():
            device = kwargs['device']
        else:
            device = args[0]
        super(MeshInterpolateModuleVoGE, self).to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.memory = self.memory.to(device)
        self.meshes = self.meshes.to(device)
        return self

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(self, campos, theta, deform_verts=None, **kwargs):
        R, T = campos_to_R_T(campos, theta, device=campos.device, )

        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)
        else:
            meshes = self.meshes
        get = forward_interpolate_voge(R, T, meshes, self.memory.repeat(R.shape[0], 1), rasterizer=self.rasterizer, )

        if self.post_process is not None:
            get = self.post_process(get)
        return get


class MeshInterpolateModuleMesh(nn.Module):
    def __init__(
        self,
        vertices,
        faces,
        memory_bank,
        rasterizer,
        post_process=None,
        off_set_mesh=False,
        interpolate_index=None,
        features=None,
    ):
        super().__init__()

        # interpolate features from memory_bank
        if interpolate_index is not None:
            interpolate_feature = []

            for mesh_id in range(len(interpolate_index)):
                sample_index = interpolate_index[mesh_id]
                vertex = vertices[mesh_id]
                sample_vertex = vertex[sample_index]
                kdtree = KDTree(sample_vertex)
                dist, nearest_idx = kdtree.query(vertex, k=3)
                # softmax dist as weight
                # print('dist: ', dist.shape)
                dist = torch.from_numpy(dist).to(memory_bank.device) + 1e-4
                dist = dist.type(torch.float32)
                weight = torch.softmax(1 / dist, dim=1)
                # print('weight: ', weight[1])
                # print('dist: ', dist.shape)
                # interpolate
                nearest_feature = [memory_bank[nearest] for nearest in nearest_idx]
                # print('nearest_feature: ', torch.stack(nearest_feature, dim=0).shape)
                # print('dist: ', weight.unsqueeze(-1).shape)
                feature = (torch.stack(nearest_feature, dim=0) * weight.unsqueeze(-1)).sum(dim=1)
                feature = feature / torch.norm(feature, dim=1, keepdim=True)
                interpolate_feature.append(feature)
                # print('interpolate_feature: ', interpolate_feature[-1].shape)

            memory_bank = interpolate_feature

        if features is not None:
            memory_bank = features

        # Convert memory features of vertices to faces
        self.faces = faces
        self.face_memory = None
        self.update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple meshes at same time
        verts = vertices

        # Create Pytorch3D meshes
        self.meshes = Meshes(verts=verts, faces=faces, textures=None)

        # Device is used during theta to R
        self.rasterizer = rasterizer
        self.post_process = post_process
        self.off_set_mesh = off_set_mesh

    def update_memory(self, memory_bank, faces=None):
        if faces is None:
            faces = self.faces
        # Convert memory features of vertices to faces
        self.face_memory = [
                vertex_memory_to_face_memory(m, f).to(m.device)
                for m, f in zip(memory_bank, faces)
            ]

        # print('face_memory: ', self.face_memory[0].shape, 'face: ', faces[0].shape)

    def to(self, *args, **kwargs):
        if "device" in kwargs.keys():
            device = kwargs["device"]
        else:
            device = args[0]
        super().to(device)
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)
        self.face_memory = [memory.to(device) for memory in self.face_memory]
        self.meshes = self.meshes.to(device)
        return self

    def update_rasterizer(self, rasterizer):
        device = self.rasterizer.cameras.device
        self.rasterizer = rasterizer
        self.rasterizer.cameras = self.rasterizer.cameras.to(device)

    def cuda(self, device=None):
        return self.to(torch.device("cuda"))

    def forward(self, campos, theta, blur_radius=0, deform_verts=None, mode="bilinear", indexs=None, part_poses=None, **kwargs):
        if indexs is not None:
            meshes, _ = func_reselect(self.meshes, indexs)
            face_memory = torch.cat([self.face_memory[idx] for idx in indexs], dim=0)
        else:
            meshes = self.meshes
            face_memory = torch.cat(self.face_memory, dim=0)
        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)

        device = meshes.device
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)
        R = R.to(device)
        T = T.to(device)

        if part_poses is not None:
            vert_list = []
            face_list = []
            for idx in range(len(campos)):
                offset = part_poses['offset'][idx][None]
                xscale = part_poses['xscale'][idx]
                yscale = part_poses['yscale'][idx]
                zscale = part_poses['zscale'][idx]
                azimuth = part_poses['azimuth'][idx]
                elevation = part_poses['elevation'][idx]
                rotate = rotation_matrix(azimuth, elevation)

                # transform = Transform3d(matrix=rotate.to(device), device=device)
                transform = Transform3d(device=device)
                transform = transform.scale(x=xscale.to(device), y=yscale.to(device), z=zscale.to(device))
                transform = transform.rotate(rotate.to(device))
                transform = transform.translate(offset.to(device))

                # print('transform done')
                verts = meshes._verts_list[0]
                faces = meshes._faces_list[0]
                verts = transform.transform_points(verts)
                vert_list.append(verts)
                face_list.append(faces)

            meshes = Meshes(verts=vert_list, faces=face_list).to(meshes.device)
            # exit(0)

        n_cam = campos.shape[0]
        # print('n_cam: ', n_cam)
        if n_cam > 1:
            get = forward_interpolate(
                R,
                T,
                meshes,
                face_memory.repeat(n_cam, 1, 1),
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )
        else:
            get = forward_interpolate(
                R,
                T,
                meshes,
                face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )

        # print('get: ', get.shape)
        if self.post_process is not None:
            get = self.post_process(get)
        return get

    def forward_seperate(self, campos, theta, blur_radius=0, deform_verts=None, mode="bilinear", **kwargs):
        gets = []
        for index in range(len(self.meshes._verts_list)):
            verts_ = [self.meshes._verts_list[index]]
            faces_ = [self.meshes._faces_list[index]]
            meshes = Meshes(verts=verts_, faces=faces_).to(self.meshes.device)
            face_memory = torch.cat([self.face_memory[index]], dim=0)
            if self.off_set_mesh:
                meshes = self.meshes.offset_verts(deform_verts)

            device = meshes.device
            R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)
            R = R.to(device)
            T = T.to(device)

            n_cam = campos.shape[0]
            if n_cam > 1:
                get = forward_interpolate(
                    R,
                    T,
                    meshes,
                    face_memory.repeat(n_cam, 1, 1),
                    rasterizer=self.rasterizer,
                    blur_radius=blur_radius,
                    mode=mode,
                )
            else:
                get = forward_interpolate(
                    R,
                    T,
                    meshes,
                    face_memory,
                    rasterizer=self.rasterizer,
                    blur_radius=blur_radius,
                    mode=mode,
                )

            gets.append(get)

        gets = torch.stack(gets, dim=0)
        # print('gets: ', gets.shape)
        if self.post_process is not None:
            gets = self.post_process(gets)
        return gets

    def forward_whole(self, campos, theta, blur_radius=0, deform_verts=None, mode="bilinear", part_poses=None, **kwargs):
        meshes = self.meshes
        face_memory = torch.cat(self.face_memory, dim=0)
        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)

        device = meshes.device
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)
        R = R.to(device)
        T = T.to(device)

        if part_poses is not None:
            vert_list = []
            face_list = []
            for idx in range(len(campos)):
                offsets = part_poses['offset'][idx]
                xscales = part_poses['xscale'][idx]
                # print('xscales: ', xscales)
                yscales = part_poses['yscale'][idx]
                zscales = part_poses['zscale'][idx]
                azimuths = part_poses['azimuth'][idx]
                elevations = part_poses['elevation'][idx]
                whole_vert = []
                whole_face = []
                for part_id in range(len(offsets)):
                    offset = offsets[part_id][None]
                    xscale = xscales[part_id]
                    yscale = yscales[part_id]
                    zscale = zscales[part_id]
                    azimuth = azimuths[part_id]
                    elevation = elevations[part_id]
                    # print('azimuth: ', azimuth, 'elevation: ', elevation)
                    rotate = rotation_matrix(azimuth, elevation)

                    transform = Transform3d(device=device)
                    transform = transform.scale(x=xscale.to(device), y=yscale.to(device), z=zscale.to(device))
                    transform = transform.rotate(rotate.to(device))
                    transform = transform.translate(offset.to(device))

                    # print('transform done')
                    verts = meshes[part_id]._verts_list[0]
                    faces = meshes[part_id]._faces_list[0]
                    verts = transform.transform_points(verts)
                    faces = faces + len(whole_vert)
                    # print('verts: ', verts.shape, verts[0])
                    # print('faces: ', faces.shape, faces[0], faces.max())
                    whole_vert.extend(verts)
                    whole_face.extend(faces)

                whole_vert = torch.stack(whole_vert, dim=0)
                whole_face = torch.stack(whole_face, dim=0)
                # print('whole_vert: ', whole_vert.shape)
                # print('whole_face: ', whole_face.shape)
                # print('face_max: ', whole_face.max())
                vert_list.append(whole_vert)
                face_list.append(whole_face)

            meshes = Meshes(verts=vert_list, faces=face_list).to(meshes.device)

        n_cam = campos.shape[0]
        if n_cam > 1:
            get = forward_interpolate(
                R,
                T,
                meshes,
                face_memory.repeat(n_cam, 1, 1),
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )
        else:
            get = forward_interpolate(
                R,
                T,
                meshes,
                face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )

        # print('get: ', get.shape)
        if self.post_process is not None:
            get = self.post_process(get)
        return get
