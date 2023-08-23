import torch
import torch.nn as nn

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


class MeshInterpolateModuleVoGE(nn.Module):
    def __init__(self, vertices, faces, memory_bank, rasterizer, post_process=None, off_set_mesh=False):
        super(MeshInterpolateModuleVoGE, self).__init__()

        # Convert memory features of vertices to faces
        self.memory = None
        self.update_memory(memory_bank=memory_bank,)

        self.n_mesh = 1
        
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
        interpolate_index=None
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
                dist = torch.from_numpy(dist).to(memory_bank.device) + 1e-2
                dist = dist.type(torch.float32)
                weight = torch.softmax(1 / dist, dim=1)
                # print('weight: ', weight[1])
                # print('dist: ', dist.shape)
                # interpolate
                nearest_feature = [memory_bank[nearest] for nearest in nearest_idx]
                # print('nearest_feature: ', torch.stack(nearest_feature, dim=0).shape)
                # print('dist: ', weight.unsqueeze(-1).shape)
                interpolate_feature.append((torch.stack(nearest_feature, dim=0) * weight.unsqueeze(-1)).sum(dim=1))
                # print('interpolate_feature: ', interpolate_feature[-1].shape)

            memory_bank = interpolate_feature

        # Convert memory features of vertices to faces
        self.faces = faces
        self.face_memory = None
        self.update_memory(memory_bank=memory_bank, faces=faces)

        # Support multiple meshes at same time
        self.n_mesh = len(vertices)
        # # Preprocess convert mesh in PASCAL3d+ standard to Pytorch3D
        # verts = [pre_process_mesh_pascal(t) for t in vertices]
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

    def forward(self, campos, theta, blur_radius=0, deform_verts=None, mode="bilinear", indexs=None, **kwargs):
        R, T = campos_to_R_T(campos, theta, device=campos.device, **kwargs)

        if indexs is not None:
            meshes, _ = func_reselect(self.meshes, indexs)
            face_memory = torch.cat([self.face_memory[idx] for idx in indexs], dim=0)
        else:
            meshes = self.meshes
            face_memory = torch.cat(self.face_memory, dim=0)
        if self.off_set_mesh:
            meshes = self.meshes.offset_verts(deform_verts)

        n_cam = campos.shape[0]
        if n_cam > 1 and self.n_mesh > 1:
            get = forward_interpolate(
                R,
                T,
                meshes,
                face_memory,
                rasterizer=self.rasterizer,
                blur_radius=blur_radius,
                mode=mode,
            )
        elif n_cam > 1 and self.n_mesh == 1:
            get = forward_interpolate(
                R,
                T,
                meshes.extend(campos.shape[0]),
                face_memory.repeat(campos.shape[0], 1, 1).view(
                    -1, *self.face_memory.shape[1:]
                ),
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
