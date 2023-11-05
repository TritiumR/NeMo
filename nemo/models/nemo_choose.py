import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from PIL import Image

from nemo.models.base_model import BaseModel
from nemo.models.feature_banks import mask_remove_near, remove_near_vertices_dist
from nemo.models.mesh_interpolate_module import MeshInterpolateModule
from nemo.models.solve_pose import pre_compute_kp_coords
from nemo.models.solve_pose import solve_pose
from nemo.models.batch_solve_pose import get_pre_render_samples, loss_curve_part, batch_only_scale, part_initialization
from nemo.models.batch_solve_pose import solve_pose as batch_solve_pose
from nemo.models.batch_solve_pose import solve_part_pose as batch_solve_part_pose
from nemo.models.batch_solve_pose import solve_part_whole as batch_solve_part_whole
from nemo.utils import center_crop_fun
from nemo.utils import construct_class_by_name
from nemo.utils import get_param_samples
from nemo.utils import normalize_features
from nemo.utils import pose_error, iou, pre_process_mesh_pascal, load_off
from nemo.utils.pascal3d_utils import IMAGE_SIZES

from nemo.datasets.synthetic_shapenet import MeshLoader, PartsLoader

from nemo.models.project_kp import PackedRaster, func_reselect, to_tensor
# from nemo.lib.MeshUtils import *


class NeMo(BaseModel):
    def __init__(
            self,
            cfg,
            cate,
            mode,
            backbone,
            memory_bank,
            num_noise,
            max_group,
            down_sample_rate,
            training,
            inference,
            proj_mode='runtime',
            checkpoint=None,
            transforms=[],
            device="cuda:0",
            **kwargs
    ):
        super().__init__(cfg, cate, mode, checkpoint, transforms, ['loss', 'loss_main', 'loss_reg'], device)
        self.net_params = backbone
        self.memory_bank_params = memory_bank
        self.num_noise = num_noise
        self.max_group = max_group
        self.down_sample_rate = down_sample_rate
        self.training_params = training
        self.inference_params = inference
        self.dataset_config = cfg.dataset
        self.accumulate_steps = 0
        if cate in ['car', 'aeroplane']:
            self.num_verts = 3072
        else:
            self.num_verts = 1024
        self.visual_kp = cfg.training.visual_kp
        self.visual_mesh = cfg.training.visual_mesh
        self.visual_pose = cfg.inference.visual_pose
        if mode == 'test':
            self.folder = cfg.args.save_dir.split('/')[-1]
        self.ori_mesh = cfg.ori_mesh

        if self.ori_mesh:
            self.dataset_config['ori_mesh'] = True

        if cfg.task == 'correlation_marking':
            self.build()
            return

        if cate == 'car':
            self.chosen_id = '4d22bfe3097f63236436916a86a90ed7'
            # self.chosen_ids = ['5edaef36af2826762bf75f4335c3829b', 'e0bf76446d320aa9aa69dfdc5532bb13', 'ea0d722f312d1262e0095a904eb93647',
            #                    'aeac711326961038939aeffada2c0c5', 'd2064d59beb9f24e8810bd18ea9969c']
            self.chosen_ids = ['372ceb40210589f8f500cc506a763c18', ]
        elif cate == 'aeroplane':
            self.chosen_id = '1d63eb2b1f78aa88acf77e718d93f3e1'
            self.chosen_ids = ['1d63eb2b1f78aa88acf77e718d93f3e1', ]
        elif cate == 'boat':
            self.chosen_id = '246335e0dfc3a0ea834ac3b5e36b95c'
            self.chosen_ids = ['246335e0dfc3a0ea834ac3b5e36b95c']
        elif cate == 'bicycle':
            self.chosen_id = '91k7HKqdM9'
            self.chosen_ids = ['91k7HKqdM9', 'LGj1dKhcY1', 'Mlb3AKfw61']
        elif cate == 'airliner':
            self.chosen_id = '22831bc32bd744d3f06dea205edf9704'
            self.chosen_ids = ['22831bc32bd744d3f06dea205edf9704']

        self.parts_loader = PartsLoader(self.dataset_config, cate=cate, chosen_ids=self.chosen_ids)
        self.mesh_loader = MeshLoader(self.dataset_config, cate=cate)
        self.anno_parts = self.mesh_loader.anno_parts
        self.build()

        self.raster_conf = {
            'image_size': (self.dataset_config.image_sizes, self.dataset_config.image_sizes),
            **self.training_params.kp_projecter
        }
        if self.raster_conf['down_rate'] == -1:
            self.raster_conf['down_rate'] = self.net.module.net_stride
        self.net.module.kwargs['n_vert'] = self.num_verts

        if self.ori_mesh:
            self.projector = PackedRaster(self.raster_conf, self.mesh_loader.get_ori_mesh_listed(), device='cuda')
        else:
            self.projector = PackedRaster(self.raster_conf, self.mesh_loader.get_mesh_listed(), device='cuda')

    def build(self):
        if self.mode == "train":
            self._build_train()
        else:
            self._build_inference()

    def _build_train(self):
        self.n_gpus = torch.cuda.device_count()
        if self.training_params.separate_bank:
            self.ext_gpu = f"cuda:{self.n_gpus - 1}"
        else:
            self.ext_gpu = ""

        net = construct_class_by_name(**self.net_params)
        if self.training_params.separate_bank:
            self.net = nn.DataParallel(net, device_ids=[i for i in range(self.n_gpus - 1)]).cuda()
        else:
            self.net = nn.DataParallel(net).cuda()

        memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            output_size=self.num_verts + self.num_noise * self.max_group,
            num_pos=self.num_verts,
            num_noise=self.num_noise)
        if self.training_params.separate_bank:
            self.memory_bank = memory_bank.cuda(self.ext_gpu)
        else:
            self.memory_bank = memory_bank.cuda()

        self.optim = construct_class_by_name(
            **self.training_params.optimizer, params=self.net.parameters())
        self.scheduler = construct_class_by_name(
            **self.training_params.scheduler, optimizer=self.optim)

    def step_scheduler(self):
        self.scheduler.step()
        if self.training_params.kp_projecter.type == 'voge' or self.training_params.kp_projecter.type == 'vogew':
            self.projector.step()

    def train(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()

        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()
        mesh_index = [self.mesh_loader.mesh_name_dict[t] for t in sample['instance_id']]

        kwargs_ = dict(principal=sample['principal'], func_of_mesh=func_reselect, indexs=mesh_index) if 'principal' in sample.keys() else dict(func_of_mesh=func_reselect, indexs=mesh_index)
        get_mesh_index = self.mesh_loader.get_index_list(mesh_index).cuda()

        with torch.no_grad():
            kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(), dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(), **kwargs_)
            kp = torch.gather(kp, dim=1, index=get_mesh_index[..., None].expand(-1, -1, 2))
            kpvis = torch.gather(kpvis, dim=1, index=get_mesh_index)

        features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True,)

        if self.training_params.separate_bank:
            get, y_idx, noise_sim = self.memory_bank(
                features.to(self.ext_gpu), index.to(self.ext_gpu), kpvis.to(self.ext_gpu)
            )
        else:
            get, y_idx, noise_sim = self.memory_bank(features, index, kpvis)

        if 'voge' in self.projector.raster_type:
            kpvis = kpvis > self.projector.kp_vis_thr

        get /= self.training_params.T

        kappas = {'pos': self.training_params.get('weight_pos', 0),
                  'near': self.training_params.get('weight_near', 1e5),
                  'clutter': -math.log(self.training_params.weight_noise)}

        # The default manner in VoGE-NeMo
        if self.training_params.remove_near_mode == 'vert':
            with torch.no_grad():
                verts_ = func_reselect(self.projector.meshes, mesh_index)[1]
                vert_ = torch.gather(verts_, dim=1, index=get_mesh_index[..., None].expand(-1, -1, 3))

                vert_dis = (vert_.unsqueeze(1) - vert_.unsqueeze(2)).pow(2).sum(-1).pow(.5)

                mask_distance_legal = remove_near_vertices_dist(
                    vert_dis,
                    thr=self.training_params.distance_thr,
                    num_neg=self.num_noise * self.max_group,
                    kappas=kappas,
                )
                if mask_distance_legal.shape[0] != get.shape[0]:
                    mask_distance_legal = mask_distance_legal.expand(get.shape[0], -1, -1).contiguous()
        # The default manner in original-NeMo
        else:
            mask_distance_legal = mask_remove_near(
                kp,
                thr=self.training_params.distance_thr
                    * torch.ones((img.shape[0],), dtype=torch.float32).cuda(),
                num_neg=self.num_noise * self.max_group,
                dtype_template=get,
                kappas=kappas,
            )
        if self.training_params.get('training_loss_type', 'nemo') == 'nemo':
            loss_main = nn.CrossEntropyLoss(reduction="none").cuda()(
                (get.view(-1, get.shape[2]) - mask_distance_legal.view(-1, get.shape[2]))[kpvis.view(-1), :],
                y_idx.view(-1)[kpvis.view(-1)],
            )
            loss_main = torch.mean(loss_main)
        elif self.training_params.get('training_loss_type', 'nemo') == 'kl_alan':
            loss_main = torch.mean(
                (get.view(-1, get.shape[2]) * mask_distance_legal.view(-1, get.shape[2]))[kpvis.view(-1), :])

        if self.num_noise > 0:
            loss_reg = torch.mean(noise_sim) * self.training_params.loss_reg_weight
            loss = loss_main + loss_reg
        else:
            loss_reg = torch.zeros(1)
            loss = loss_main

        loss.backward()

        self.accumulate_steps += 1
        if self.accumulate_steps % self.training_params.train_accumulate == 0:
            self.optim.step()
            self.optim.zero_grad()

        self.loss_trackers['loss'].append(loss.item())
        self.loss_trackers['loss_main'].append(loss_main.item())
        self.loss_trackers['loss_reg'].append(loss_reg.item())

        return {'loss': loss.item(), 'loss_main': loss_main.item(), 'loss_reg': loss_reg.item()}

    def _build_inference(self):
        self.net = construct_class_by_name(**self.net_params)
        self.net = nn.DataParallel(self.net).to(self.device)
        self.net.load_state_dict(self.checkpoint["state"])

        self.memory_bank = construct_class_by_name(
            **self.memory_bank_params,
            output_size=self.num_verts,
            num_pos=self.num_verts,
            num_noise=0
        ).to(self.device)

        with torch.no_grad():
            self.memory_bank.memory.copy_(
                self.checkpoint["memory"][0: self.memory_bank.memory.shape[0]]
            )
        memory = (
            self.checkpoint["memory"][0: self.memory_bank.memory.shape[0]]
            .detach()
            .cpu()
            .numpy()
        )
        clutter = (
            self.checkpoint["memory"][self.memory_bank.memory.shape[0]:]
            .detach()
            .cpu()
            .numpy()
        )
        self.feature_bank = torch.from_numpy(memory)
        self.clutter_bank = torch.from_numpy(clutter).to(self.device)
        self.clutter_bank = normalize_features(
            torch.mean(self.clutter_bank, dim=0)
        ).unsqueeze(0)
        self.kp_features = self.checkpoint["memory"][
                           0: self.memory_bank.memory.shape[0]
                           ].to(self.device)

        if self.cfg.task == 'correlation_marking':
            return

        image_h, image_w = (self.dataset_config.image_sizes, self.dataset_config.image_sizes)
        map_shape = (image_h // self.down_sample_rate, image_w // self.down_sample_rate)

        if self.inference_params.cameras.get('image_size', 0) == -1:
            self.inference_params.cameras['image_size'] = (map_shape, )
        if self.inference_params.cameras.get('principal_point', 0) == -1:
            self.inference_params.cameras['principal_point'] = ((map_shape[1] // 2, map_shape[0] // 2), )
            print('principal_point: ', self.inference_params.cameras['principal_point'])
        if self.inference_params.cameras.get('focal_length', None) is not None:
            self.inference_params.cameras['focal_length'] = self.inference_params.cameras['focal_length'] / self.down_sample_rate
            print('focal_length: ', self.inference_params.cameras['focal_length'])

        cameras = construct_class_by_name(**self.inference_params.cameras, device=self.device)
        raster_settings = construct_class_by_name(
            **self.inference_params.raster_settings, image_size=map_shape
        )
        if self.inference_params.rasterizer.class_name == 'VoGE.Renderer.GaussianRenderer':
            self.rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, render_settings=raster_settings
            )
        else:
            self.rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, raster_settings=raster_settings
            )

        if self.ori_mesh:
            chosen_verts, chosen_faces = self.mesh_loader.get_ori_meshes(self.chosen_id)
        else:
            chosen_verts, chosen_faces = self.mesh_loader.get_meshes(self.chosen_id)

        xvert = [chosen_verts]
        xface = [chosen_faces]
        mesh_index = [self.mesh_loader.mesh_name_dict[self.chosen_id]]
        get_mesh_index = self.mesh_loader.get_index_list(mesh_index).cuda()

        if self.cfg.part_consistency:
            self.nearest_pairs = []
            verts_with_feature = chosen_verts[get_mesh_index[0]]
            kdtree = KDTree(verts_with_feature)
            dist, near_idx = kdtree.query(verts_with_feature, k=2)
            dist = dist[:, 1]
            near_idx = near_idx[:, 1]
            nearest = np.argwhere(dist < self.cfg.inference.dis_threshold)
            for idx in nearest:
                # print('nearest: ', idx, near_idx[idx])
                self.nearest_pairs.append((idx, near_idx[idx]))
                self.nearest_pairs.append((near_idx[idx], idx))

            print('len of pairs: ', len(self.nearest_pairs))

        self.inter_module = MeshInterpolateModule(
            xvert,
            xface,
            self.feature_bank,
            rasterizer=self.rasterizer,
            post_process=None,
            interpolate_index=get_mesh_index,
        ).to(self.device)

        # load chosen meshes
        chosen_verts = []
        chosen_faces = []
        mesh_indexes = []
        for chosen_id in self.chosen_ids:
            if self.ori_mesh:
                chosen_vert, chosen_face = self.mesh_loader.get_ori_meshes(chosen_id)
            else:
                chosen_vert, chosen_face = self.mesh_loader.get_meshes(chosen_id)
            chosen_verts.append(chosen_vert)
            chosen_faces.append(chosen_face)
            mesh_indexes.append(self.mesh_loader.mesh_name_dict[chosen_id])
            # print('mesh_index: ', mesh_indexes[-1])
        get_mesh_indexes = self.mesh_loader.get_index_list(mesh_indexes).cuda()

        # load parts and compute part features
        self.parts_feature = []
        self.parts_xvert = []
        self.parts_xface = []
        self.parts_offset = []
        self.parts_indexes_on_corr = []
        for idx, chosen_id in enumerate(self.chosen_ids):
            part_xvert = self.parts_loader.get_part_mesh(chosen_id)[0]
            part_xface = self.parts_loader.get_part_mesh(chosen_id)[1]
            part_off_set = self.parts_loader.get_offset(chosen_id)

            verts_with_feature = chosen_verts[idx][get_mesh_indexes[idx]]

            # interpolate part features from memory_bank
            part_feature = []
            part_indexes_on_corr = []
            kdtree = KDTree(verts_with_feature)
            for part_vert, off_set in zip(part_xvert, part_off_set):
                if len(part_vert) == 1:
                    part_feature.append(torch.zeros((1, 128)))
                    part_indexes_on_corr.append(np.zeros(1, dtype=np.int32) - 1)
                    continue
                part_vert = part_vert + off_set
                dist, nearest_idx = kdtree.query(part_vert, k=self.cfg.inference.nearest_k)
                # print('dist_min: ', dist.min())

                # print('dist: ', dist.shape, nearest_idx.shape)

                eps = self.cfg.inference.on_corr_eps
                indexes_on_corr = np.zeros(len(part_vert), dtype=np.int32) - 1
                indexes_on_corr[dist[:, 0] < eps] = nearest_idx[:, 0][dist[:, 0] < eps]
                print('num of indexes_on_corr: ', len(indexes_on_corr[indexes_on_corr != -1]))
                part_indexes_on_corr.append(indexes_on_corr)

                dist = torch.from_numpy(dist).to(self.feature_bank.device) + 1e-4
                dist = dist.type(torch.float32)
                weight = torch.softmax(1 / dist, dim=1)
                nearest_feature = [self.feature_bank[nearest] for nearest in nearest_idx]
                feature = (torch.stack(nearest_feature, dim=0) * weight.unsqueeze(-1)).sum(dim=1)
                feature = feature / feature.norm(dim=1).unsqueeze(-1)
                part_feature.append(feature)

            self.parts_xvert.append([torch.from_numpy(part_vert) for part_vert in part_xvert])
            self.parts_xface.append([torch.from_numpy(part_face) for part_face in part_xface])
            self.parts_feature.append(part_feature)
            self.parts_offset.append(part_off_set)
            self.parts_indexes_on_corr.append(part_indexes_on_corr)

        if self.cfg.task == 'part_locate':
            return

        (azimuth_samples,
            elevation_samples,
            theta_samples,
            distance_samples,
            px_samples,
            py_samples,
        ) = get_param_samples(self.cfg)

        self.init_mode = self.cfg.inference.get('init_mode', '3d_batch')

        if 'batch' in self.init_mode:
            self.feature_pre_rendered, self.cam_pos_pre_rendered, self.theta_pre_rendered = get_pre_render_samples(
                self.inter_module,
                azum_samples=azimuth_samples,
                elev_samples=elevation_samples,
                theta_samples=theta_samples,
                distance_samples=distance_samples,
                device=self.device
            )
            assert distance_samples.shape[0] == 1
            self.record_distance = distance_samples[0]

        else:
            self.poses, self.kp_coords, self.kp_vis = pre_compute_kp_coords(
                self.mesh_path,
                azimuth_samples=azimuth_samples,
                elevation_samples=elevation_samples,
                theta_samples=theta_samples,
                distance_samples=distance_samples,
            )

        print('build inference done')

    def evaluate(self, sample, debug=False):
        self.net.eval()

        ori_img = sample['img_ori'].numpy()
        sample = self.transforms(sample)
        img = sample['img'].cuda()

        mesh_index = [self.mesh_loader.mesh_name_dict[self.chosen_id]] * img.shape[0]

        kwargs_ = dict(indexs=mesh_index)

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)
        if 'batch' in self.init_mode:
            dof = int(self.init_mode.split('d_')[0])
            if dof == 6:
                principal = self.samples_principal
            elif ('principal' in sample.keys()) and self.cfg.inference.get('realign', False):
                principal = sample['principal'].float().to(self.device) / self.down_sample_rate
            else:
                principal = None

            preds = batch_solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.clutter_bank,
                cam_pos_pre_rendered=self.cam_pos_pre_rendered,
                theta_pre_rendered=self.theta_pre_rendered,
                feature_pre_rendered=self.feature_pre_rendered,
                device=self.device,
                principal=principal,
                distance_source=sample['distance'].to(feature_map.device) if dof == 3 else torch.ones(
                    feature_map.shape[0]).to(feature_map.device),
                distance_target=self.record_distance * torch.ones(feature_map.shape[0]).to(
                    feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
                pre_render=self.cfg.inference.get('pre_render', True),
                dof=dof,
                **kwargs_
            )
        else:
            assert len(img) == 1, "The batch size during validation should be 1"
            preds = solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.kp_features,
                self.clutter_bank,
                self.poses,
                self.kp_coords,
                self.kp_vis,
                debug=debug,
                device=self.device,
                **kwargs_
            )
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            if "azimuth" in sample and "elevation" in sample and "theta" in sample:
                pred["pose_error"] = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]},
                                                pred["final"][0])
        if self.visual_pose:
            for idx in range(len(img)):
                self.projector.visual_pose(mesh_index[idx], sample['img_ori'][idx], preds[idx]["final"][0],
                                           self.folder, preds[idx]["pose_error"])

        return preds

    def evaluate_part(self, sample, debug=False):
        self.net.eval()

        ori_img = sample['img_ori'].numpy()
        sample = self.transforms(sample)
        img = sample['img'].cuda()

        mesh_index = [self.mesh_loader.mesh_name_dict[self.chosen_id]] * img.shape[0]

        kwargs_ = dict(indexs=mesh_index)

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)
        if 'batch' in self.init_mode:
            dof = int(self.init_mode.split('d_')[0])
            if ('principal' in sample.keys()) and self.cfg.inference.get('realign', False):
                principal = sample['principal'].float().to(self.device) / self.down_sample_rate
            else:
                principal = None

            preds = batch_solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.clutter_bank,
                cam_pos_pre_rendered=self.cam_pos_pre_rendered,
                theta_pre_rendered=self.theta_pre_rendered,
                feature_pre_rendered=self.feature_pre_rendered,
                device=self.device,
                principal=principal,
                distance_source=sample['distance'].to(feature_map.device) if dof == 3 else torch.ones(
                    feature_map.shape[0]).to(feature_map.device),
                distance_target=self.record_distance * torch.ones(feature_map.shape[0]).to(
                    feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
                pre_render=self.cfg.inference.get('pre_render', True),
                dof=dof,
                **kwargs_
            )
        else:
            assert len(img) == 1, "The batch size during validation should be 1"
            preds = solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.kp_features,
                self.clutter_bank,
                self.poses,
                self.kp_coords,
                self.kp_vis,
                debug=debug,
                device=self.device,
                **kwargs_
            )
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            if "azimuth" in sample and "elevation" in sample and "theta" in sample:
                pred["pose_error"] = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]},
                                                pred["final"][0])
        if self.visual_pose:
            for idx in range(len(img)):
                self.projector.visual_pose(mesh_index[idx], sample['img_ori'][idx], preds[idx]["final"][0],
                                           self.folder, preds[idx]["pose_error"])

        distances = torch.from_numpy(np.array([pred["final"][0]['distance'] for pred in preds])).cuda()
        elevations = torch.from_numpy(np.array([pred["final"][0]['elevation'] for pred in preds])).cuda()
        azimuths = torch.from_numpy(np.array([pred["final"][0]['azimuth'] for pred in preds])).cuda()
        thetas = torch.from_numpy(np.array([pred["final"][0]['theta'] for pred in preds])).cuda()
        initial_pose = dict(
            distance=distances,
            elevation=elevations,
            azimuth=azimuths,
            theta=thetas,
        )

        chosen_indexes = []
        chosen_scales = []
        chosen_offsets = []
        offset = None
        for part_id, part_name in enumerate(self.parts_loader.get_name_listed()):
            min_loss = None
            min_idx = None
            min_scale = None
            min_offset = None
            max_score = None
            for chosen_idx in range(len(self.chosen_ids)):
                part_vert = [self.parts_xvert[chosen_idx][part_id]]
                part_face = [self.parts_xface[chosen_idx][part_id]]
                part_feature = [self.parts_feature[chosen_idx][part_id]]
                part_offsets = self.parts_offset[chosen_idx][part_id]

                if part_vert[0].shape[0] == 1:
                    print(self.chosen_ids[chosen_idx], 'has no ', part_name)
                    continue

                part_inter_module = MeshInterpolateModule(
                    part_vert,
                    part_face,
                    self.feature_bank,
                    rasterizer=self.rasterizer,
                    post_process=None,
                    interpolate_index=None,
                    features=part_feature,
                ).to(self.device)

                if self.cfg.part_initialization is True:
                    loss, offset, scale, score = part_initialization(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )
                else:
                    loss, scale, score = batch_only_scale(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )
                if max_score is None or score > max_score:
                    max_score = score
                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_idx = chosen_idx
                    min_scale = scale
                    min_offset = offset

            if max_score > 0:
                chosen_indexes.append(min_idx)
                chosen_scales.append(min_scale)
                chosen_offsets.append(min_offset)
            else:
                chosen_indexes.append(-1)

        parts_xvert = []
        parts_xface = []
        parts_feature = []
        parts_offset = []
        for idx, chosen_idx in enumerate(chosen_indexes):
            if chosen_idx != -1:
                parts_xvert.append(self.parts_xvert[chosen_idx][idx])
                parts_xface.append(self.parts_xface[chosen_idx][idx])
                parts_feature.append(self.parts_feature[chosen_idx][idx])
                parts_offset.append(self.parts_offset[chosen_idx][idx])

        # print('chosen_scales: ', chosen_scales)
        kwargs_ = dict(chosen_scales=chosen_scales, chosen_offsets=chosen_offsets)

        parts_inter_module = MeshInterpolateModule(
            parts_xvert,
            parts_xface,
            self.feature_bank,
            rasterizer=self.rasterizer,
            post_process=None,
            interpolate_index=None,
            features=parts_feature,
        ).to(self.device)

        part_preds = batch_solve_part_whole(
            self.cfg,
            feature_map,
            self.clutter_bank,
            parts_inter_module,
            parts_feature,
            initial_pose,
            parts_offset,
            **kwargs_
        )

        parts_xvert = []
        parts_xface = []
        for idx, name in enumerate(self.parts_loader.get_name_listed()):
            part_vert, part_face = self.parts_loader.get_ori_part(self.chosen_ids[chosen_indexes[idx]], name)
            parts_xvert.append(part_vert)
            parts_xface.append(part_face)

        for idx in range(len(img)):
            self.projector.visual_part_pose(parts_xvert, parts_xface, sample['img_ori'][idx], preds[idx]["final"][0],
                                            part_preds[idx]["final"], self.folder, preds[idx]["pose_error"])

        # exit(0)

        return preds, part_preds

    def evaluate_imagepart(self, sample, debug=False):
        self.net.eval()

        ori_img = sample['img_ori'].numpy()
        sample = self.transforms(sample)
        img = sample['img'].cuda()
        # print('img: ', img.shape)
        if img.shape[2] != 512 or img.shape[3] != 512:
            print('wrong img shape')
            return None


        mesh_index = [0] * img.shape[0]
        vis_mesh_index = [self.mesh_loader.mesh_name_dict[self.chosen_id]] * img.shape[0]
        print('mesh_index: ', mesh_index)

        kwargs_ = dict(indexs=mesh_index)

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        if self.cfg.use_pred and sample['pose_pred'] == 0:
            return None
        pose_pred = []
        if not self.cfg.use_pred:
            print('sample distance: ', sample['distance'])
            dof = int(self.init_mode.split('d_')[0])

            preds = batch_solve_pose(
                self.cfg,
                feature_map,
                self.inter_module,
                self.clutter_bank,
                cam_pos_pre_rendered=self.cam_pos_pre_rendered,
                theta_pre_rendered=self.theta_pre_rendered,
                feature_pre_rendered=self.feature_pre_rendered,
                device=self.device,
                principal=None,
                distance_source=sample['distance'].to(feature_map.device) if dof == 3 else None,
                distance_target=self.record_distance * torch.ones(feature_map.shape[0]).to(
                    feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
                pre_render=self.cfg.inference.get('pre_render', True),
                dof=dof,
                **kwargs_
            )
            if isinstance(preds, dict):
                preds = [preds]

            for i, pred in enumerate(preds):
                pose_pred.append(pred["final"][0])
                if "azimuth" in sample and "elevation" in sample and "theta" in sample:
                    pred["pose_error"] = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]},
                                                    pred["final"][0])
                    print('pose_error: ', pred["pose_error"])
                else:
                    pred["pose_error"] = np.random.rand()

            if self.visual_pose:
                for idx in range(len(img)):
                    self.projector.visual_pose(vis_mesh_index[idx], sample['img_ori'][idx], preds[idx]["final"][0],
                                               self.folder, preds[idx]["pose_error"])

            distances = torch.from_numpy(np.array([pred["final"][0]['distance'] for pred in preds])).cuda()
            elevations = torch.from_numpy(np.array([pred["final"][0]['elevation'] for pred in preds])).cuda()
            azimuths = torch.from_numpy(np.array([pred["final"][0]['azimuth'] for pred in preds])).cuda()
            thetas = torch.from_numpy(np.array([pred["final"][0]['theta'] for pred in preds])).cuda()
        else:
            preds = sample['pose_pred']
            if isinstance(preds, dict):
                preds = [preds]

            pose_pred = preds

            distances = torch.from_numpy(np.array([pred['distance'] for pred in preds])).cuda()
            elevations = torch.from_numpy(np.array([pred['elevation'] for pred in preds])).cuda()
            azimuths = torch.from_numpy(np.array([pred['azimuth'] for pred in preds])).cuda()
            thetas = torch.from_numpy(np.array([pred['theta'] for pred in preds])).cuda()

        print('distances: ', distances)
        initial_pose = dict(
            distance=distances,
            elevation=elevations,
            azimuth=azimuths,
            theta=thetas,
        )

        chosen_indexes = []
        chosen_scales = []
        chosen_offsets = []
        offset = None
        for part_id, part_name in enumerate(self.parts_loader.get_name_listed()):
            min_loss = None
            min_idx = None
            min_scale = None
            min_offset = None
            max_score = None
            for chosen_idx in range(len(self.chosen_ids)):
                part_vert = [self.parts_xvert[chosen_idx][part_id]]
                part_face = [self.parts_xface[chosen_idx][part_id]]
                part_feature = [self.parts_feature[chosen_idx][part_id]]
                part_offsets = self.parts_offset[chosen_idx][part_id]

                if part_vert[0].shape[0] == 1:
                    print(self.chosen_ids[chosen_idx], 'has no', part_name)
                    continue

                part_inter_module = MeshInterpolateModule(
                    part_vert,
                    part_face,
                    self.feature_bank,
                    rasterizer=self.rasterizer,
                    post_process=None,
                    interpolate_index=None,
                    features=part_feature,
                ).to(self.device)

                if self.cfg.part_initialization is True:
                    loss, offset, scale, score = part_initialization(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )
                else:
                    loss, scale, score = batch_only_scale(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )

                if max_score is None or score > max_score:
                    max_score = score

                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_idx = chosen_idx
                    min_scale = scale
                    min_offset = offset

            if max_score > 0:
                chosen_indexes.append(min_idx)
                chosen_scales.append(min_scale)
                if min_offset is not None:
                    chosen_offsets.append(min_offset[0])
            else:
                print('no part ', part_name)
                chosen_indexes.append(-1)

        parts_xvert = []
        parts_xface = []
        parts_feature = []
        parts_offset = []
        parts_name = []
        for idx, chosen_idx in enumerate(chosen_indexes):
            if chosen_idx != -1:
                parts_xvert.append(self.parts_xvert[chosen_idx][idx])
                parts_xface.append(self.parts_xface[chosen_idx][idx])
                parts_feature.append(self.parts_feature[chosen_idx][idx])
                parts_offset.append(self.parts_offset[chosen_idx][idx])
                parts_name.append(self.parts_loader.get_name_listed()[idx])

        kwargs_ = dict(chosen_scales=chosen_scales, chosen_offsets=chosen_offsets)

        if len(parts_name) > 0:
            if self.cfg.part_consistency is True:
                parts_indexes_on_corr = []
                for idx, chosen_idx in enumerate(chosen_indexes):
                    if chosen_idx != -1:
                        parts_indexes_on_corr.append(self.parts_indexes_on_corr[chosen_idx][idx])

                near_pairs = []
                for pair in self.nearest_pairs:
                    id1, id2 = pair
                    # print('0000')
                    index_offset_1 = 0
                    for part_id_1 in range(len(parts_name)):
                        if part_id_1 > 0:
                            index_offset_1 += len(parts_xvert[part_id_1 - 1])
                        if id1 in parts_indexes_on_corr[part_id_1]:
                            # print('1111')
                            # print('find: ', np.where(parts_indexes_on_corr[part_id_1] == id1)[0][0])
                            index_1 = np.where(parts_indexes_on_corr[part_id_1] == id1)[0][0] + index_offset_1
                            index_offset_2 = index_offset_1
                            for part_id_2 in range(part_id_1 + 1, len(parts_name)):
                                index_offset_2 += len(parts_xvert[part_id_2 - 1])
                                if id2 in parts_indexes_on_corr[part_id_2]:
                                    # print('2222')
                                    index_2 = np.where(parts_indexes_on_corr[part_id_2] == id2)[0][0] + index_offset_2
                                    near_pairs.append((index_1, index_2))
                                if id1 in parts_indexes_on_corr[part_id_2]:
                                    # print('2222')
                                    index_2 = np.where(parts_indexes_on_corr[part_id_2] == id1)[0][0] + index_offset_2
                                    if (index_1, index_2) not in near_pairs:
                                        near_pairs.append((index_1, index_2))

                print('near_pairs: ', len(near_pairs))
                kwargs_['near_pairs'] = near_pairs

            parts_inter_module = MeshInterpolateModule(
                parts_xvert,
                parts_xface,
                self.feature_bank,
                rasterizer=self.rasterizer,
                post_process=None,
                interpolate_index=None,
                features=parts_feature,
            ).to(self.device)

            part_preds = batch_solve_part_whole(
                self.cfg,
                feature_map,
                self.clutter_bank,
                parts_inter_module,
                parts_feature,
                initial_pose,
                parts_offset,
                **kwargs_
            )

            # parts_xvert = []
            # parts_xface = []
            # for idx, name in enumerate(self.parts_loader.get_name_listed()):
            #     if chosen_indexes[idx] != -1:
            #         part_vert, part_face = self.parts_loader.get_ori_part(self.chosen_ids[chosen_indexes[idx]], name)
            #         parts_xvert.append(part_vert)
            #         parts_xface.append(part_face)
            # #
            parts_xvert = [part_vert.numpy() for part_vert in parts_xvert]
            parts_xface = [part_face.numpy() for part_face in parts_xface]

            segment = self.projector.get_segment_depth(parts_xvert, parts_xface, pose_pred[0],
                                                       part_preds[0]["final"])
        else:
            segment = np.ones((512, 512)) * len(self.anno_parts)
        annotations = sample['seg']
        vis_imgs = sample['img_ori'].numpy()
        iou_dict = dict()
        print('segment: ', segment.min(), segment.max())

        img_mask = np.sum(vis_imgs[0], axis=2) > 0
        # segment[~img_mask] = -1

        anno = annotations[0].type(torch.int32)
        anno_compare = anno[img_mask]
        seg = torch.zeros_like(anno) + len(self.anno_parts)

        total_intersection = 0
        total_union = 0
        intersections = []
        unions = []
        for anno_id, name in enumerate(self.anno_parts):
            for part_id, part_name in enumerate(parts_name):
                if name in part_name:
                    seg[segment == part_id] = anno_id

            seg_compare = seg[img_mask]

            intersection = ((seg_compare == anno_id) & (anno_compare == anno_id)).sum()
            union = ((seg_compare == anno_id) | (anno_compare == anno_id)).sum()
            iou = intersection / union
            # print(f'{name} IOU: ', iou)
            iou_dict[name] = iou
            total_intersection += intersection
            total_union += union
            intersections.append(intersection)
            unions.append(union)

        seg_compare = seg[img_mask]
        bg_intersection = ((seg_compare == len(self.anno_parts)) & (anno_compare == len(self.anno_parts))).sum()
        bg_union = ((seg_compare == len(self.anno_parts)) | (anno_compare == len(self.anno_parts))).sum()
        total_intersection += bg_intersection
        total_union += bg_union
        bg_iou = bg_intersection / bg_union
        intersections.append(bg_intersection)
        unions.append(bg_union)
        iou_dict['bg'] = bg_iou

        miou = total_intersection / total_union
        iou_dict['mIoU'] = miou

        iou_dict['intersections'] = intersections
        iou_dict['unions'] = unions

        save_path = './visual/segment/' + self.folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vis_get = seg.clone() / seg.max() * 255
        vis_get = vis_get.detach().cpu().numpy().astype(np.uint8)
        vis_get = Image.fromarray(vis_get)
        vis_get.save(f'{save_path}/{miou}_seg.jpg')

        vis_img = vis_imgs[0].astype(np.uint8)
        vis_img = Image.fromarray(vis_img)
        vis_img.save(f'{save_path}/{miou}_ori.png')

        vis_anno = anno.clone() / anno.max() * 255
        vis_anno = vis_anno.numpy().astype(np.uint8)
        vis_anno = Image.fromarray(vis_anno)
        vis_anno.save(f'{save_path}/{miou}_gt.png')

        return iou_dict

    def evaluate_dst_part(self, sample, debug=False):
        self.net.eval()

        sample = self.transforms(sample)
        img = sample['img'].cuda()

        mesh_index = [0] * img.shape[0]
        vis_mesh_index = [self.mesh_loader.mesh_name_dict[self.chosen_id]] * img.shape[0]

        kwargs_ = dict(indexs=mesh_index)

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        pose_pred = []
        print('sample distance: ', sample['distance'])
        dof = int(self.init_mode.split('d_')[0])

        preds = batch_solve_pose(
            self.cfg,
            feature_map,
            self.inter_module,
            self.clutter_bank,
            cam_pos_pre_rendered=self.cam_pos_pre_rendered,
            theta_pre_rendered=self.theta_pre_rendered,
            feature_pre_rendered=self.feature_pre_rendered,
            device=self.device,
            principal=None,
            distance_source=sample['distance'].to(feature_map.device),
            distance_target=self.record_distance * torch.ones(feature_map.shape[0]).to(
                feature_map.device) if dof == 3 else torch.ones(feature_map.shape[0]).to(feature_map.device),
            pre_render=self.cfg.inference.get('pre_render', True),
            dof=dof,
            **kwargs_
        )
        if isinstance(preds, dict):
            preds = [preds]

        for i, pred in enumerate(preds):
            pose_pred.append(pred["final"][0])
            if "azimuth" in sample and "elevation" in sample and "theta" in sample:
                pred["pose_error"] = pose_error({k: sample[k][i] for k in ["azimuth", "elevation", "theta"]},
                                                pred["final"][0])
                print('pose_error: ', pred["pose_error"])
            else:
                pred["pose_error"] = np.random.rand()

        if self.visual_pose:
            for idx in range(len(img)):
                self.projector.visual_pose(vis_mesh_index[idx], sample['img_ori'][idx], preds[idx]["final"][0],
                                           self.folder, preds[idx]["pose_error"])

        distances = torch.from_numpy(np.array([pred["final"][0]['distance'] for pred in preds])).cuda()
        elevations = torch.from_numpy(np.array([pred["final"][0]['elevation'] for pred in preds])).cuda()
        azimuths = torch.from_numpy(np.array([pred["final"][0]['azimuth'] for pred in preds])).cuda()
        thetas = torch.from_numpy(np.array([pred["final"][0]['theta'] for pred in preds])).cuda()

        print('distances: ', distances)
        initial_pose = dict(
            distance=distances,
            elevation=elevations,
            azimuth=azimuths,
            theta=thetas,
        )

        chosen_indexes = []
        chosen_scales = []
        chosen_offsets = []
        offset = None
        for part_id, part_name in enumerate(self.parts_loader.get_name_listed()):
            min_loss = None
            min_idx = None
            min_scale = None
            min_offset = None
            max_score = None
            for chosen_idx in range(len(self.chosen_ids)):
                part_vert = [self.parts_xvert[chosen_idx][part_id]]
                part_face = [self.parts_xface[chosen_idx][part_id]]
                part_feature = [self.parts_feature[chosen_idx][part_id]]
                part_offsets = self.parts_offset[chosen_idx][part_id]

                if part_vert[0].shape[0] == 1:
                    print(self.chosen_ids[chosen_idx], 'has no', part_name)
                    continue

                part_inter_module = MeshInterpolateModule(
                    part_vert,
                    part_face,
                    self.feature_bank,
                    rasterizer=self.rasterizer,
                    post_process=None,
                    interpolate_index=None,
                    features=part_feature,
                ).to(self.device)

                if self.cfg.part_initialization is True:
                    loss, offset, scale, score = part_initialization(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )
                else:
                    loss, scale, score = batch_only_scale(
                        self.cfg,
                        feature_map,
                        self.clutter_bank,
                        part_inter_module,
                        part_feature,
                        initial_pose,
                        part_offsets,
                    )

                if max_score is None or score > max_score:
                    max_score = score

                if min_loss is None or loss < min_loss:
                    min_loss = loss
                    min_idx = chosen_idx
                    min_scale = scale
                    min_offset = offset

            if max_score > 0:
                chosen_indexes.append(min_idx)
                chosen_scales.append(min_scale)
                if min_offset is not None:
                    chosen_offsets.append(min_offset[0])
            else:
                print('no part ', part_name)
                chosen_indexes.append(-1)

        parts_xvert = []
        parts_xface = []
        parts_feature = []
        parts_offset = []
        parts_name = []
        for idx, chosen_idx in enumerate(chosen_indexes):
            if chosen_idx != -1:
                parts_xvert.append(self.parts_xvert[chosen_idx][idx])
                parts_xface.append(self.parts_xface[chosen_idx][idx])
                parts_feature.append(self.parts_feature[chosen_idx][idx])
                parts_offset.append(self.parts_offset[chosen_idx][idx])
                parts_name.append(self.parts_loader.get_name_listed()[idx])

        kwargs_ = dict(chosen_scales=chosen_scales, chosen_offsets=chosen_offsets)

        if len(parts_name) > 0:
            if self.cfg.part_consistency is True:
                parts_indexes_on_corr = []
                for idx, chosen_idx in enumerate(chosen_indexes):
                    if chosen_idx != -1:
                        parts_indexes_on_corr.append(self.parts_indexes_on_corr[chosen_idx][idx])

                near_pairs = []
                for pair in self.nearest_pairs:
                    id1, id2 = pair
                    index_offset_1 = 0
                    for part_id_1 in range(len(parts_name)):
                        if part_id_1 > 0:
                            index_offset_1 += len(parts_xvert[part_id_1 - 1])
                        if id1 in parts_indexes_on_corr[part_id_1]:
                            index_1 = np.where(parts_indexes_on_corr[part_id_1] == id1)[0][0] + index_offset_1
                            index_offset_2 = index_offset_1
                            for part_id_2 in range(part_id_1 + 1, len(parts_name)):
                                index_offset_2 += len(parts_xvert[part_id_2 - 1])
                                if id2 in parts_indexes_on_corr[part_id_2]:
                                    index_2 = np.where(parts_indexes_on_corr[part_id_2] == id2)[0][0] + index_offset_2
                                    near_pairs.append((index_1, index_2))
                                if id1 in parts_indexes_on_corr[part_id_2]:
                                    index_2 = np.where(parts_indexes_on_corr[part_id_2] == id1)[0][0] + index_offset_2
                                    if (index_1, index_2) not in near_pairs:
                                        near_pairs.append((index_1, index_2))

                print('near_pairs: ', len(near_pairs))
                kwargs_['near_pairs'] = near_pairs

            parts_inter_module = MeshInterpolateModule(
                parts_xvert,
                parts_xface,
                self.feature_bank,
                rasterizer=self.rasterizer,
                post_process=None,
                interpolate_index=None,
                features=parts_feature,
            ).to(self.device)

            part_preds = batch_solve_part_whole(
                self.cfg,
                feature_map,
                self.clutter_bank,
                parts_inter_module,
                parts_feature,
                initial_pose,
                parts_offset,
                **kwargs_
            )

            parts_xvert = [part_vert.numpy() for part_vert in parts_xvert]
            parts_xface = [part_face.numpy() for part_face in parts_xface]

            segment = self.projector.get_segment_depth(parts_xvert, parts_xface, pose_pred[0],
                                                       part_preds[0]["final"])
        else:
            segment = np.ones((512, 512)) * len(self.anno_parts)
        annotations = sample['seg']
        vis_imgs = sample['img_ori'].numpy()
        iou_dict = dict()
        print('segment: ', segment.min(), segment.max())

        img_mask = np.sum(vis_imgs[0], axis=2) > 0

        anno = annotations[0].type(torch.int32)
        anno_compare = anno[img_mask]
        seg = torch.zeros_like(anno) + len(self.anno_parts)

        total_intersection = 0
        total_union = 0
        intersections = []
        unions = []
        for anno_id, name in enumerate(self.anno_parts):
            for part_id, part_name in enumerate(parts_name):
                if name in part_name:
                    seg[segment == part_id] = anno_id

            seg_compare = seg[img_mask]

            intersection = ((seg_compare == anno_id) & (anno_compare == anno_id)).sum()
            union = ((seg_compare == anno_id) | (anno_compare == anno_id)).sum()
            iou = intersection / union
            iou_dict[name] = iou
            total_intersection += intersection
            total_union += union
            intersections.append(intersection)
            unions.append(union)

        seg_compare = seg[img_mask]
        bg_intersection = ((seg_compare == len(self.anno_parts)) & (anno_compare == len(self.anno_parts))).sum()
        bg_union = ((seg_compare == len(self.anno_parts)) | (anno_compare == len(self.anno_parts))).sum()
        total_intersection += bg_intersection
        total_union += bg_union
        bg_iou = bg_intersection / bg_union
        intersections.append(bg_intersection)
        unions.append(bg_union)
        iou_dict['bg'] = bg_iou

        miou = total_intersection / total_union
        iou_dict['mIoU'] = miou

        iou_dict['intersections'] = intersections
        iou_dict['unions'] = unions

        save_path = './visual/segment/' + self.folder
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        vis_get = seg.clone() / seg.max() * 255
        vis_get = vis_get.detach().cpu().numpy().astype(np.uint8)
        vis_get = Image.fromarray(vis_get)
        vis_get.save(f'{save_path}/{miou}_seg.jpg')

        vis_img = vis_imgs[0].astype(np.uint8)
        vis_img = Image.fromarray(vis_img)
        vis_img.save(f'{save_path}/{miou}_ori.png')

        vis_anno = anno.clone() / anno.max() * 255
        vis_anno = vis_anno.numpy().astype(np.uint8)
        vis_anno = Image.fromarray(vis_anno)
        vis_anno.save(f'{save_path}/{miou}_gt.png')

        return iou_dict

    def evaluate_locate(self, sample, debug=False):
        import BboxTools as bbt
        from PIL import Image, ImageDraw
        self.net.eval()

        ori_img = sample["img"].numpy()
        sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)
        # print('feature_map: ', feature_map.shape)
        feature_shape = feature_map.shape[2:]
        image_shape = ori_img.shape[2:]

        imd_list = []
        img_list = []
        for batch_id in range(img.shape[0]):
            img_ = Image.fromarray((ori_img[batch_id].transpose(1, 2, 0) * 255).astype(np.uint8))
            img_list.append(img_)
            imd = ImageDraw.ImageDraw(img_)
            imd_list.append(imd)

        for part_id, interpolate_feature in enumerate(self.parts_feature):
            # name = self.part_loader.get_name_listed()[part_id]
            # if name != 'wheel4':
            #     continue
            cmap = plt.get_cmap('jet')
            colors = cmap(part_id / len(self.parts_feature))
            R = int(colors[0] * 255)
            G = int(colors[1] * 255)
            B = int(colors[2] * 255)
            for point_feature in interpolate_feature:
                point_feature = point_feature.view(1, -1, 1, 1).to(feature_map.device)
                similarity = point_feature * feature_map
                similarity = similarity.sum(dim=1)
                similarity = similarity / torch.norm(point_feature) / torch.norm(feature_map, dim=1)
                similarity = similarity.view(similarity.shape[0], -1)
                max_points = torch.argmax(similarity, dim=1).detach().cpu().numpy()
                max_sims = torch.max(similarity, dim=1)[0].detach().cpu().numpy()
                xs, ys = (max_points / feature_shape[1]) / feature_shape[1] * image_shape[1], \
                    (max_points % feature_shape[1]) / feature_shape[0] * image_shape[0]

                point_size = 3
                threshold = 0.80
                for batch_id in range(img.shape[0]):
                    x, y = xs[batch_id], ys[batch_id]
                    sim = max_sims[batch_id]
                    # print('similarity: ', sim)
                    if sim < threshold:
                        continue
                    # print('x, y: ', x, y)
                    this_bbox = bbt.box_by_shape((point_size, point_size), (int(x), int(y)), image_boundary=image_shape)
                    imd_list[batch_id].ellipse(this_bbox.pillow_bbox(), fill=((R, G, B)))

        saved_path = self.folder
        if not os.path.exists(f'./visual/Locate/{saved_path}'):
            os.makedirs(f'./visual/Locate/{saved_path}')
        for img_ in img_list:
            img_.save(f'./visual/Locate/{saved_path}/{np.random.random()}.png')

        return None

    def get_ckpt(self, **kwargs):
        ckpt = {}
        ckpt['state'] = self.net.state_dict()
        ckpt['memory'] = self.memory_bank.memory
        ckpt['lr'] = self.optim.param_groups[0]['lr']
        for k in kwargs:
            ckpt[k] = kwargs[k]
        return ckpt

    def predict_inmodal(self, sample, visualize=False):
        self.net.eval()

        # sample = self.transforms(sample)
        img = sample["img"].to(self.device)
        assert len(img) == 1, "The batch size during validation should be 1"

        with torch.no_grad():
            feature_map = self.net.module.forward_test(img)

        clutter_score = None
        if not isinstance(self.clutter_bank, list):
            clutter_bank = [self.clutter_bank]
        for cb in clutter_bank:
            _score = (
                torch.nn.functional.conv2d(feature_map, cb.unsqueeze(2).unsqueeze(3))
                .squeeze(0)
                .squeeze(0)
            )
            if clutter_score is None:
                clutter_score = _score
            else:
                clutter_score = torch.max(clutter_score, _score)

        nkpt, c = self.kp_features.shape
        feature_map_nkpt = feature_map.expand(nkpt, -1, -1, -1)
        kp_features = self.kp_features.view(nkpt, c, 1, 1)
        kp_score = torch.sum(feature_map_nkpt * kp_features, dim=1)
        kp_score, _ = torch.max(kp_score, dim=0)

        clutter_score = clutter_score.detach().cpu().numpy().astype(np.float32)
        kp_score = kp_score.detach().cpu().numpy().astype(np.float32)
        pred_mask = (kp_score > clutter_score).astype(np.uint8)
        pred_mask_up = cv2.resize(
            pred_mask, dsize=(pred_mask.shape[1] * self.down_sample_rate, pred_mask.shape[0] * self.down_sample_rate),
            interpolation=cv2.INTER_NEAREST)

        pred = {
            'clutter_score': clutter_score,
            'kp_score': kp_score,
            'pred_mask_orig': pred_mask,
            'pred_mask': pred_mask_up,
        }

        if 'inmodal_mask' in sample:
            gt_mask = sample['inmodal_mask'][0].detach().cpu().numpy()
            pred['gt_mask'] = gt_mask
            pred['iou'] = iou(gt_mask, pred_mask_up)

            obj_mask = sample['amodal_mask'][0].detach().cpu().numpy()
            pred['obj_mask'] = obj_mask

            # pred_mask_up[obj_mask == 0] = 0
            thr = 0.8
            new_mask = (kp_score > thr).astype(np.uint8)
            new_mask = cv2.resize(new_mask, dsize=(obj_mask.shape[1], obj_mask.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
            new_mask[obj_mask == 0] = 0
            pred['iou'] = iou(gt_mask, new_mask)
            pred['pred_mask'] = new_mask

        return pred

    def fix_init(self, sample):
        self.net.train()
        sample = self.transforms(sample)

        img = sample['img'].cuda()
        obj_mask = sample["obj_mask"].cuda()
        index = torch.Tensor([[k for k in range(self.num_verts)]] * img.shape[0]).cuda()

        kwargs_ = dict(principal=sample['principal']) if 'principal' in sample.keys() else dict()
        if 'voge' in self.projector.raster_type:
            with torch.no_grad():
                frag_ = self.projector(azim=sample['azimuth'].float().cuda(), elev=sample['elevation'].float().cuda(),
                                       dist=sample['distance'].float().cuda(), theta=sample['theta'].float().cuda(),
                                       **kwargs_)

            features, kpvis = self.net.forward(img, keypoint_positions=frag_, obj_mask=1 - obj_mask,
                                               do_normalize=True, )
        else:
            if self.training_params.proj_mode == 'prepared':
                kp = sample['kp'].cuda()
                kpvis = sample["kpvis"].cuda().type(torch.bool)
            else:
                with torch.no_grad():
                    kp, kpvis = self.projector(azim=sample['azimuth'].float().cuda(),
                                               elev=sample['elevation'].float().cuda(),
                                               dist=sample['distance'].float().cuda(),
                                               theta=sample['theta'].float().cuda(), **kwargs_)

            features = self.net.forward(img, keypoint_positions=kp, obj_mask=1 - obj_mask, do_normalize=True, )
        return features.detach(), kpvis.detach()
