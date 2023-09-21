import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import os
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt

from nemo.models.base_model import BaseModel
from nemo.models.feature_banks import mask_remove_near, remove_near_vertices_dist
from nemo.models.mesh_interpolate_module import MeshInterpolateModule
from nemo.models.solve_pose import pre_compute_kp_coords
from nemo.models.solve_pose import solve_pose
from nemo.models.batch_solve_pose import get_pre_render_samples, loss_curve_part
from nemo.models.batch_solve_pose import solve_pose as batch_solve_pose
from nemo.models.batch_solve_pose import solve_part_pose as batch_solve_part_pose
from nemo.utils import center_crop_fun
from nemo.utils import construct_class_by_name
from nemo.utils import get_param_samples
from nemo.utils import normalize_features
from nemo.utils import pose_error, iou, pre_process_mesh_pascal, load_off
from nemo.utils.pascal3d_utils import IMAGE_SIZES

from nemo.datasets.synthetic_shapenet import MeshLoader, PartLoader

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
        self.num_verts = self.dataset_config.num_verts
        self.visual_kp = cfg.training.visual_kp
        self.visual_mesh = cfg.training.visual_mesh
        self.visual_pose = cfg.inference.visual_pose
        if mode == 'test':
            self.folder = cfg.args.checkpoint
        self.ori_mesh = cfg.ori_mesh

        if self.ori_mesh:
            self.dataset_config['ori_mesh'] = True

        if cfg.task == 'correlation_marking':
            self.build()
            return

        self.part_loader = PartLoader(self.dataset_config, cate=cate)
        self.mesh_loader = MeshLoader(self.dataset_config, cate=cate, type=mode)
        self.chosen_id = '4d22bfe3097f63236436916a86a90ed7'
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

        # print('loss: ', loss.item())

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
            rasterizer = construct_class_by_name(
                **self.inference_params.rasterizer, cameras=cameras, render_settings=raster_settings
            )
        else:
            rasterizer = construct_class_by_name(
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

        self.inter_module = MeshInterpolateModule(
            xvert,
            xface,
            self.feature_bank,
            rasterizer=rasterizer,
            post_process=None,
            interpolate_index=get_mesh_index,
        ).to(self.device)

        # deal with part rendering
        part_xvert = self.part_loader.get_part_mesh()[0]
        part_xface = self.part_loader.get_part_mesh()[1]
        parts_off_set = self.part_loader.get_offset()

        verts_with_feature = chosen_verts[get_mesh_index[0]]

        # interpolate part features from memory_bank
        self.parts_feature = []
        kdtree = KDTree(verts_with_feature)
        for part_vert, off_set in zip(part_xvert, parts_off_set):
            # print('part_vert: ', part_vert.shape)
            # print('off_set: ', off_set.shape)
            part_vert = part_vert + off_set
            dist, nearest_idx = kdtree.query(part_vert, k=3)
            dist = torch.from_numpy(dist).to(self.feature_bank.device) + 1e-4
            dist = dist.type(torch.float32)
            weight = torch.softmax(1 / dist, dim=1)
            nearest_feature = [self.feature_bank[nearest] for nearest in nearest_idx]
            part_feature = (torch.stack(nearest_feature, dim=0) * weight.unsqueeze(-1)).sum(dim=1)
            # print('part_feature: ', part_feature.shape)
            part_feature = part_feature / part_feature.norm(dim=1).unsqueeze(-1)
            self.parts_feature.append(part_feature)

        part_xvert = to_tensor(part_xvert)
        part_xface = to_tensor(part_xface)

        self.part_inter_modules = []
        for part_id, part_name in enumerate(self.part_loader.get_name_listed()):
            part_inter_module = MeshInterpolateModule(
                [part_xvert[part_id]],
                [part_xface[part_id]],
                self.feature_bank,
                rasterizer=rasterizer,
                post_process=None,
                interpolate_index=None,
                features=[self.parts_feature[part_id]],
            ).to(self.device)
            self.part_inter_modules.append(part_inter_module)

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
        # print('self.feature_pre_rendered: ', self.feature_pre_rendered.shape)

        print('build inference done')

    def evaluate(self, sample, debug=False):
        self.net.eval()

        img = sample['img'].cuda()

        mesh_index = [0] * img.shape[0]

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

        img = sample['img'].cuda()
        ori_img = sample['img_ori'].numpy()

        from PIL import Image
        # print('shape: ', ori_img[0].shape)
        img_ = Image.fromarray((ori_img[0].transpose(1, 2, 0) * 255).astype(np.uint8))
        img_.save(f'./visual/image.png')

        mesh_index = [0] * img.shape[0]

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
        part_offsets = self.part_loader.get_offset()

        # # print('names: ', self.part_loader.get_name_listed())
        # loss_curve_part(
        #     self.cfg,
        #     feature_map,
        #     self.part_inter_modules,
        #     self.parts_feature,
        #     initial_pose,
        #     part_offsets,
        # )

        # print('part_feature: ', self.parts_feature[0].shape)
        part_preds = batch_solve_part_pose(
            self.cfg,
            feature_map,
            self.part_inter_modules,
            self.parts_feature,
            initial_pose,
            part_offsets,
        )

        for idx in range(len(img)):
            part_verts, part_faces = self.part_loader.get_part_mesh()
            self.projector.visual_part_pose(part_verts, part_faces, sample['img_ori'][idx], preds[idx]["final"][0],
                                            part_preds[idx]["final"][0], self.folder, preds[idx]["pose_error"])

        # exit(0)

        return preds, part_preds

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

        saved_path = self.folder.split('/')[1] + '/' + self.folder.split('/')[3]
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
