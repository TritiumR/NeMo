import torch
import numpy as np
import torch.nn as nn
import copy
from model_utils.decode_utils import KpDecoder, MlpPointsFC
from model_utils.acne_utils import AcneKpEncoder


def evaluate_pose(x, att):
    # x: B3N, att: B1KN1
    # ts: B3k1
    pai = att.sum(dim=3, keepdim=True) # B1K11
    att = att / torch.clamp(pai, min=1e-3)

    ts = torch.sum(
        att * x[:, :, None, :, None], dim=3) # B3K1
    return ts


def procruste_pose(pts0, pts1, conf=None, std_noise=1e-8):
    # pts0(Bx3xN), pts1(Bx3xN)
    indim = pts0.shape[1]
    if conf is None:
        conf = 1 / pts0.shape[-1]
    else:
        conf = (conf / conf.sum(1, keepdim=True)).unsqueeze(1)  # (B1N)
    if std_noise > 0:
        pts0 = pts0 + torch.normal(0, std_noise, size=pts0.shape).to(pts0.device)
        pts1 = pts1 + torch.normal(0, std_noise, size=pts0.shape).to(pts1.device)
    center_pts0 = (pts0 * conf).sum(dim=2, keepdim=True)
    center_pts1 = (pts1 * conf).sum(dim=2, keepdim=True)

    pts0_centered = pts0 - center_pts0
    pts1_centered = pts1 - center_pts1

    cov = torch.matmul(
        pts0_centered * conf, pts1_centered.transpose(2, 1))

    # Faster but having memory issue.
    # U, _, V = torch.svd(cov.cpu())
    # U = U.cuda()
    # V = V.cuda()
    # d = torch.eye(indim).unsqueeze(0).repeat(U.shape[0], 1, 1).to(U.device)
    # d[:, -1, -1] = torch.det(torch.matmul(V, U.transpose(2, 1))) # scalar
    # Vd = torch.matmul(V, d)
    # Rs = torch.matmul(Vd, U.transpose(2, 1))

    # Slower.
    Rs = []
    for i in range(pts0.shape[0]):
        U, S, V = torch.svd(cov[i])
        d = torch.det(torch.matmul(V, U.transpose(1, 0)))  # scalar
        Vd = torch.cat([V[:, :-1], V[:, -1:] * d], dim=-1)
        R = torch.matmul(Vd, U.transpose(1, 0))
        Rs += [R]
    Rs = torch.stack(Rs, dim=0)

    ts = center_pts1 - torch.matmul(Rs, center_pts0)  # R * pts0 + t = pts1
    return Rs, ts


class AcneAe(nn.Module):
    def __init__(self, config):
        super(AcneAe, self).__init__()
        self.config = config
        encoder = eval(self.config.feat_net)

        # size of each capsule's descriptor
        indim = config.indim
        self.encoder = encoder(config, indim)
        if config.aligner == "init":
            aligner_config = copy.copy(config)
            aligner_config.aligner = "None"
            aligner_config.ref_kp_type = "mlp_fc"
            aligner_config.loss_reg_att_f = 1  # localization/equi/equili losses
            aligner_config.loss_beta = 1  # invariance loss
            aligner_config.loss_ref_kp_can = 1  # canonical loss
            self.aligner = AcneAeAligner(aligner_config)

        if config.ref_kp_type == "mlp_fc":
            self.ref_kp_net = MlpPointsFC(
                config.acne_dim * config.acne_num_g, config.acne_num_g, indim, config)

    def decompose_one_pc(self, x, R_can=None, T_can=None):
        """ decompose one pc"""
        assert x.shape[2] == self.config.indim

        x = x.transpose(2, 1)
        input_feat = x[..., None]

        # print('input_feat: ', input_feat.shape)
        # high = input_feat.max(dim=2)[0]
        # low = input_feat.min(dim=2)[0]
        # print(f'high: {high}; low: {low}')

        with torch.no_grad():
            gc_att = self.aligner.encoder(input_feat, att_aligner=None, return_att=True)  # BCK1, B1GN1
            gc, att = gc_att

            # Canonicalize
            if R_can is not None and T_can is not None:
                x = torch.matmul(R_can, x) + T_can

            # pose_local = evaluate_pose(x, att)
            # kps = pose_local.squeeze(-1)
            # kps_ref = self.aligner.ref_kp_net(gc.reshape(gc.shape[0], -1))
            # kps_ref = kps_ref - kps_ref.mean(dim=2, keepdim=True)
            # R_can, T_can = procruste_pose(kps, kps_ref, std_noise=0)  # kps_ref = R * kps + T
            # print(f'R_can: {R_can}; T_can: {T_can}')
            # x = torch.matmul(R_can, x) + T_can

            input_feat = x[..., None]
            gc_att = self.encoder(input_feat, att_aligner=att, return_x=True)  # BCK1, B1GN1
            gc, feat = gc_att

            # torch.Size([1, 1, 10, 1024, 1]) torch.Size([1, 128, 1, 1024, 1])
            # print('shapes: ', att.shape, feat.shape)
            label_map = torch.argmax(att, dim=2)[0].squeeze()

        return label_map, feat


class AcneAeAligner(AcneAe):
    pass

