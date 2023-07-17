import torch
import numpy as np
import torch.nn as nn
import copy
from model_utils.decode_utils import KpDecoder, MlpPointsFC
from model_utils.acne_utils import AcneKpEncoder


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

    def decompose_one_pc(self, x):
        """ decompose one pc"""
        assert x.shape[2] == self.config.indim

        x = x.transpose(2, 1)
        input_feat = x[..., None]

        # print('input_feat: ', input_feat.shape)
        # high = input_feat.max(dim=2)[0]
        # low = input_feat.min(dim=2)[0]
        # print(f'high: {high}; low: {low}')

        with torch.no_grad():
            gc_att = self.aligner.encoder(input_feat, att_aligner=None, return_att=True, return_x=True)  # BCK1, B1GN1
            gc, att, feat = gc_att
            # torch.Size([1, 1, 10, 1024, 1]) torch.Size([1, 128, 1, 1024, 1])
            # print('shapes: ', att.shape, feat.shape)
            label_map = torch.argmax(att, dim=2)[0].squeeze()

        return label_map, feat


class AcneAeAligner(AcneAe):
    pass

