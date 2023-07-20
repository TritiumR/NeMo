import os
import numpy as np
import torch


class Network(object):
    """Wrapper for training and testing procedure"""

    def __init__(self, config):
        """init"""
        self.config = config
        # make network deterministic
        torch.backends.cudnn.deterministic = True
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        np.random.seed(1234)

        # build model
        from acne_ae import AcneAe
        # from models.acne import Acne
        model = eval(self.config.model)
        model = model(config)  # 2_1 output one_dimensional score

        if self.config.use_cuda:
            model.cuda()
        self.model = model

        suffix = self.config.log_dir
        self.res_dir = os.path.join(self.config.res_dir, suffix)
        self.checkpoint_file = os.path.join(self.res_dir, "checkpoint.pth")

    def _restore(self, pt_file):
        # Read checkpoint file.
        load_res = torch.load(pt_file)

        self.model.load_state_dict(load_res["model"], strict=False)

    def load_checkpoint(self):
        if self.config.pt_file == "":
            self._restore(self.checkpoint_file)
            print(f'loaded from {self.checkpoint_file}')
        else:
            self._restore(self.config.pt_file)

#
# network.py ends here
