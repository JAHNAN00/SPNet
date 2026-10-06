"""Released SPNet-Large/CNX with random weights and explicit NYU preprocessing."""
from types import SimpleNamespace

from torch import nn
from torch.nn import functional as F

MODEL_NAME = 'SPNet'
DCN_BACKEND = 'none (native ConvNeXt/SP-Norm)'


def model_config(seed, data_dir):
    from config import Configs
    config = Configs(gpus_num=1, model_type='Large')
    return SimpleNamespace(seed=seed, from_scratch=True, variant='SPNet-Large',
                           dims=config.dims, depths=config.depths, dp_rate=config.dp_rate,
                           norm_type=config.norm_type, prop_time=0, input_hw=[228, 304],
                           internal_hw=[256, 320], padding_lrtb=[0, 16, 0, 28],
                           depth_scale_m=20.0, mask='1 for valid sparse points, 0 otherwise',
                           rgb_preprocessing='common ImageNet-normalized NYU RGB, unlike original [0,1] RGB',
                           configuration_source='config.py / test.py default Large/CNX')


def build_model(args):
    from src.networks import V2Net
    return V2Net(args.dims, args.depths, args.dp_rate, args.norm_type)


class DepthOnly(nn.Module):
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, rgb, dep):
        # Preserve the network's 5 input channels: RGB, normalized depth, validity.
        padding = (0, 16, 0, 28)
        result = self.net(F.pad(rgb, padding), F.pad(dep / 20.0, padding),
                           F.pad((dep > 0).to(dep.dtype), padding))
        return result[:, :, :dep.shape[-2], :dep.shape[-1]] * 20.0


def reference_prediction(model, rgb, dep):
    padding = (0, 16, 0, 28)
    raw = F.pad(dep / 20.0, padding)
    return model.net(F.pad(rgb, padding), raw, (raw > 0).to(raw.dtype))[
        :, :, :dep.shape[-2], :dep.shape[-1]] * 20.0
