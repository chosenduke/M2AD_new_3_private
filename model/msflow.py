from model import MODEL

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from ._base_model import BaseModel
from .msflow_utils.default import DefaultConfig
from .msflow_utils.models import resnet, extractors, flow_models
import numpy as np

def post_process(c, size_list, outputs_list):
    # print('Multi-scale sizes:', size_list)
    logp_maps = [list() for _ in size_list]
    prop_maps = [list() for _ in size_list]
    for l, outputs in enumerate(outputs_list):
        # output = torch.tensor(output, dtype=torch.double)
        outputs = torch.cat(outputs, 0)
        logp_maps[l] = F.interpolate(outputs.unsqueeze(1),
                                     size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)
        output_norm = outputs - outputs.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
        prob_map = torch.exp(output_norm)  # convert to probs in range [0:1]
        prop_maps[l] = F.interpolate(prob_map.unsqueeze(1),
                                     size=c.input_size, mode='bilinear', align_corners=True).squeeze(1)

    logp_map = sum(logp_maps)
    logp_map -= logp_map.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0]
    prop_map_mul = torch.exp(logp_map)
    anomaly_score_map_mul = prop_map_mul.max(-1, keepdim=True)[0].max(-2, keepdim=True)[0] - prop_map_mul
    batch = anomaly_score_map_mul.shape[0]
    top_k = int(c.input_size[0] * c.input_size[1] * c.top_k)
    anomaly_score = np.mean(
        anomaly_score_map_mul.reshape(batch, -1).topk(top_k, dim=-1)[0].detach().cpu().numpy(),
        axis=1)

    prop_map_add = sum(prop_maps)
    prop_map_add = prop_map_add.detach().cpu().numpy()
    anomaly_score_map_add = prop_map_add.max(axis=(1, 2), keepdims=True) - prop_map_add

    return anomaly_score, anomaly_score_map_add, anomaly_score_map_mul.detach().cpu().numpy()

def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :]  = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :]  = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :]  = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :]  = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P

class MSFlow(BaseModel):
    def __init__(self, image_size):
        super(MSFlow, self).__init__()

        self.opt = DefaultConfig()
        self.opt.input_size = (image_size, image_size)

        self.extractor, output_channels = extractors.build_extractor(self.opt)
        self.extractor = self.extractor.to(self.opt.device).eval()
        self.parallel_flows, self.fusion_flow = flow_models.build_msflow_model(self.opt, output_channels)
        self.parallel_flows = [parallel_flow.to(self.opt.device) for parallel_flow in self.parallel_flows]
        self.fusion_flow = self.fusion_flow.to(self.opt.device)

        self.set_frozen_layers(["extractor"])

    def forward(self, image):
        h_list = self.extractor(image)
        if self.opt.pool_type == 'avg':
            pool_layer = nn.AvgPool2d(3, 2, 1)
        elif self.opt.pool_type == 'max':
            pool_layer = nn.MaxPool2d(3, 2, 1)
        else:
            pool_layer = nn.Identity()

        z_list = []
        parallel_jac_list = []
        for idx, (h, parallel_flow, c_cond) in enumerate(zip(h_list, self.parallel_flows, self.opt.c_conds)):
            y = pool_layer(h)
            B, _, H, W = y.shape
            cond = positionalencoding2d(c_cond, H, W).to(self.opt.device).unsqueeze(0).repeat(B, 1, 1, 1)
            z, jac = parallel_flow(y, [cond, ])
            z_list.append(z)
            parallel_jac_list.append(jac)

        z_list, fuse_jac = self.fusion_flow(z_list)
        jac = fuse_jac + sum(parallel_jac_list)

        return z_list, jac

    def postprocess(self, z_list, jac):
        outputs_list = [list() for _ in self.parallel_flows]
        size_list = []
        for lvl, z in enumerate(z_list):
            size_list.append(list(z.shape[-2:]))
            logp = - 0.5 * torch.mean(z ** 2, 1)
            outputs_list[lvl].append(logp)

        anomaly_score, anomaly_score_map_add, anomaly_score_map_mul = post_process(self.opt, size_list, outputs_list)

        return anomaly_score, anomaly_score_map_add


@MODEL.register_module
def msflow(pretrained=False, **kwargs):
    model = MSFlow(**kwargs)
    return model
