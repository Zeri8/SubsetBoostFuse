import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def modality_mask_to_code(mask):
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    weights = torch.tensor([8, 4, 2, 1], device=mask.device, dtype=torch.long)
    return torch.sum((mask > 0).long() * weights.view(1, -1), dim=1).clamp_(0, 15)


class SubsetAwareAdapter3D(nn.Module):
    def __init__(self, channels, num_subsets=16):
        super().__init__()
        self.channels = channels
        self.embedding = nn.Embedding(num_subsets, channels * 2)
        self.reset_stable_parameters()

    def reset_stable_parameters(self):
        with torch.no_grad():
            self.embedding.weight[:, :self.channels].fill_(1.0)
            self.embedding.weight[:, self.channels:].zero_()

    def forward(self, x, mask):
        code = modality_mask_to_code(mask)
        gamma_beta = self.embedding(code).to(dtype=x.dtype)
        gamma, beta = torch.chunk(gamma_beta, 2, dim=1)
        gamma = gamma.view(x.size(0), self.channels, 1, 1, 1)
        beta = beta.view(x.size(0), self.channels, 1, 1, 1)
        return x * gamma + beta


class ResidualBooster3D(nn.Module):
    def __init__(
        self,
        num_cls,
        hidden=16,
        max_scale=0.3,
        init_scale=0.1,
        subset_size_gate=False,
        min_gate=0.25,
    ):
        super().__init__()
        self.max_scale = max_scale
        self.init_scale = init_scale
        self.register_buffer(
            "subset_size_gate_flag",
            torch.tensor(1.0 if subset_size_gate else 0.0, dtype=torch.float32),
        )
        self.register_buffer("min_gate_value", torch.tensor(float(min_gate), dtype=torch.float32))
        self.conv1 = nn.Conv3d(num_cls + 1 + 4, hidden, kernel_size=3, padding=1, bias=True)
        self.norm = nn.InstanceNorm3d(hidden, affine=True)
        self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.conv2 = nn.Conv3d(hidden, num_cls, kernel_size=1, bias=True)
        init_ratio = max(min(init_scale / max_scale, 0.999), -0.999)
        self.scale_logit = nn.Parameter(torch.tensor(math.atanh(init_ratio), dtype=torch.float32))
        self.reset_stable_parameters()

    def reset_stable_parameters(self):
        nn.init.kaiming_normal_(self.conv1.weight, a=0.2)
        if self.conv1.bias is not None:
            nn.init.constant_(self.conv1.bias, 0)
        nn.init.constant_(self.conv2.weight, 0)
        if self.conv2.bias is not None:
            nn.init.constant_(self.conv2.bias, 0)
        with torch.no_grad():
            init_ratio = max(min(self.init_scale / self.max_scale, 0.999), -0.999)
            self.scale_logit.fill_(math.atanh(init_ratio))

    def forward(self, base_pred, mask):
        base_pred = torch.clamp(base_pred, min=1e-5, max=1.0)
        uncertainty = 1.0 - base_pred.max(dim=1, keepdim=True).values
        mask_map = mask.float().view(mask.size(0), 4, 1, 1, 1)
        mask_map = mask_map.expand(-1, -1, base_pred.size(2), base_pred.size(3), base_pred.size(4))
        x = torch.cat([base_pred, uncertainty, mask_map.to(dtype=base_pred.dtype)], dim=1)
        residual_logits = self.conv2(self.act(self.norm(self.conv1(x))))
        scale = self.max_scale * torch.tanh(self.scale_logit)
        scale_for_log = scale
        if self.subset_size_gate_flag.item() > 0.5:
            subset_size = mask.float().sum(dim=1, keepdim=True).clamp(min=1.0, max=4.0)
            min_gate = self.min_gate_value.to(device=mask.device, dtype=base_pred.dtype)
            gate = min_gate + (1.0 - min_gate) * (4.0 - subset_size) / 3.0
            scale = scale * gate.view(mask.size(0), 1, 1, 1, 1).to(dtype=base_pred.dtype)
            scale_for_log = scale.detach().mean()
        logits = torch.log(base_pred) + scale * residual_logits
        return F.softmax(logits, dim=1), scale_for_log
