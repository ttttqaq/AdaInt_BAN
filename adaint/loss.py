import torch
import torch.nn as nn
from mmedit.models.registry import LOSSES
from mmedit.apis import init_model
from mmedit.models.builder import build_loss




@LOSSES.register_module()
class KD_Loss(nn.Module):

    def __init__(self, config_path, model_path, loss_type, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.teacher = init_model(
            config_path, model_path, device=torch.device('cuda', 0))
        self.teacher.requires_grad_(False)
        self.loss = build_loss(dict(type=loss_type, loss_weight=loss_weight))

    def forward(self, pred, lq):
        self.teacher.eval()
        output_teacher, _, _ = self.teacher.forward_dummy(lq)
        output_teacher = output_teacher.detach()
        kd_loss = self.loss(pred, output_teacher)
        return kd_loss