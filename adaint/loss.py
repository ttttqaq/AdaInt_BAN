import torch
import torch.nn as nn
from mmedit.models.registry import LOSSES
from mmedit.apis import init_model
from mmedit.models.builder import build_loss
from .config_utils import load_config
from .experiment_utils import REFINEMENT_TYPES, load_enhancer_checkpoint


@LOSSES.register_module()
class KD_Loss(nn.Module):

    def __init__(self, config_path, model_path, loss_type, loss_weight=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        teacher_config = load_config(config_path)
        teacher_device = torch.device('cuda', torch.cuda.current_device())
        if teacher_config.model.type in REFINEMENT_TYPES:
            self.teacher = init_model(
                teacher_config, None, device=teacher_device)
            load_enhancer_checkpoint(self.teacher, model_path)
        else:
            self.teacher = init_model(
                teacher_config, model_path, device=teacher_device)
        self.teacher.requires_grad_(False)
        self.loss = build_loss(dict(type=loss_type, loss_weight=loss_weight))
        self.elementwise_loss = build_loss(dict(
            type=loss_type, loss_weight=loss_weight, reduction='none'))

    def forward(self, pred, lq, gt=None, use_confidence=False,
                confidence_temperature=0.05, **context):
        self.teacher.eval()
        teacher_context = context if getattr(
            self.teacher, 'use_full_image_context', False) else {}
        with torch.no_grad():
            output_teacher, _, _ = self.teacher.forward_dummy(lq, **teacher_context)
        if use_confidence:
            if gt is None or confidence_temperature <= 0:
                raise ValueError('confidence KD requires GT and a positive temperature')
            error = (output_teacher - gt).abs().mean(dim=1, keepdim=True)
            weight = torch.exp(-error / confidence_temperature).expand_as(pred)
            # Absolute confidence matters: do not renormalize it back to mean 1.
            kd_loss = (self.elementwise_loss(pred, output_teacher) * weight).mean()
        else:
            kd_loss = self.loss(pred, output_teacher)
        return kd_loss
