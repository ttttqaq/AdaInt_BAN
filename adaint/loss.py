import torch
import torch.nn as nn
from mmedit.models.registry import LOSSES
from mmedit.apis import init_model
from .config_utils import load_config
from .experiment_utils import load_enhancer_checkpoint


@LOSSES.register_module()
class KD_Loss(nn.Module):
    """Frozen same-architecture teacher with a weak RGB MSE objective."""
    def __init__(self, config_path, model_path, loss_type='MSELoss', loss_weight=0.1):
        super().__init__()
        if loss_type != 'MSELoss' or not 0 <= loss_weight < 1:
            raise ValueError('BAN requires MSELoss and a teacher weight in [0, 1)')
        config = load_config(config_path)
        if config.model.type != 'AiLUT':
            raise ValueError('BAN teacher must use this branch architecture')
        self.teacher = init_model(config, None, device=torch.device('cuda', torch.cuda.current_device()))
        load_enhancer_checkpoint(self.teacher, model_path)
        self.teacher.requires_grad_(False)
        self.loss_weight = float(loss_weight)

    def forward(self, pred, lq, **context):
        self.teacher.eval()
        with torch.no_grad():
            teacher_output, _, _ = self.teacher.forward_dummy(lq, **context)
        return self.loss_weight*torch.nn.functional.mse_loss(pred, teacher_output)
