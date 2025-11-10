import torch
import numpy as np
from utils.weight_param import set_lr_para


def get_optimizer(model: torch.nn.Module):
    """
    Create AdamW optimizer with separate learning rates for head and backbone.
    Automatically handles DDP models.
    """
    model = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model

    params = set_lr_para()
    if params is None:
        raise ValueError("❌ set_lr_para() returned None. Please check its return statement.")

    head = list(model.head.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if id(p) not in head_ids]

    return torch.optim.AdamW([
        {"params": head, "lr": params["base_lr_head"], "initial_lr": params["base_lr_head"], "weight_decay": params["weight_decay_head"]},
        {"params": backbone, "lr": params["base_lr_backbone"], "initial_lr": params["base_lr_backbone"], "weight_decay": params["weight_decay_backbone"]},
    ])


def cosine_schedule(epoch, optimizer, warmup_epochs, max_epochs, min_lr=1e-6):
    """
    Quadratic warmup + cosine LR decay.
    """
    for pg in optimizer.param_groups:
        base_lr = pg["initial_lr"]
        if epoch < warmup_epochs:
            lr = min_lr + (base_lr - min_lr) * (epoch / warmup_epochs) ** 2
        else:
            progress = (epoch - warmup_epochs) / (max_epochs - warmup_epochs)
            lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + np.cos(np.pi * progress))
        pg["lr"] = lr
    return [pg["lr"] for pg in optimizer.param_groups]
