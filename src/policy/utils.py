import logging
import torch


logger = logging.getLogger(__name__)


def select_optimizer(optim_name: str):
    _optim_dict = {
        "sgd": torch.optim.SGD,
        "rmsprop": torch.optim.RMSprop,
        "adam": torch.optim.Adam,
    }
    return _optim_dict[optim_name]


def extract_gradients_info(named_parameters):
    layer_name = []
    mean_grad = []
    max_grad = []
    min_grad = []
    for n, p in named_parameters:
        if p.requires_grad:
            if p.grad is None:
                layer_name.append(n)
                mean_grad.append(0)
                max_grad.append(0)
                min_grad.append(0)
            else:
                layer_name.append(n)
                mean_grad.append(p.grad.abs().mean().item())
                max_grad.append(p.grad.abs().max().item())
                min_grad.append(p.grad.abs().min().item())
    return {
        "layer_name": layer_name,
        "mean_gradient": mean_grad,
        "max_gradient": max_grad,
        "min_gradient": min_grad,
    }
