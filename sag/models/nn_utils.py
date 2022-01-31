
import torch
import torch.nn.functional as F
from .es_module import ESModule


activation_function_dict={
    "tanh": torch.nn.Tanh,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "relu": torch.nn.ReLU
}


def get_activation_function(activation="gelu"):
    if activation == "selu":
        return F.selu
    elif activation == "gelu":
        return F.gelu
    elif activation == "relu":
        return F.relu
    elif activation == "tanh":
        return F.tanh
    else:
        raise NotImplementedError


def save_init_args(init_method):
    def wrapper(self, *args, **kwargs):
        self._init_args = args
        self._init_kwargs = kwargs
        init_method(self, *args, **kwargs)
    return wrapper


def load_model(directory, device="cpu", strict=True):
    return ESModule.load(directory, device=device, strict=strict)
