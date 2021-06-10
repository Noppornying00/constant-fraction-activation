from typing import Dict, Iterable, Callable
from torch import nn, Tensor
import torch


class constant_fraction_wrapper(nn.Module):
    def __init__(self, model: nn.Module, prev_module_names: Iterable[str], relu_module_names: Iterable[str]):
        super().__init__()
        self.model = model
        self.prev_layer_outputs = {layer: torch.empty(
            0) for layer in prev_module_names}

        self.model_layer_dict = dict([*model.named_modules()])

        self.hook_handles = {layer_id: self.model_layer_dict[layer_id].register_forward_hook(
            self.generate_hook_fn(layer_id)) for layer_id in prev_module_names}
        self.prev_relu_pairs = list(zip(prev_module_names, relu_module_names))

    def __getattribute__(self, name: str):
        # the last three are used in nn.Module.__setattr__
        if name in ["model",
                    "prev_layer_outputs",
                    "hook_handles",
                    "generate_hook_fn",
                    "close",
                    "adjust_bias",
                    "model_layer_dict",
                    "prev_relu_pairs",
                    "get_activation_fractions",
                    "__dict__",
                    "_parameters",
                    "_buffers",
                    "_non_persistent_buffers_set"]:
            return object.__getattribute__(self, name)
        else:
            return getattr(self.model, name)

    def generate_hook_fn(self, layer_id: str) -> Callable:
        def fn(_, __, output):
            self.prev_layer_outputs[layer_id] = output
        return fn

    def close(self):
        [hook_handle.remove() for hook_handle in self.hook_handles.values()]

    def adjust_bias(self, fraction):
        for prev_name, relu_name in self.prev_relu_pairs:
            self.model_layer_dict[relu_name].adjust_bias(
                fraction, self.prev_layer_outputs[prev_name])

    def get_activation_fractions(self):
        fractions = []
        for prev_name, relu_name in self.prev_relu_pairs:
            fractions.append(self.model_layer_dict[relu_name].get_activation_fractions(
                self.prev_layer_outputs[prev_name]))

        return fractions
