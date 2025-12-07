import torch
from torch import nn
from functools import partial

def skip_backward_hook(module, grad_input, grad_output):
    """Custom backward hook to detach and zero grads for inactive SkipLoRA layers."""
    if hasattr(module, 'is_active') and not module.is_active:
        # Detach to prevent grad flow
        return tuple(gi.detach() if gi is not None else None for gi in grad_input)
    return grad_input

def register_skip_hooks(model: nn.Module, layers_to_skip: list = None):
    """
    Register hooks for SkipLoRA layers.
    """
    if layers_to_skip is None:
        layers_to_skip = [name for name, module in model.named_modules() if isinstance(module, SkipLoRALayer)]
    
    for name in layers_to_skip:
        layer = dict(model.named_modules())[name]
        layer.register_backward_hook(skip_backward_hook)
        # Also zero param grads post-backward if inactive
        layer.register_post_hook(partial(zero_inactive_grads, layer))
    
    def zero_inactive_grads(layer, grad_input, grad_output):
        if not layer.is_active:
            for param in [layer.lora_A, layer.lora_B]:
                if param is not None and param.grad is not None:
                    param.grad.zero_()
    
    return model