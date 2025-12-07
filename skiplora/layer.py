```python
import torch
import torch.nn as nn
from peft.tuners.lora.layer import LoraLayer  # Base from PEFT

class SkipLoRALayer(LoraLayer):
    """
    SkipLoRA Layer: Extends LoRA with impact-based gradient skipping.
    """
    def __init__(self, in_features: int, out_features: int, r: int = 8, lora_alpha: int = 16,
                 lora_dropout: float = 0.1, fan_in_fan_out: bool = False, merge_weights: bool = True,
                 threshold: float = 0.05, **kwargs):
        super().__init__(in_features, out_features, r, lora_alpha, lora_dropout, fan_in_fan_out, merge_weights, **kwargs)
        self.threshold = threshold
        self.impact_metric = None
        self.is_active = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Standard LoRA forward
        result = super().forward(x)
        
        # Low-cost impact assessment: M_l = ||delta||_2 / ||h||_2
        if self.lora_A is not None and self.r > 0:
            h_orig = x  # Original hidden state
            delta = self.lora_B(self.lora_A(x)) * (self.scaling if hasattr(self, 'scaling') else 1.0)
            if not self.fan_in_fan_out:
                delta = delta.transpose(0, 1)
            
            norm_delta = torch.norm(delta, p=2, dim=-1).mean()
            norm_h = torch.norm(h_orig, p=2, dim=-1).mean()
            self.impact_metric = norm_delta / (norm_h + 1e-8)  # Avoid div by zero
            
            # Decision: Mark inactive if below threshold
            self.is_active = self.impact_metric >= self.threshold
        
        return result

    def zero_gradients(self):
        """Zero gradients for inactive layers during backward."""
        if not self.is_active:
            if self.lora_A is not None:
                self.lora_A.grad = None
            if self.lora_B is not None:
                self.lora_B.grad = None
