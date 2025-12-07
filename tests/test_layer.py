import torch
import torch.nn as nn
from skiptora.layer import SkipLoRALayer
from skiptora.hooks import register_skip_hooks

def test_impact_metric():
    layer = SkipLoRALayer(64, 64, r=4, threshold=0.01)
    x = torch.randn(2, 64)
    
    out = layer(x)
    assert layer.impact_metric is not None
    assert 0 <= layer.impact_metric <= 1.0

def test_skipping():
    model = nn.Linear(64, 64)
    # Simulate LoRA setup
    skiptora_layer = SkipLoRALayer(64, 64, r=4, threshold=1.0)  # High threshold to activate
    register_skip_hooks(model, [""])  # Mock
    
    x = torch.randn(2, 64, requires_grad=True)
    y = model(x).sum()
    y.backward()
    assert model.weight.grad is not None  # Should compute
    
    # Low impact simulation (manual set)
    skiptora_layer.is_active = False
    # Re-forward/backward would zero grads in hook

if __name__ == "__main__":
    test_impact_metric()
    test_skipping()
    print("All tests passed!")
