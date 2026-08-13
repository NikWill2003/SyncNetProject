from torch import Tensor

def accuracy(logits: Tensor, targets: Tensor) -> float:
    
    if targets.numel() == 0:
        return float('nan')
    return (logits.argmax(dim=-1) == targets).float().mean().item()