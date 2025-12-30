import torch
from torch.autograd import Function

class _GRL(Function):
    @staticmethod
    def forward(ctx, x, lambd: float):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambd * grad_output, None

def grad_reverse(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return _GRL.apply(x, lambd)
