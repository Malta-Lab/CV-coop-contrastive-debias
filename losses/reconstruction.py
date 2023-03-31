import torch
from torch import nn

class ReconstructionLoss(nn.Module):
    def __init__(self, distance: nn.Module) -> None:
        super().__init__()

        self.distance = distance
        #self.weight = weight

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        #moved this section to autoencoder.py on 'iter' function
        #if self.delta < epoch:
        #    return x * 0
        return self.distance(x, y) #* self.weight
