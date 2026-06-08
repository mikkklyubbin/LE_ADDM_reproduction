import torch
from torch import nn
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

class LossAcum(nn.Module):

    def __init__(self, weights, list_of_losses):
        super().__init__()
        self.weights = weights
        self.list_of_losses = nn.ModuleList(list_of_losses)

    def forward(self, **batch):
        for el in self.list_of_losses:
            batch.update(el(**batch))
        loss = 0
        res = {}
        for el in self.weights:
            loss = loss + self.weights[el] * batch[el]
            res[el] = batch[el]
        return {"loss": loss, **res}
    
class MSE_loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.MSELoss()

    def forward(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **batch):
        return {"lossL2": self.loss(reconstructed, lensed)}

class LPIPS_loss(nn.Module):
    def __init__(self, net_type="vgg", normalize=True, device="cuda"):
        super().__init__()
        self.loss = LearnedPerceptualImagePatchSimilarity(net_type=net_type, normalize=normalize)
        if (device == "auto"):
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss.to(device)

    def forward(self, reconstructed: torch.Tensor, lensed: torch.Tensor, **batch):
        return {"lossLPIPS": self.loss(reconstructed, lensed)}
