import torch
import torch.nn.functional as F
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)

from src.metrics.base_metric import BaseMetric
from src.transforms import get_roi_tensors
import matplotlib.pyplot as plt

class ImageMetric(BaseMetric):
    def __init__(self, name, metric, *args, **kwargs):
        super().__init__(name, *args, **kwargs)
        self.args_for_metric = {}
        if isinstance(metric, str):
            if metric == "PSNR":
                self.metric = peak_signal_noise_ratio
                self.args_for_metric["data_range"] = 1.0
            elif metric == "SSIM":
                self.metric = structural_similarity_index_measure
                self.args_for_metric["data_range"] = 1.0
            elif metric == "LPIPS":
                self.metric = learned_perceptual_image_patch_similarity
                self.args_for_metric["net_type"] = "vgg"
                self.args_for_metric["normalize"] = True
            elif metric == "MSE":
                self.metric = F.mse_loss
            else:
                assert False, f"Bad metric {metric}"

    def __call__(self, reconstructed, lensed, **batch):
        plt.imshow(reconstructed[0].cpu().numpy())
        plt.imshow(lensed[0].cpu().numpy())
        reconstructed_roi = get_roi_tensors(reconstructed)
        lensed_roi = get_roi_tensors(lensed)

        return self.metric(reconstructed_roi, lensed_roi, **self.args_for_metric).item()
