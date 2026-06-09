import torch
import torch.nn.functional as F
from torchmetrics.functional.image import (
    peak_signal_noise_ratio,
    structural_similarity_index_measure,
)
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)
from pathlib import Path
from src.metrics.base_metric import BaseMetric
from src.transforms.lenslees_helpers import get_roi_tensors
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
        plt.imshow(reconstructed[0].permute((1, 2, 0)).cpu().numpy())
        Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(parents=True, exist_ok=True)
        plt.savefig("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug/debug2", bbox_inches="tight", pad_inches=0)
        plt.close()
        plt.imshow(lensed[0].permute((1, 2, 0)).cpu().numpy())
        Path("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug").mkdir(parents=True, exist_ok=True)
        plt.savefig("/home/mik/hse/Dl/project/LE_ADDM_reproduction/data/debug/debug", bbox_inches="tight", pad_inches=0)
        plt.close()
        reconstructed_roi = get_roi_tensors(reconstructed)
        lensed_roi = get_roi_tensors(lensed)
        met = self.metric(reconstructed_roi, lensed_roi, **self.args_for_metric).item()
        print(f"{self.name}: {met}")
        return met
