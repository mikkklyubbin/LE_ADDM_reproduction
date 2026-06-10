
from logging import root
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
from src.transforms.lenslees_helpers.preprocessor import get_cropped_lensed
from src.metrics.image_metrics import ImageMetric
def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = to_tensor(img).unsqueeze(0)
    return x



def calc_metrics(pred_path, target_path, name_of_files = "Image"):
    metrics = {
        "PSNR": ImageMetric("PSNR", "PSNR"),
        "SSIM": ImageMetric("SSIM", "SSIM"),
        "LPIPS": ImageMetric("LPIPS", "LPIPS"),
        "MSE": ImageMetric("MSE", "MSE"),
    }
    data_metrics = {
        "PSNR": 0,
        "SSIM": 0,
        "LPIPS": 0,
        "MSE": 0,
    }
    root = Path(pred_path)
    cnt = 0
    for file_path in root.iterdir():
        if file_path.is_file():
            pred = torch.load(file_path)
            id = pred["id"]
            img = Image.open(target_path + "/" + name_of_files + str(id) + ".png").convert("RGB")
            img = to_tensor(img)
            img = get_cropped_lensed(img.permute(1, 2, 0).numpy(), pred["reconstructed"])
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            for metric_name, metric in metrics.items():
                data_metrics[metric_name] += metric(pred["reconstructed"], img)
            cnt += 1
    for metric_name in data_metrics:
        data_metrics[metric_name] /= cnt
    
    return data_metrics