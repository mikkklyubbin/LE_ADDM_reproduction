
from logging import root
from pathlib import Path

import torch
from PIL import Image
from torchvision.transforms.functional import to_tensor
import urllib.request
from src.transforms.lenslees_helpers.preprocessor import get_cropped_lensed, get_roi_tensors
from src.metrics.image_metrics import ImageMetric
import zipfile
import gdown
import matplotlib.pyplot as plt
def load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    x = to_tensor(img).unsqueeze(0)
    return x

def download_and_unpack(url, save_path):
    save_path = Path(save_path)
    save_path.mkdir(parents=True, exist_ok=True)
    gdown.download(url, str(save_path / "arc.zip"), fuzzy=True, quiet=False)

    with zipfile.ZipFile(save_path / "arc.zip", "r") as zip_file:
        zip_file.extractall(save_path)


def show_tensor_image(x, title, ax):
    x = torch.clamp(x, 0, 1)
    x = x.permute(1, 2, 0).cpu().numpy()
    ax.imshow(x)
    ax.set_title(title)

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
    images_for_display = 4
    fig, axes = plt.subplots(images_for_display, 3, figsize=(12, 8))
    for file_path in root.iterdir():
        if file_path.is_file():
            pred = torch.load(file_path)
            id = pred["id"].item()
            img = Image.open(target_path + "/" + str(id) + ".png").convert("RGB")
            img = to_tensor(img)
            img = get_cropped_lensed(img.permute(1, 2, 0).numpy(), pred["reconstructed"].permute(1,2,0))
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            print(img.shape)
            print(pred["reconstructed"].shape)
            for metric_name, metric in metrics.items():
                data_metrics[metric_name] += metric(pred["reconstructed"].unsqueeze(0), img)
            if (cnt < images_for_display):
                show_tensor_image(pred["reconstructed"], f"Predicted {id}", axes[cnt, 0])
                show_tensor_image(img.squeeze(0), f"Target {id}", axes[cnt, 1])
            cnt += 1
    for metric_name in data_metrics:
        data_metrics[metric_name] /= cnt
        print(metric_name, data_metrics[metric_name])
    plt.show()
    return data_metrics