import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download

from datasets import load_dataset
from src.utils.io_utils import ROOT_PATH

DATASET_NAME = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
OUT_DIR = ROOT_PATH / "new_dataset"
N = 8

lensless_dir = OUT_DIR / "lensless"
lensed_dir = OUT_DIR / "lensed"
masks_dir = OUT_DIR / "masks"

for d in [lensless_dir, lensed_dir, masks_dir]:
    d.mkdir(parents=True, exist_ok=True)

ds = load_dataset(DATASET_NAME, split=f"test[:{N}]")

for i, sample in enumerate(ds):
    id = str(i)

    lensless = sample["lensless"]
    lensed = sample["lensed"]
    mask_label = sample["mask_label"]

    lensless.save(lensless_dir / f"{id}.png")
    lensed.save(lensed_dir / f"{id}.png")

    mask_path = hf_hub_download(
        repo_id=DATASET_NAME,
        repo_type="dataset",
        filename=f"masks/mask_{mask_label}.npy",
    )

    shutil.copy(mask_path, masks_dir / f"{id}.npy")

print("saved to:", OUT_DIR)
print("lensless:", len(list(lensless_dir.iterdir())))
print("lensed:", len(list(lensed_dir.iterdir())))
print("masks:", len(list(masks_dir.iterdir())))
