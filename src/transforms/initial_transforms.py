import numpy as np
import torch
from torch import nn

from src.transforms.lenslees_helpers.preprocessor import get_dataset_object


def load_mask_by_id(mask_label, masks_root):
    mask_label = int(mask_label)
    mask = np.load(f"{masks_root}/mask_{mask_label}.npy")
    return mask


def DoubleSizes(masks_root, fast=False, **data):
    mask = None
    if not fast:
        mask = load_mask_by_id(data["mask_label"], masks_root)
    lensed, lensless, psf = get_dataset_object(
        data["lensed"], data["lensless"], mask, fast=fast
    )
    return {"lensed": lensed, "lensless": lensless, "psf": psf}


def ChangeData(**data):
    lensed, lensless, psf = get_dataset_object(
        data["lensed"], data["lensless"], data["mask"]
    )
    return {"lensed": lensed, "lensless": lensless, "psf": psf}
