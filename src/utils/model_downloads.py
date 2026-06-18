from huggingface_hub import snapshot_download

from src.utils.io_utils import ROOT_PATH

REPO = "Noinoiio/LeADMM"
MODEL_PATH = ROOT_PATH / "data" / "models"


def download_models():
    snapshot_download(repo_id=REPO, local_dir=ROOT_PATH / "data" / "models")


def get_model_path(model_name):
    if model_name == "preADMM":
        return MODEL_PATH / "pre_admm.pth"
    if model_name == "leADMM":
        return MODEL_PATH / "leADMM"
    if model_name == "ADMMpost":
        return MODEL_PATH / "post_admm_12.pth"
    if model_name == "preADMMpost":
        return MODEL_PATH / "pre_post_12_epoch.pth"
    if model_name == "best":
        return MODEL_PATH / "pre_post_best.pth"
    raise ValueError("Unknown model name")
