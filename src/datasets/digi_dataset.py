from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.transforms import DoubleSizes
from src.utils.io_utils import ROOT_PATH, read_json, write_json


class DigiCamDataset(BaseDataset):
    """
    Example of a nested dataset class to show basic structure.

    Uses random vectors as objects and random integers between
    0 and n_classes-1 as labels.
    """

    def __init__(self, dataset_length, name="train", *args, **kwargs):
        """
        Args:
            input_length (int): length of the random vector.
            n_classes (int): number of classes.
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        """
        index_path = ROOT_PATH / "data" / "DigiCam" / name / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_length, name)
        data_path = ROOT_PATH / "data" / "DigiCam" / name
        self.masks = data_path / "masks"
        super().__init__(index, *args, **kwargs)

    def _create_index(self, dataset_length, name):
        """
        Create index for the dataset. The function processes dataset metadata
        and utilizes it to get information dict for each element of
        the dataset.

        Args:
            dataset_length (int): the total number of elements in
                this random dataset.
            name (str): partition name
        Returns:
            index (list[dict]): list, containing dict for each element of
                the dataset. The dict has required metadata information,
                such as label and object path.
        """

        index = []
        data_path = ROOT_PATH / "data" / "DigiCam" / name
        data_path.mkdir(exist_ok=True, parents=True)
        DATASET_ID = "bezzam/DigiCam-Mirflickr-MultiMask-10K"
        ds = load_dataset(DATASET_ID, split=name)
        # to get pretty object names
        if dataset_length <= 0:
            dataset_length = len(ds)
        number_of_zeros = int(np.log10(dataset_length)) + 1

        # In this example, we create a synthesized dataset. However, in real
        # tasks, you should process dataset metadata and append it
        # to index. See other branches.
        for i in tqdm(range(dataset_length)):
            # create dataset
            obj_path = data_path / f"{i: 0{number_of_zeros}d}"
            obj_path.mkdir(exist_ok=True, parents=True)
            path_less = data_path / f"{i: 0{number_of_zeros}d}" / "lensless.pt"
            path_lensed = data_path / f"{i: 0{number_of_zeros}d}" / "lensed.pt"
            torch.save(ds[i]["lensless"], path_less)
            torch.save(ds[i]["lensed"], path_lensed)

            # parse dataset metadata and append it to index
            index.append({"path": str(obj_path), "mask_label": ds[i]["mask_label"]})

        masks = snapshot_download(
            repo_id=DATASET_ID,
            repo_type="dataset",
            allow_patterns="masks/*.npy",
            local_dir=data_path,
        )
        self.masks = masks

        # write index to disk
        write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, path):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        lensless = torch.load(Path(path) / "lensless.pt")
        lensed = torch.load(Path(path) / "lensed.pt")
        data_object = {"lensless": lensless, "lensed": lensed}
        return data_object

    def __getitem__(self, ind):
        """
        Get element from the index, preprocess it, and combine it
        into a dict.

        Notice that the choice of key names is defined by the template user.
        However, they should be consistent across dataset getitem, collate_fn,
        loss_function forward method, and model forward method.

        Args:
            ind (int): index in the self.index list.
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        """
        data_dict = self._index[ind]
        data_path = data_dict["path"]
        data_object = self.load_object(data_path)
        data_label = data_dict["mask_label"]
        data_object.update({"mask_label": data_label})

        instance_data = data_object
        instance_data = self.preprocess_data(instance_data)

        return instance_data

    def preprocess_data(self, instance_data):
        """
        Preprocess data with instance transforms.

        Each tensor in a dict undergoes its own transform defined by the key.

        Args:
            instance_data (dict): dict, containing instance
                (a single dataset element).
        Returns:
            instance_data (dict): dict, containing instance
                (a single dataset element) (possibly transformed via
                instance transform).
        """
        instance_data.update(DoubleSizes(masks_root=self.masks, **instance_data))
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data
