from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download, snapshot_download
from tqdm.auto import tqdm

from datasets import load_dataset
from src.datasets.base_dataset import BaseDataset
from src.transforms import DoubleSizes
from src.utils.io_utils import ROOT_PATH, read_json, write_json
from src.utils.metric_utils import load_image
from src.transforms.initial_transforms import ChangeData
class CustomDirDataset(BaseDataset):

    def __init__(self, dataset_length, data_path, *args, **kwargs):
        if (isinstance(data_path, str)):
            data_path = Path(data_path)
        index_path = data_path / "index.json"

        # each nested dataset class must have an index field that
        # contains list of dicts. Each dict contains information about
        # the object, including label, path, etc.
        self.data_path = data_path
        if index_path.exists():
            index = read_json(str(index_path))
        else:
            index = self._create_index(dataset_length, data_path)
        super().__init__(index, *args, **kwargs)

    def _create_index(self, dataset_length, data_path):


        index = []
        data_path.mkdir(exist_ok=True, parents=True)
        # to get pretty object names
        if dataset_length <= 0:
            dataset_length = 1e6
        data_path = data_path / "lensless"
        number_of_zeros = int(np.log10(dataset_length)) + 1
        for file_path in data_path.iterdir():
            if file_path.is_file():
                id = int(file_path.name.split(".")[0])
                index.append({"path": str(file_path), "id": id})
                if  (len(index) >= dataset_length):
                    break
        write_json(index, str(data_path / "index.json"))

        return index

    def load_object(self, path, id):
        """
        Load object from disk.

        Args:
            path (str): path to the object.
        Returns:
            data_object (Tensor):
        """
        lensless = load_image(Path(path))
        path_mask = Path(path).parent.parent / "masks" / (str(id) + ".npy")
        mask = np.load(path_mask)
        lensed_fake = torch.zeros_like(lensless)
        data_object = {"lensless": lensless, "lensed": lensed_fake, "mask": mask}
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
        data_id = data_dict["id"]
        data_object = self.load_object(data_path, data_id)
        data_object.update({"id": data_dict["id"]})

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
        instance_data.update(ChangeData(masks_root=self.masks, **instance_data))
        if self.instance_transforms is not None:
            for transform_name in self.instance_transforms.keys():
                instance_data[transform_name] = self.instance_transforms[
                    transform_name
                ](instance_data[transform_name])
        return instance_data
