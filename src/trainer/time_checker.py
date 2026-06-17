import time

import torch
from tqdm.auto import tqdm

from src.metrics.tracker import MetricTracker
from src.trainer.inferencer import Inferencer
from src.utils.io_utils import ROOT_PATH


class TimeChecker(Inferencer):
    """
    Inferencer (Like Trainer but for Inference) class

    The class is used to process data without
    the need of optimizers, writers, etc.
    Required to evaluate the model on the dataset, save predictions, etc.
    """

    def __init__(
        self,
        **args,
    ):
        self.mean_time = []
        super().__init__(**args)

    def process_batch(self, batch_idx, batch, metrics, part):
        """
        Run batch through the model, compute metrics, and
        save predictions to disk.

        Save directory is defined by save_path in the inference
        config and current partition.

        Args:
            batch_idx (int): the index of the current batch.
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type
                of the partition (train or inference).
            part (str): name of the partition. Used to define proper saving
                directory.
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform)
                and model outputs.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster
        start = time.time()
        outputs = self.model(**batch)
        end = time.time()
        self.mean_time.append(end - start)
        batch.update(outputs)

        return batch
