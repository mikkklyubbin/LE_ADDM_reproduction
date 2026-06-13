import torch


def collate_fn(dataset_items: list[dict]):
    """
    Collate and pad fields in the dataset items.
    Converts individual items into a batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch = {}

    # example of collate_fn
    result_batch["lensed"] = torch.stack([elem["lensed"] for elem in dataset_items])
    result_batch["lensless"] = torch.stack([elem["lensless"] for elem in dataset_items])
    if "psf" in dataset_items[0]:
        result_batch["psf"] = torch.stack([elem["psf"] for elem in dataset_items])
    if "psf_fft" in dataset_items[0]:
        result_batch["psf_fft"] = torch.stack(
            [elem["psf_fft"] for elem in dataset_items]
        )
    result_batch["id"] = torch.tensor([elem["id"] for elem in dataset_items])
    # result_batch["labels"] = torch.tensor([elem["labels"] for elem in dataset_items])

    return result_batch
