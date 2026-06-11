from src.transforms import get_roi_tensors


def prepare_for_save(
    data, is_batch=True, need_roi=True, detached=False, channel_first=True
):
    if not detached:
        data = data.clone().detach()
    if not (is_batch):
        data = data.unsqueeze(0)
    if need_roi:
        data = get_roi_tensors(data)
    data = data.squeeze(0).cpu().clamp(0, 1)
    if channel_first:
        data = data.permute(1, 2, 0)
    return data
