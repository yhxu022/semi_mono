# from mmdet3d.utils import SampleList
from typing import List, Sequence, Tuple, Union
def reweight_loss_dict(losses: dict, weight: float) -> dict:
    """Reweight losses in the dict by weight.

    Args:
        losses (dict):  A dictionary of loss components.
        weight (float): Weight for loss components.

    Returns:
            dict: A dictionary of weighted loss components.
    """
    for name, loss in losses.items():
        if 'loss' in name:
            if isinstance(loss, Sequence):
                losses[name] = [item * weight for item in loss]
            else:
                losses[name] = loss * weight
    return losses

def rename_loss_dict(prefix: str, losses: dict) -> dict:
    """Rename the key names in loss dict by adding a prefix.

    Args:
        prefix (str): The prefix for loss components.
        losses (dict):  A dictionary of loss components.

    Returns:
            dict: A dictionary of loss components with prefix.
    """
    return {prefix + k: v for k, v in losses.items()}

def _filter_gt_instances_by_score(batch_data_samples,
                                  score_thr: float):
    """Filter ground truth (GT) instances by score.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        assert 'scores' in data_samples.gt_instances, \
            'there does not exit scores in instances'
        assert 'scores_3d' in data_samples.gt_instances_3d, \
            'there does not exit scores in instances_3d'
        if data_samples.gt_instances.bboxes.shape[0] > 0:
            data_samples.gt_instances = data_samples.gt_instances[
                data_samples.gt_instances.scores > score_thr]
            data_samples.gt_instances_3d = data_samples.gt_instances_3d[
                data_samples.gt_instances_3d.scores_3d > score_thr]
    return batch_data_samples


def _filter_gt_instances_by_size(batch_data_samples,
                                 wh_thr: tuple):
    """Filter ground truth (GT) instances by size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score.
    """
    for data_samples in batch_data_samples:
        bboxes = data_samples.gt_instances.bboxes
        if bboxes.shape[0] > 0:
            w = bboxes[:, 2] - bboxes[:, 0]
            h = bboxes[:, 3] - bboxes[:, 1]
            data_samples.gt_instances = data_samples.gt_instances[
                (w > wh_thr[0]) & (h > wh_thr[1])]
            data_samples.gt_instances_3d = data_samples.gt_instances_3d[
                (w > wh_thr[0]) & (h > wh_thr[1])]
    return batch_data_samples

def filter_gt_instances(batch_data_samples,
                        score_thr: float = None,
                        wh_thr: tuple = None):
    """Filter ground truth (GT) instances by score and/or size.

    Args:
        batch_data_samples (SampleList): The Data
            Samples. It usually includes information such as
            `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
        score_thr (float): The score filter threshold.
        wh_thr (tuple):  Minimum width and height of bbox.

    Returns:
        SampleList: The Data Samples filtered by score and/or size.
    """

    if score_thr is not None:
        batch_data_samples = _filter_gt_instances_by_score(
            batch_data_samples, score_thr)
    if wh_thr is not None:
        batch_data_samples = _filter_gt_instances_by_size(
            batch_data_samples, wh_thr)
    return batch_data_samples