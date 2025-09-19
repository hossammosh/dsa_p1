import torch
import torch.nn.functional as F

def compute_confidence(logits):
    """
    Step 2A — Inference
    Convert confidence logits to probability.
    Args:
        logits (Tensor): shape (...,), raw confidence logits
    Returns:
        Tensor: confidence score in [0,1]
    """
    return torch.sigmoid(logits)


def confidence_loss(pred_boxes, gt_boxes, pred_conf, iou_fn, tau=0.5):
    """
    Step 2B — Training
    Binary cross-entropy loss on confidence prediction.

    Args:
        pred_boxes (Tensor): predicted boxes, shape (..., 4)
        gt_boxes (Tensor): ground-truth boxes, shape (..., 4)
        pred_conf (Tensor): predicted confidence scores, shape (...)
        iou_fn (callable): function to compute IoU(pred_boxes, gt_boxes)
        tau (float): IoU threshold for positive label

    Returns:
        Tensor: scalar confidence loss
    """
    iou = iou_fn(pred_boxes, gt_boxes)
    targets = (iou >= tau).float()
    return F.binary_cross_entropy(pred_conf, targets)


def gate_update(confidence, threshold=0.5):
    """
    Step 2C — Inference gating decision.
    Args:
        confidence (float or Tensor): confidence score
        threshold (float): acceptance threshold
    Returns:
        bool: True if box accepted, False if rejected
    """
    return (confidence >= threshold).item()
