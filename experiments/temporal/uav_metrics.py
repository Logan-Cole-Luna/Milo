"""
Provides metric functions for evaluating UAV object detection models.
Currently includes Intersection over Union (IoU) for bounding boxes.
"""

import torch

def bbox_iou(pred_cxcywh, target_cxcywh, eps=1e-6):
    """
    Calculates Intersection over Union (IoU) between predicted and target bounding boxes.

    Both predicted and target bounding boxes are expected in normalized [cx, cy, w, h]
    format (center x, center y, width, height), where coordinates and dimensions
    are normalized by image width and height respectively.

    Args:
        pred_cxcywh (torch.Tensor): Predicted bounding boxes, shape (N, 4) or (4,).
        target_cxcywh (torch.Tensor): Target bounding boxes, shape (N, 4) or (4,).
        eps (float): A small epsilon value to prevent division by zero in IoU calculation.

    Returns:
        torch.Tensor: The IoU scores, shape (N,) or a scalar if inputs are single boxes.
    """
    def to_xyxy(bbox_cxcywh_tensor):
        """Converts bounding boxes from [cx, cy, w, h] to [x1, y1, x2, y2] format."""
        cx, cy, w, h = bbox_cxcywh_tensor.unbind(dim=-1)
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return torch.stack((x1, y1, x2, y2), dim=-1)

    pred_xyxy = to_xyxy(pred_cxcywh)
    target_xyxy = to_xyxy(target_cxcywh)

    ix1 = torch.max(pred_xyxy[..., 0], target_xyxy[..., 0])
    iy1 = torch.max(pred_xyxy[..., 1], target_xyxy[..., 1])
    ix2 = torch.min(pred_xyxy[..., 2], target_xyxy[..., 2])
    iy2 = torch.min(pred_xyxy[..., 3], target_xyxy[..., 3])

    inter_w = (ix2 - ix1).clamp(min=0)
    inter_h = (iy2 - iy1).clamp(min=0)
    intersection = inter_w * inter_h

    pred_area = (pred_xyxy[..., 2] - pred_xyxy[..., 0]).clamp(min=0) * \
                (pred_xyxy[..., 3] - pred_xyxy[..., 1]).clamp(min=0)
    target_area = (target_xyxy[..., 2] - target_xyxy[..., 0]).clamp(min=0) * \
                  (target_xyxy[..., 3] - target_xyxy[..., 1]).clamp(min=0)
    
    union = pred_area + target_area - intersection + eps
    
    return intersection / union
