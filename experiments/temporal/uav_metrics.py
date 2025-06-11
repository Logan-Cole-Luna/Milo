import torch

def bbox_iou(pred_cxcywh, target_cxcywh, eps=1e-6):
    """IoU between normalized [cx,cy,w,h]."""
    def to_xyxy(bbox_cxcywh_tensor):
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
