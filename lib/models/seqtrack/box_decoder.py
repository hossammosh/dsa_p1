import torch

def tokens_to_boxes(tokens, B, W, H):
    """
    Convert token sequence (t_x, t_y, t_w, t_h) into pixel bounding boxes.

    Args:
        tokens (Tensor): shape (..., 4), integer tokens in [0, B-1]
        B (int): number of bins (e.g. 1000 or 4000)
        W (int): search image width
        H (int): search image height

    Returns:
        Tensor: bounding box in pixels (x, y, w, h), shape (..., 4)
    """
    # Normalize to [0,1]
    x = tokens[..., 0].float() / B
    y = tokens[..., 1].float() / B
    w = tokens[..., 2].float() / B
    h = tokens[..., 3].float() / B

    # Scale to pixels
    x = x * W
    y = y * H
    w = w * W
    h = h * H
    tob=torch.stack([x, y, w, h], dim=-1)
    return tob
