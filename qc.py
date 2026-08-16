"""桥帧质量打分：解码像素上的 Laplacian 清晰度 + 曝光/对比度（纯 torch 数学）。

打分对象是「将成为下段重叠桥」的尾部像素帧（解码产物现成，零额外开销）；
分数低于阈值判定为坏尾，自动回退档位固定 0 / 17 / 34 帧——
17 帧 = 5 个 latent token，天然踩 H3 的 17k+5 帧网格，切片 token 数不变。
"""

import torch
import torch.nn.functional as F

BACKTRACK_STEPS = (17, 34)


def frame_scores(frames):
    """frames: [N,H,W,3] 0-1 float（任意设备）。返回每帧总分张量 [N]。

    总分 = log1p(Laplacian方差×1000) + 对比度×2 − 曝光裁剪占比×2.5
    （与 LtoJ 总控台选帧打分同形，便于阈值互相参考；灰度、最长边 ≤512 计算）
    """
    rgb = frames[..., :3].float().movedim(-1, 1)                 # [N,3,H,W]
    gray = rgb[:, 0] * 0.299 + rgb[:, 1] * 0.587 + rgb[:, 2] * 0.114
    h, w = gray.shape[-2:]
    scale = 512 / max(h, w)
    if scale < 1.0:
        gray = F.interpolate(gray.unsqueeze(1), scale_factor=scale,
                             mode="bilinear", align_corners=False).squeeze(1)
    lap_k = gray.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0],
                             [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    lap = F.conv2d(gray.unsqueeze(1), lap_k, padding=1).squeeze(1)
    sharp = lap.flatten(1).var(dim=1, unbiased=False)
    contrast = gray.flatten(1).std(dim=1, unbiased=False)
    clipped = ((gray < 0.02) | (gray > 0.98)).float().flatten(1).mean(dim=1)
    return torch.log1p(sharp * 1000.0) + contrast * 2.0 - clipped * 2.5


def pick_backtrack(score, limit: int, threshold: float):
    """尾部窗口总分 -> (回退帧数, 回退点得分)。

    尾帧达标 -> 不回退；否则依序尝试 17 / 34 帧（受 limit 截断），
    取首个达标档位；全部不达标 -> 不硬剪，返回 (0, 尾帧分) 由调用方警告。
    """
    tail = float(score[-1])
    if tail >= threshold:
        return 0, tail
    for back in BACKTRACK_STEPS:
        if back > limit:
            break
        idx = len(score) - 1 - back
        if idx < 0:
            break
        s = float(score[idx])
        if s >= threshold:
            return back, s
    return 0, tail
