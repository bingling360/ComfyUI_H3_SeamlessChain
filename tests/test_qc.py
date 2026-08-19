"""桥帧打分单测：python tests/test_qc.py（需 torch；无 torch 环境自动跳过）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    if not hasattr(torch, "arange"):
        # 合集运行时 test_node_structure 的假 torch stub 可能已在 sys.modules——视为无 torch
        raise ImportError
    from qc import (frame_scores, pick_backtrack, seam_metrics,
                    smoothstep_blend_head, smoothstep_fade_head, loudness_align_head)
except ImportError:
    torch = None


def _frames(pattern: str, n=6, size=96):
    """pattern: checker=清晰棋盘格 / blur=平滑渐变 / blown=全过曝。"""
    base = torch.arange(size).float()
    if pattern == "blur":
        plane = (base / size).unsqueeze(0).unsqueeze(0).repeat(3, 1, 1)
    elif pattern == "blown":
        plane = torch.ones(3, size, size) * 0.995
    else:  # checker
        row = base.unsqueeze(0).repeat(size, 1)
        grid = (row % 8 < 4)
        grid = grid ^ grid.t()
        plane = grid.unsqueeze(0).repeat(3, 1, 1).float()
        plane = 0.2 + plane * 0.6   # 0.2/0.8 二值：保持锐边，避开全图曝光裁剪惩罚
    return plane.unsqueeze(0).repeat(n, 1, 1, 1).movedim(1, -1)


def test_sharpness_ordering():
    if torch is None:
        return print("SKIP test_sharpness_ordering (no torch)")
    sharp = float(frame_scores(_frames("checker"))[-1])
    blurry = float(frame_scores(_frames("blur"))[-1])
    blown = float(frame_scores(_frames("blown"))[-1])
    assert sharp > blurry, f"checker={sharp} blur={blurry}"
    assert blown < blurry, f"blown={blown} blur={blurry}"   # 过曝惩罚生效


def test_pick_backtrack_levels():
    if torch is None:
        return print("SKIP test_pick_backtrack_levels (no torch)")
    scores = torch.full((60,), 50.0)
    scores[-1] = 10.0                                     # 尾帧坏
    back, hit = pick_backtrack(scores, 34, 30.0)
    assert back == 17 and hit == 50.0                     # 17 帧处达标 -> 回退 17
    scores[len(scores) - 1 - 17] = 12.0                   # 17 帧处也坏
    back, hit = pick_backtrack(scores, 34, 30.0)
    assert back == 34 and hit == 50.0                     # 34 帧处达标
    back, hit = pick_backtrack(scores, 17, 30.0)          # 上限 17：无档可退
    assert back == 0 and hit == 10.0                      # 不硬剪
    scores[-1] = 45.0
    back, hit = pick_backtrack(scores, 34, 30.0)
    assert back == 0 and hit == 45.0                      # 尾帧达标 -> 不回退


def test_seam_metrics():
    if torch is None:
        return print("SKIP test_seam_metrics (no torch)")
    a = torch.zeros(8, 8, 3)
    d, db = seam_metrics(a, a)
    assert d == 0.0 and db is None                       # 同帧 + 无 wav
    d, _ = seam_metrics(a, torch.ones(8, 8, 3))
    assert d == 1.0                                      # 全反相 -> 最大帧差
    loud = torch.full((1, 2205), 0.5)                    # 0.05s @44100
    silent = torch.zeros(1, 2205)
    _, db_same = seam_metrics(a, a, silent, silent)
    assert abs(db_same) < 0.01                           # 同波形 -> ≈0 dB
    _, db_big = seam_metrics(a, a, loud, silent)
    assert db_big > 20.0                                 # 前响后静 -> 大正跳变
    _, db_short = seam_metrics(a, a, silent[..., :10], silent[..., :10])
    assert db_short is None                              # 过短 wav -> None


def test_smoothstep_blend_head():
    if torch is None:
        return print("SKIP test_smoothstep_blend_head (no torch)")
    frames = torch.zeros(6, 8, 8, 3)
    frames[5] = 1.0                                        # span=3 窗外的帧
    anchor = torch.full((8, 8, 3), 0.5)
    out = smoothstep_blend_head(frames, anchor, 3)
    assert torch.equal(out[0], anchor)                     # 第 0 帧逐像素硬锁
    w_mid = 0.5 * 0.5 * (3 - 2 * 0.5)                      # t=0.5 -> w=0.5
    assert torch.allclose(out[1], torch.full_like(out[1], 0.5 * (1 - w_mid)))
    assert torch.equal(out[5], frames[5])                  # 窗外帧原样保留
    assert torch.equal(out[2], frames[2])                  # 窗尾 w=1 -> 纯生成帧
    out99 = smoothstep_blend_head(frames, anchor, 99)      # span 超帧数 -> 钳制
    assert out99.shape[0] == 6
    assert torch.equal(out99[0], anchor) and torch.equal(out99[-1], frames[-1])
    out1 = smoothstep_blend_head(frames, anchor, 1)        # span=1 -> 仅硬锁
    assert torch.equal(out1[0], anchor) and torch.equal(out1[1], frames[1])
    assert float(out1.min()) >= 0.0 and float(out1.max()) <= 1.0


def test_smoothstep_fade_head():
    if torch is None:
        return print("SKIP test_smoothstep_fade_head (no torch)")
    # 已弃用（拼接期音频叠加导致双声部重叠）：任何输入都应原样返回、零干预
    wav = torch.randn(2, 100)
    anchor = torch.randn(2, 41)
    out = smoothstep_fade_head(wav, anchor)
    assert out.shape == wav.shape
    assert torch.equal(out, wav)
    assert torch.equal(smoothstep_fade_head(wav, None), wav)


def test_loudness_align_head():
    if torch is None:
        return print("SKIP test_loudness_align_head (no torch)")
    rate, n = 1000, 2000
    prev = torch.full((1, int(rate * 0.25)), 0.4)          # 上段尾 RMS 0.4
    quiet = torch.full((1, n), 0.2)                        # 本段 RMS 0.2 -> +6.02dB
    out, db = loudness_align_head(quiet, prev, rate=rate, fade_s=1.0)
    assert abs(db - 6.0) < 0.01                            # 钳制在 +6dB
    assert torch.allclose(out[..., 0], quiet[..., 0] * 10 ** (6.0 / 20.0), rtol=1e-4)
    assert torch.equal(out[..., -1], quiet[..., -1])       # 渐出末端 gain=1
    loud = torch.full((1, n), 0.8)                         # 本段 RMS 0.8 -> -6.02dB
    out2, db2 = loudness_align_head(loud, prev, rate=rate, fade_s=1.0)
    assert abs(db2 - (-6.0)) < 0.01                        # 钳制在 -6dB
    same = torch.full((1, n), 0.4)
    _, db_same = loudness_align_head(same, prev, rate=rate)
    assert abs(db_same) < 0.05                             # 同响度 -> ≈0dB 微干预
    silent = torch.zeros(1, n)
    out3, db3 = loudness_align_head(silent, prev, rate=rate)
    assert db3 is None and torch.equal(out3, silent)       # 本段静音 -> 不干预
    out4, db4 = loudness_align_head(quiet, torch.zeros(1, 100), rate=rate)
    assert db4 is None and torch.equal(out4, quiet)        # 上段静音 -> 不干预


if __name__ == "__main__":
    if torch is None:
        print("no torch in this env; qc tests skipped")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
