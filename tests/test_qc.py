"""桥帧打分单测：python tests/test_qc.py（需 torch；无 torch 环境自动跳过）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch
    from qc import frame_scores, pick_backtrack
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


if __name__ == "__main__":
    if torch is None:
        print("no torch in this env; qc tests skipped")
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
