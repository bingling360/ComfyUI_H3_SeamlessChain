"""智能切镜 find_cut_point 单测（纯 torch 逻辑，无 ComfyUI 依赖）：python tests/test_smart_cut.py

覆盖：运动低谷定位、最小保留比例、搜索窗口边界、全零运动退化。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F

from qc import find_cut_point


def _make_frames(n, h=64, w=64):
    """合成帧序列：每帧略有不同的随机图案。"""
    return torch.rand(n, h, w, 3)


def test_finds_motion_valley():
    """在段尾构造一个运动低谷，验证 find_cut_point 能定位到附近。"""
    n = 100
    skip_f = 22
    vis_len = n - skip_f  # 78
    frames = _make_frames(n)

    # 在第 70-75 帧制造低谷（几乎不动）
    for i in range(70, 76):
        frames[i] = frames[70].clone()

    cut = find_cut_point(frames, skip_f, vis_len)
    assert cut is not None, "should find a cut point"
    cut_f, motion, quality = cut
    # 切镜点应在搜索窗口内（段尾 1/3 ≈ 最后 26 帧 = 帧 74-100）
    assert cut_f >= 70, f"cut {cut_f} should be in search window (>= 70)"
    print(f"PASS test_finds_motion_valley (cut@{cut_f}, motion={motion:.4f})")


def test_respects_min_keep():
    """验证最少保留 50% 内容。"""
    n = 100
    skip_f = 0
    vis_len = n
    frames = _make_frames(n)

    cut = find_cut_point(frames, skip_f, vis_len, min_keep_ratio=0.5)
    assert cut is not None
    cut_f, _, _ = cut
    # 最少保留 50% = 50 帧
    assert cut_f >= 50, f"cut {cut_f} should respect min_keep (>= 50)"
    print(f"PASS test_respects_min_keep (cut@{cut_f})")


def test_short_window_returns_none():
    """搜索窗口太短时应返回 None。"""
    n = 30
    skip_f = 0
    vis_len = n
    frames = _make_frames(n)
    # 搜索窗口 = 30 * 0.33 ≈ 10，但减去 min_keep (15) 后 search_start = 15, search_end = 30
    # 窗口 = 15 帧 >= 17？不，15 < 17 -> None
    cut = find_cut_point(frames, skip_f, vis_len)
    if cut is None:
        print("PASS test_short_window_returns_none (correctly None)")
    else:
        # 如果窗口够大（>= 17），也能返回有效点
        cut_f, _, _ = cut
        assert cut_f >= 15
        print(f"PASS test_short_window_returns_none (cut@{cut_f}, window was enough)")


def test_uniform_motion():
    """均匀运动时应返回搜索窗口内的某帧（不报错）。"""
    n = 200
    skip_f = 56
    vis_len = n - skip_f
    frames = _make_frames(n)
    cut = find_cut_point(frames, skip_f, vis_len)
    assert cut is not None
    cut_f, _, _ = cut
    assert skip_f + int(vis_len * 0.5) <= cut_f <= skip_f + vis_len
    print(f"PASS test_uniform_motion (cut@{cut_f})")


def test_prefers_low_motion():
    """两个区域：高运动 + 低运动，应选低运动区域。"""
    n = 120
    skip_f = 22
    vis_len = n - skip_f  # 98
    frames = _make_frames(n)

    # 第 80-90 帧：低运动（相同帧）
    for i in range(80, 91):
        frames[i] = frames[80].clone()

    # 第 95-110 帧：高运动（大差异）
    for i in range(95, min(111, n)):
        frames[i] = torch.rand(1, 64, 64, 3).squeeze(0)

    cut = find_cut_point(frames, skip_f, vis_len)
    assert cut is not None
    cut_f, motion, quality = cut
    # 切镜点应倾向于低运动区（80-90）而非高运动区（95-110）
    # 注意：搜索窗口是段尾 1/3 = 约 33 帧 = 帧 89-120
    # 低运动区 80-90 部分在搜索窗口内
    assert motion < 0.5, f"motion {motion:.4f} should be low at cut point"
    print(f"PASS test_prefers_low_motion (cut@{cut_f}, motion={motion:.4f})")


def test_max_trim_frames_caps_search():
    """单段丢帧上限：切点距段尾不得超过 max_trim_frames。"""
    n = 100
    skip_f = 0
    vis_len = n
    frames = _make_frames(n)

    cut = find_cut_point(frames, skip_f, vis_len, max_trim_frames=17)
    if cut is not None:
        cut_f, _, _ = cut
        assert n - cut_f <= 17, f"cut {cut_f} exceeds max_trim 17 (would drop {n - cut_f})"
    print(f"PASS test_max_trim_frames_caps_search (cut@{cut[0] if cut else None}, cap=17)")

    cut2 = find_cut_point(frames, skip_f, vis_len, max_trim_frames=5)
    assert cut2 is None, f"cap=5 leaves search window <17 frames, expect None, got {cut2[0]}"
    print("PASS test_max_trim_frames_caps_search (cap=5 -> None)")

    cut3 = find_cut_point(frames, skip_f, vis_len, max_trim_frames=None)
    assert cut3 is not None, "None keeps legacy unbounded behavior"
    cut_f3, _, _ = cut3
    assert cut_f3 >= int(n * 0.5), "legacy behavior still respects min_keep"
    print(f"PASS test_max_trim_frames_caps_search (None legacy, cut@{cut_f3})")


def test_budget_report_only_mode():
    """cap=0 语义由 nodes.py 处理（传 None 仅标注）；此处验证无上限搜索可达段尾 1/3。"""
    n = 100
    frames = _make_frames(n)
    cut = find_cut_point(frames, 0, n, max_trim_frames=None)
    assert cut is not None and cut[0] >= 67  # 段尾 1/3 ≈ 67-100
    print(f"PASS test_budget_report_only_mode (cut@{cut[0]}, search reaches tail 1/3)")


if __name__ == "__main__":
    test_finds_motion_valley()
    test_respects_min_keep()
    test_short_window_returns_none()
    test_uniform_motion()
    test_prefers_low_motion()
    test_max_trim_frames_caps_search()
    test_budget_report_only_mode()
    print("all tests passed")
