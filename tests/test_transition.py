"""E4 双向过渡重生成：纯逻辑单测（transition_windows / dual_anchor_keyframes / e4_should_try_first）。

无 torch 依赖，stub 环境可跑。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ComfyUI_H3_SeamlessChain import transition
from ComfyUI_H3_SeamlessChain.grid import latent_t_to_frames, frames_to_latent_t


class _FakeLatent:
    def __init__(self, t):
        self.shape = (1, 16, t, 8, 8)
    def dim(self):
        return 5
    def clone(self):
        return self
    def __getitem__(self, s):
        return self


def test_transition_windows_basic():
    # 基本三窗划分：124 帧，缝在 22 帧（skip_f=22 典型 ctx=56 的 1/3），过渡窗 17 帧
    w = transition.transition_windows(124, 22, 17)
    assert w is not None
    ps, pe, ts, te, fs, fe = w
    # 过渡窗含缝点
    assert ts <= 22 < te
    # 长度对齐 17k+5 网格（5/22/39...）
    trans_len = te - ts
    assert trans_len % 17 == 5
    assert trans_len >= 5
    # past 在 transition 左边
    assert pe == ts
    assert ps <= pe
    # future 在 transition 右边
    assert fs == te
    assert fe >= fs
    # 所有窗在 [0, 124) 内
    assert 0 <= ps and fe <= 124


def test_transition_windows_edge_cases():
    # 缝在 0（段首首帧就是缝）→ past 为空
    w = transition.transition_windows(124, 0, 17)
    assert w is not None
    ps, pe, ts, te, fs, fe = w
    assert ps == pe == 0          # past 为空
    assert ts == 0
    # 缝在末尾 → future 为空
    w = transition.transition_windows(124, 120, 17)
    assert w is not None
    _, _, _, _, fs, fe = w
    assert fs == fe
    # 总帧过少 → None
    assert transition.transition_windows(3, 0, 17) is None
    # 过渡窗过大 → 收缩但仍有效
    w = transition.transition_windows(39, 19, 80)
    assert w is not None
    _, _, ts, te, _, _ = w
    assert te - ts <= 39
    # 负帧数 / 越界缝 → None
    assert transition.transition_windows(100, -1, 17) is None
    assert transition.transition_windows(100, 200, 17) is None


def test_transition_windows_grid_aligned():
    # 三窗长度都对齐 17k+5（past/transition/future 各自独立对齐）
    w = transition.transition_windows(200, 90, 39, past_frames=30, future_frames=30)
    assert w is not None
    ps, pe, ts, te, fs, fe = w
    for start, end, name in [(ps, pe, "past"), (ts, te, "trans"), (fs, fe, "future")]:
        length = end - start
        if length > 0:
            assert length % 17 == 5, f"{name} length {length} not 17k+5"


def test_dual_anchor_keyframes_basic():
    past = _FakeLatent(3)
    fut = _FakeLatent(3)
    kfs = transition.dual_anchor_keyframes(
        past, fut, 22, 39, latent_t_to_frames, frames_to_latent_t)
    assert len(kfs) == 2
    # 第一个锚（past）在过渡窗左端
    assert kfs[0]["resolved_frame_index"] == 22
    # 第二个锚（future）在过渡窗右端（最后一帧）
    assert kfs[1]["resolved_frame_index"] == 38   # 39 - 1
    # latent 字段存在（_FakeLatent 不追踪切片尺寸，只验存在性）
    assert hasattr(kfs[0]["latent"], "shape")
    assert hasattr(kfs[1]["latent"], "shape")


def test_dual_anchor_keyframes_missing_side():
    # 只有 past 没有 future → 单锚
    past = _FakeLatent(2)
    kfs = transition.dual_anchor_keyframes(
        past, None, 5, 22, latent_t_to_frames, frames_to_latent_t)
    assert len(kfs) == 1
    # 两个都没有 → 空
    assert transition.dual_anchor_keyframes(
        None, None, 0, 5, latent_t_to_frames, frames_to_latent_t) == []


def test_transition_noise_mask():
    ts, te = transition.transition_noise_mask((1, 16, 10, 8, 8), 3, 7)
    assert ts == 3 and te == 7
    # 越界自动钳制
    ts, te = transition.transition_noise_mask((1, 16, 10, 8, 8), -2, 15)
    assert ts == 0 and te == 10


def test_e4_should_try_first():
    # E4 关 → 不试
    assert transition.e4_should_try_first(False, True, True) == (False, False)
    # E4 开但缝没坏 → 不试
    assert transition.e4_should_try_first(True, False, True) == (False, False)
    # E4 开且缝坏 → 先试 E4，可回退重摇
    assert transition.e4_should_try_first(True, True, True) == (True, True)
    # E4 开缝坏但无重摇可用 → 先试 E4，失败不能回退重摇
    assert transition.e4_should_try_first(True, True, False) == (True, False)


_MAIN = {
    "test_transition_windows_basic": test_transition_windows_basic,
    "test_transition_windows_edge_cases": test_transition_windows_edge_cases,
    "test_transition_windows_grid_aligned": test_transition_windows_grid_aligned,
    "test_dual_anchor_keyframes_basic": test_dual_anchor_keyframes_basic,
    "test_dual_anchor_keyframes_missing_side": test_dual_anchor_keyframes_missing_side,
    "test_transition_noise_mask": test_transition_noise_mask,
    "test_e4_should_try_first": test_e4_should_try_first,
}


def _run_plain():
    for name, fn in _MAIN.items():
        try:
            fn()
            print(f"  PASS {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e!r}")


if __name__ == "__main__":
    print("transition 单测直跑：")
    _run_plain()
    print("OK（真回归请在 ComfyUI 环境用 pytest tests/）")
