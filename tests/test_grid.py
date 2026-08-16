"""纯函数单元测试：python tests/test_grid.py（仅需 torch 之外的纯 Python 环境）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import (align_frame_count, align_frame_count_down, video_latent_t,
                  latent_t_to_frames, audio_tokens_for_frames,
                  FRAME_PER_TOKEN, FRAME_RESCALE)


def test_grid_alignment():
    assert align_frame_count(5) == 5
    assert align_frame_count(6) == 22
    assert align_frame_count(124) == 124          # 17*7+5
    assert align_frame_count(124 + 22) == 158     # 17*9+5
    assert align_frame_count(146) == 158
    for k in range(0, 21):
        assert align_frame_count(17 * k + 5) == 17 * k + 5


def test_align_frame_count_down():
    assert align_frame_count_down(5) == 5
    assert align_frame_count_down(6) == 5          # 向下回到网格点
    assert align_frame_count_down(124) == 124
    assert align_frame_count_down(130) == 124
    assert align_frame_count_down(158) == 158
    assert align_frame_count_down(4) == 0          # 不足最小 5 帧 -> 调用方报错
    assert align_frame_count_down(0) == 0
    for k in range(0, 21):
        assert align_frame_count_down(17 * k + 5) == 17 * k + 5


def test_token_frame_roundtrip():
    for fc in [5, 22, 39, 56, 124, 158, 362]:
        assert latent_t_to_frames(video_latent_t(fc)) == fc
    # token 数公式：17k+5 帧 -> 5k+2 tokens
    assert video_latent_t(124) == 37
    assert video_latent_t(22) == 7


def test_valid_context_slices():
    # 尾部 N 帧 guide 的 token 切片必须整 token 覆盖且恰好 N 帧
    for ctx in [5, 22, 39, 56]:
        vt = video_latent_t(ctx)
        assert latent_t_to_frames(vt) == ctx, f"ctx={ctx}"


def test_audio_window():
    # 1 视频帧 = 40/24 = 5/3 音频 latent 帧
    assert FRAME_RESCALE == 5.0 / 3.0
    assert audio_tokens_for_frames(24) == 40      # 1 秒
    assert audio_tokens_for_frames(22) == 37      # round(22*5/3)
    assert audio_tokens_for_frames(5) == 8


def test_frame_per_token_shape():
    assert tuple(FRAME_PER_TOKEN) == (1, 4, 4, 4, 4)
    assert sum(FRAME_PER_TOKEN) == 17             # 每 5 token 一组 = 17 帧


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
