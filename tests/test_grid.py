"""纯函数单元测试：python tests/test_grid.py（仅需 torch 之外的纯 Python 环境）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from grid import (align_frame_count, align_frame_count_down, video_latent_t,
                  latent_t_to_frames, audio_tokens_for_frames,
                  token_start_frames, token_center_frames, snap_frames_to_tokens,
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


def test_frames_to_latent_t():
    from grid import frames_to_latent_t
    # 严格递增 -> 网格点上的逆映射唯一（up/down 同值）
    for t in range(2, 95):
        f = latent_t_to_frames(t)
        assert frames_to_latent_t(f, up=True) == t
        assert frames_to_latent_t(f, up=False) == t
    # 非网格点：up=最小可覆盖（不丢帧），down=最大不超出
    assert frames_to_latent_t(296, up=True) == 88 and latent_t_to_frames(88) == 298
    assert frames_to_latent_t(296, up=False) == 87 and latent_t_to_frames(87) == 294
    assert frames_to_latent_t(240, up=True) == 72 and latent_t_to_frames(72) == 243
    assert frames_to_latent_t(0) == 0 and frames_to_latent_t(-3) == 0


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


def test_token_start_center_frames():
    """首 token 1 帧、其后 4 帧（每 5 token 循环）——逐 token 掩码必须按真实帧位算。"""
    assert token_start_frames(0) == []
    assert token_start_frames(7) == [0, 1, 5, 9, 13, 17, 18]
    assert token_center_frames(3) == [0.5, 3.0, 7.0]
    # 起止自洽：第 k 个 token 的起点 = 前 k 个 token 承载帧数之和
    for t in (1, 7, 37, 47):
        starts = token_start_frames(t)
        assert starts[0] == 0
        assert latent_t_to_frames(t) == (
            starts[-1] + FRAME_PER_TOKEN[(t - 1) % len(FRAME_PER_TOKEN)])


def test_snap_frames_to_tokens():
    """钉住帧数必须落到真实 token 网格（latent 不能切半个 token）。"""
    assert snap_frames_to_tokens(0) == 0
    assert snap_frames_to_tokens(-5) == 0
    for f in (1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35, 39):
        assert snap_frames_to_tokens(f) == f                 # 网格点原样
        assert snap_frames_to_tokens(f, up=False) == f
    assert snap_frames_to_tokens(21) == 22                   # up：不丢帧
    assert snap_frames_to_tokens(21, up=False) == 18         # down：不超出
    assert snap_frames_to_tokens(124) == 124
    assert sum(FRAME_PER_TOKEN) == 17             # 每 5 token 一组 = 17 帧


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
