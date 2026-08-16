"""MiniMax H3 帧网格纯数学，不依赖 ComfyUI 运行环境，可独立单元测试。

token 与像素帧的映射、音频窗换算全部由官方常量推导，不写死比例。
"""

try:
    from comfy.ldm.minimax.model import FRAME_PER_TOKEN, FRAME_RESCALE
except Exception:
    # 无 ComfyUI 环境（单元测试）时使用官方当前值
    FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    FRAME_RESCALE = 5.0 / 3.0


def align_frame_count(n):
    """像素帧数向上对齐到 H3 的 17k+5 网格（5/22/39/56/73...）。"""
    while n % 17 != 5:
        n += 1
    return n


def video_latent_t(frame_count):
    """像素帧数 -> 视频 latent token 数（官方 temporal_shape 同式）。"""
    return 2 if frame_count <= 5 else ((frame_count - 5) // 17) * 5 + 2


def latent_t_to_frames(latent_t):
    """视频 latent token 数 -> 像素帧数（官方 AddGuide 同式）。"""
    return sum(FRAME_PER_TOKEN[k % 5] for k in range(latent_t))


def audio_tokens_for_frames(frames):
    """像素帧数 -> 音频 latent token 数（1 视频帧 = FRAME_RESCALE 个音频 latent 帧）。"""
    return round(frames * FRAME_RESCALE)
