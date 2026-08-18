"""跨缝窗口潜空间精修：接缝跳变（段首偏差）的根治层。

段间引导桥走 conditioning 路径（minimax_keyframes 每步重注入），但下段
生成区 latent 仍从纯噪声、denoise=1.0 起步——锚定只是"建议"，模型每步
都有偏离自由度，段首偏差大小取决于内容与种子的抽卡运气。开源共识的
根治思路是让衔接内容进入 latent 主路径（Wan2.1 I2V 首帧落位 / SVI 2.0
尾 latent 直通 / SkyReels-V2 重叠区低噪共同去噪）。

本模块实现其中与官方采样协议完全兼容的一种：把「上段尾 W 帧 + 本段头
W 帧」的干净 latent 沿 token 网格拼成跨缝窗口，整体作为初始 latent 以
denoise=精修强度 联合重去噪（common_ksampler 公开 API，flow-matching
noise_scaling=(1-σ)x0+σε，输出经 process_latent_in/out 对称往返）。
模型在自己的潜空间里调和两侧运动矢量——缝后帧与缝前帧出自同一次去噪，
缝差≈段内正常帧差；上段帧、存档、分段 mp4 一概不动，只替换本段头部。

纯后处理不进存档指纹（同接缝混合惯例）；精修种子=本段种子+1 派生，
同输入重放逐帧一致。
"""

import torch

from . import grid


def seam_window_tokens(side_frames):
    """窗口 token 布局：两侧各 side_frames 帧的 token 数 + 网格点补齐。

    返回 (prev_tokens, cur_tokens, window_tokens, window_frames)。视频 latent
    只能整 token 切，窗口总 token 数必须踩 17k+5 网格的 token 形态
    （≡2 mod 5）；两个网格 token 数相加 ≡4 mod 5 永不落点，缺的 token
    全部从本段侧补齐——窗口起点对齐上段末端，本段替换区随之多几帧。
    """
    vt = grid.video_latent_t(int(side_frames))
    tw = 2 * vt
    while tw % 5 != 2 or tw < 7:
        tw += 1
    return vt, tw - vt, tw, grid.latent_t_to_frames(tw)


def build_seam_window(prev_lat, cur_lat, side_frames):
    """切两侧 kept latent 拼（视频+音频）跨缝窗口，不够切返回 None。

    prev_lat/cur_lat = (video_t, audio_t, kt0, kt1, ka0, ka1)：k* 为 kept 区
    token 边界——生成段 [video_latent_t(skip_f), end_t) / 序章 [0, 全长)。
    返回 (win_video_t, win_audio_t, prev_tokens, cur_tokens, window_frames)。
    """
    vt_p, vt_c, tw, wf = seam_window_tokens(side_frames)
    pv, pa, pk0, pk1, pa0, pa1 = prev_lat
    cv, ca, ck0, ck1, ca0, ca1 = cur_lat
    if pk1 - pk0 < vt_p or ck1 - ck0 < vt_c:
        return None
    win_v = torch.cat([pv[:, :, pk1 - vt_p:pk1].to(cv.device), cv[:, :, ck0:ck0 + vt_c]], dim=2)
    at = grid.audio_tokens_for_frames(wf)
    ap = min(grid.audio_tokens_for_frames(grid.latent_t_to_frames(vt_p)), at)
    ac = at - ap
    if ap <= 0 or ac <= 0 or pa1 - pa0 < ap or ca1 - ca0 < ac:
        return None
    win_a = torch.cat([pa[..., pa1 - ap:pa1].to(ca.device), ca[..., ca0:ca0 + ac]], dim=-1)
    return win_v, win_a, vt_p, vt_c, wf


def adaptive_strength(seam_diff):
    """按缝差分档精修强度：坏缝需要更强的重去噪调和，好缝轻修保细节。

    d < 0.04 → 0.30（轻调）；0.04-0.08 → 0.45（标准）；> 0.08 → 0.55（强调和）。
    固定 0.45 整窗重去噪会把好缝也重新加噪-去噪一遍——细节软化的主因之一。
    """
    if seam_diff is None:
        return 0.45
    if seam_diff < 0.04:
        return 0.30
    if seam_diff <= 0.08:
        return 0.45
    return 0.55


def refine_seam(模型, negative, prompt, refs, clip, video_vae, audio_vae,
                win_video, win_audio, window_frames, width, height,
                strength, seed, 步数, CFG, 采样器, 调度器,
                seam_kf_latent=None, seam_kf_index=None,
                tail_kf_latent=None, tail_kf_index=None):
    """窗口联合重去噪，返回精修后的窗口 video latent（丢弃精修音频）。

    cond 复用官方节点按 window_frames 构造（r2v 链带 refs 保身份连续）；
    **双端 keyframe 注入 cond**（minimax_keyframes 协议）：缝前侧（上段尾）
    整段 video latent 锚 index=0，本段侧窗口末端 latent 锚窗口末帧——
    模型只能改写中间区，两端都续得上。单侧锚定时模型会自由改写无锚端，
    精修末端再羽化回原帧就造成二次边界（糊感来源之一）。初始 latent 仍为
    跨缝窗口切片（承载两侧结构），denoise=强度 保留大部分原结构。

    音频不精修：音轨沿用现有响度对齐路径，避免重去噪改变人声/音色。
    """
    import comfy.model_management
    import comfy.nested_tensor
    import node_helpers
    from comfy_extras.nodes_minimax_h3 import (MiniMaxH3ImageToVideo,
                                               MiniMaxH3ReferenceToVideo)
    dev = comfy.model_management.intermediate_device()
    win_v = win_video.to(device=dev, dtype=torch.float32)
    win_a = win_audio.to(device=dev, dtype=torch.float32)
    if refs:
        out = MiniMaxH3ReferenceToVideo.execute(
            clip=clip, vae=video_vae, audio_vae=audio_vae,
            prompt=prompt, width=width, height=height, length=window_frames,
            ref_image_size="match", **refs)
    else:
        out = MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt=prompt,
            width=width, height=height, length=window_frames)
    cond, latent = out[0], out[1]
    keyframes = []
    if seam_kf_latent is not None and seam_kf_index is not None:
        keyframes.append({"resolved_frame_index": seam_kf_index,
                          "latent": seam_kf_latent.to(device=dev, dtype=torch.float32)})
    if tail_kf_latent is not None and tail_kf_index is not None:
        keyframes.append({"resolved_frame_index": tail_kf_index,
                          "latent": tail_kf_latent.to(device=dev, dtype=torch.float32)})
    if keyframes:
        cond = node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": window_frames,
        })
    latent["samples"] = comfy.nested_tensor.NestedTensor((win_v, win_a))
    sampled = nodes_common_ksampler(
        模型, int(seed), 步数, CFG, 采样器, 调度器, cond, negative, latent,
        denoise=float(strength))[0]
    return sampled["samples"].unbind()[0]


def nodes_common_ksampler(*args, **kwargs):
    """common_ksampler 延迟导入（无 ComfyUI 环境的单测只测纯数学部分）。"""
    import nodes
    return nodes.common_ksampler(*args, **kwargs)
