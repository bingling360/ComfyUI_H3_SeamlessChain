"""H3 Seamless Chain —— MiniMax H3 多段视频段间引导（无缝续拍）单节点。

原理：生成第 N+1 段时，把第 N 段结尾 context_frames 帧从其采样输出的 AV latent
里直接切片（不解码不重编码，零颜色漂移），作为 keyframe 钉在第 N+1 段头部
（官方 conditioning 协议 minimax_keyframes，锚 resolved_frame_index=0），
采样每步重注入，顺着上一段尾部的运动继续画；解码后裁掉头部重叠桥再拼接。

兼容性：不 monkey-patch；conditioning/latent 构造直接调用官方
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo 节点类，采样走官方
common_ksampler。keyframe 携带 audio_latent 需要 ComfyUI 含 PR #15439
（2026-08-09 之后构建），旧版本自动降级为仅视频引导并在报告中说明。
"""

import torch
import nodes
import node_helpers
import comfy.utils
import comfy.samplers
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

from .grid import video_latent_t, latent_t_to_frames, audio_tokens_for_frames

try:
    import comfy.ldm.minimax.model as _minimax_model
    # PR #15439 引入的模块级函数；存在即代表支持 keyframe 音频与 refs/keyframes 合并
    KEYFRAME_AUDIO_SUPPORTED = hasattr(_minimax_model, "_ref_t_span")
except Exception:
    KEYFRAME_AUDIO_SUPPORTED = False


def _decode_audio(audio_vae, audio_latent):
    # 与官方 VAEDecodeAudio（comfy_extras/nodes_audio.vae_decode_audio）保持一致
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sample_rate = getattr(audio_vae, "audio_sample_rate_output",
                           getattr(audio_vae, "audio_sample_rate", 44100))
    return audio, sample_rate


def _tail_keyframe(av_samples, ctx_frames, with_audio):
    video, audio = av_samples.unbind()
    vt = video_latent_t(ctx_frames)
    kf = {
        "resolved_frame_index": 0,
        "latent": video[:, :, -vt:, :, :].clone(),
    }
    if with_audio and audio is not None:
        at = audio_tokens_for_frames(ctx_frames)
        if 0 < at <= audio.shape[-1]:
            kf["audio_latent"] = audio[..., -at:].clone()
    return kf


class H3SeamlessChainSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": ("MODEL", {"tooltip": "t2v/i2v 接 fl2va UNET；r2v（参考图片）接 ref2va UNET。建议先经官方 ModelSamplingMiniMaxH3（shift 12/3）"}),
                "文本编码器": ("CLIP", {"tooltip": "CLIP Loader 的 type 必须选 minimax（Qwen3-VL）"}),
                "视频VAE": ("VAE", {"tooltip": "minimax_h3_video_vae"}),
                "音频VAE": ("VAE", {"tooltip": "minimax_h3_audio_vae"}),
                "提示词": ("STRING", {"multiline": True, "default": "", "tooltip": "每个非空行 = 一段视频的提示词。至少两段才体现段间引导"}),
                "宽度": ("INT", {"default": 864, "min": 32, "max": 16384, "step": 32}),
                "高度": ("INT", {"default": 480, "min": 32, "max": 16384, "step": 32}),
                "每段帧数": ("INT", {"default": 124, "min": 5, "max": 3600, "tooltip": "每段可见帧数 @24fps（124≈5秒，训练范围约 124-362）"}),
                "引导帧数": ([5, 22, 39, 56], {"default": 22, "tooltip": "段间引导重叠桥：钉入下段头部的上段尾帧数。越大衔接越顺、越慢越吃显存"}),
                "种子": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
                "步数": ("INT", {"default": 25, "min": 1, "max": 100}),
                "CFG": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
                "采样器": (comfy.samplers.KSampler.SAMPLERS, {"default": "res_multistep"}),
                "调度器": (comfy.samplers.KSampler.SCHEDULERS, {"default": "simple"}),
            },
            "optional": {
                "首帧图片": ("IMAGE", {"tooltip": "第一段的起始帧（i2v）。接了它请用 fl2va UNET"}),
                "参考图片": ("IMAGE", {"tooltip": "参考主体图（r2v，batch 多张，≤9）。接了它请用 ref2va UNET，与首帧图片互斥"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "AUDIO", "INT", "STRING", "IMAGE", "AUDIO")
    RETURN_NAMES = ("图像", "音频", "帧率", "报告", "分段图像", "分段音频")
    OUTPUT_TOOLTIPS = ("拼接后的全部视频帧", "拼接后的音轨", "帧率（24）", "分段执行报告",
                       "每段裁剪后的可见帧（逐段展开，接 Create Video 可单独保存）",
                       "每段对应的音轨（与分段图像逐段配对）")
    OUTPUT_IS_LIST = (False, False, False, False, True, True)
    FUNCTION = "run"
    CATEGORY = "MiniMaxH3"
    DESCRIPTION = "MiniMax H3 多段视频段间引导：上段尾帧 latent 直切钉入下段头部，无缝续拍后裁剪拼接，输出完整视频+音频。"

    def run(self, **kw):
        model = kw["模型"]
        clip = kw["文本编码器"]
        video_vae = kw["视频VAE"]
        audio_vae = kw["音频VAE"]
        prompts = kw["提示词"]
        width = kw["宽度"]
        height = kw["高度"]
        length = kw["每段帧数"]
        context_frames = kw["引导帧数"]
        seed = kw["种子"]
        steps = kw["步数"]
        cfg = kw["CFG"]
        sampler_name = kw["采样器"]
        scheduler = kw["调度器"]
        first_frame = kw.get("首帧图片")
        ref_images = kw.get("参考图片")
        seg_prompts = [ln.strip() for ln in prompts.splitlines() if ln.strip()]
        if not seg_prompts:
            raise ValueError("提示词不能为空：请每个非空行写一段提示词")
        if first_frame is not None and ref_images is not None:
            raise ValueError("首帧图片（i2v，fl2va UNET）与参考图片（r2v，ref2va UNET）不能同时连接")

        use_refs = ref_images is not None
        if use_refs:
            chain = "r2v（ref2va UNET）"
        elif first_frame is not None:
            chain = "i2v 首段 + t2v 续段（fl2va UNET）"
        else:
            chain = "t2v（fl2va UNET）"

        ctx = int(context_frames)
        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        refs = None
        if use_refs:
            refs = {f"ref_image_{i}": ref_images[i:i + 1] for i in range(ref_images.shape[0])}

        report = [f"H3 Seamless Chain：{len(seg_prompts)} 段，链路 {chain}，上下文 {ctx} 帧"]
        if not KEYFRAME_AUDIO_SUPPORTED:
            report.append("注意：当前 ComfyUI 不含 PR #15439（Add Guide 协议），段间引导降级为仅视频锚定，音频不锚定"
                          + ("；r2v 链上引导会与参考素材冲突失效，强烈建议升级 ComfyUI" if use_refs else ""))

        pbar = comfy.utils.ProgressBar(len(seg_prompts))
        all_frames = []
        all_wav = None
        seg_frames = []
        seg_wavs = []
        sample_rate = None
        guide = None

        for i, prompt in enumerate(seg_prompts):
            seg_len = length + (ctx if i > 0 else 0)
            if use_refs:
                out = MiniMaxH3ReferenceToVideo.execute(
                    clip=clip, vae=video_vae, audio_vae=audio_vae,
                    prompt=prompt, width=width, height=height, length=seg_len,
                    ref_image_size="match", ref_images=refs)
            else:
                out = MiniMaxH3ImageToVideo.execute(
                    clip=clip, vae=video_vae, prompt=prompt,
                    width=width, height=height, length=seg_len,
                    first_frame=(first_frame if i == 0 else None))
            cond, latent = out[0], out[1]

            sampled_fc = latent_t_to_frames(latent["samples"].tensors[0].shape[2])
            if guide is not None:
                cond = node_helpers.conditioning_set_values(cond, {
                    "minimax_keyframes": [guide],
                    "minimax_frame_count": sampled_fc,
                })

            sampled = nodes.common_ksampler(
                model, (seed + i) % 0xffffffffffffffff, steps, cfg,
                sampler_name, scheduler, cond, negative, latent, denoise=1.0)[0]
            av = sampled["samples"]

            video_t, audio_t = av.unbind()
            frames = video_vae.decode(video_t)
            if len(frames.shape) == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            wav, sample_rate = _decode_audio(audio_vae, audio_t)

            skip_f = ctx if i > 0 else 0
            frames = frames[skip_f:skip_f + length]
            total = wav.shape[-1]
            skip_s = round(total * skip_f / sampled_fc)
            take_s = round(total * length / sampled_fc)
            wav = wav[..., skip_s:skip_s + take_s]

            all_frames.append(frames.cpu())
            seg_frames.append(frames.cpu())
            seg_wav = wav.cpu()
            seg_wavs.append({"waveform": seg_wav, "sample_rate": sample_rate})
            all_wav = seg_wav if all_wav is None else torch.cat([all_wav, seg_wav], dim=-1)

            if guide is not None:
                note = f"guide=上段尾{ctx}帧" + ("+音频" if "audio_latent" in guide else "")
            else:
                note = "guide=无（首段）"
            report.append(f"段{i + 1}/{len(seg_prompts)}：采样{sampled_fc}帧 裁头{skip_f}帧 留{frames.shape[0]}帧 | {note}")

            if i + 1 < len(seg_prompts):
                guide = _tail_keyframe(av, ctx, KEYFRAME_AUDIO_SUPPORTED)

            pbar.update(1)

        images = torch.cat(all_frames, dim=0)
        return (images, {"waveform": all_wav, "sample_rate": sample_rate}, 24,
                "\n".join(report), seg_frames, seg_wavs)
