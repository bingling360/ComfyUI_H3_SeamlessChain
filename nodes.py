"""H3 Seamless Chain —— MiniMax H3 多段视频段间引导（无缝续拍）单节点。

原理：生成第 N+1 段时，把第 N 段结尾 context_frames 帧从其采样输出的 AV latent
里直接切片（不解码不重编码，零颜色漂移），作为 keyframe 钉在第 N+1 段头部
（官方 conditioning 协议 minimax_keyframes，锚 resolved_frame_index=0），
采样每步重注入，顺着上一段尾部的运动继续画；解码后裁掉头部重叠桥再拼接。

支持逐段审片（每次运行只生成一段新内容即返回，重新运行继续）与任意段重跑
（改某段提示词自动从该段重做；「重跑起始段」+ 换种子可从指定段重摇），
进度存于 latent 存档（v2 schema：逐段种子 / 提示词哈希 / 裁剪量）。

输入形态与官方 MiniMaxH3ReferenceToVideo 对齐（autogrow）：
- 提示词_1..N：每段一个输入框（或接 PrimitiveStringMultiline）
- 参考图片_0..9（<Picture i>）/ 参考视频_0..3（<Video k>）
- 参考视频音轨_0..3（与同号视频配对）/ 参考音频_0..3（<Audio j>）
- 起始视频 + 起始视频音轨：可选序章——上传视频编码为第 0 段（进存档可回放），
  成片以它开头，后续生成段从其结尾续拍（24fps 约定，经一次 VAE 重编码）

配套节点：成片历史画廊走 H3ChainSaver（web/h3chain_saver.js）；
「自动保存=分段+成片」时每段 mp4 与完整成片自动落盘 output/h3_auto/<存档名>/。

兼容性：不 monkey-patch；conditioning/latent 构造直接调用官方
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo 节点类，采样走官方
common_ksampler。多 token 桥与 keyframe 音频需要 ComfyUI 含 PR #15439
（2026-08-09 之后构建）；旧版 PackedLayout 每个 keyframe 只分配单帧 latent
行数，钉多 token 桥会形状错位，运行时自动探测并降级为单帧桥（报告说明）。
"""

import os
import time

import torch
import nodes
import node_helpers
import comfy.utils
import comfy.samplers
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
from comfy_api.latest import io

from . import checkpoint
from . import refine
from .grid import (video_latent_t, latent_t_to_frames, frames_to_latent_t,
                   audio_tokens_for_frames, align_frame_count_down)

try:
    import comfy.ldm.minimax.model as _minimax_model
    # PR #15439 引入的模块级函数；存在即代表支持 keyframe 音频与 refs/keyframes 合并
    KEYFRAME_AUDIO_SUPPORTED = hasattr(_minimax_model, "_ref_t_span")
except Exception:
    _minimax_model = None
    KEYFRAME_AUDIO_SUPPORTED = False

_full_bridge_cache = None


def cond_audio_rows_guard(dit):
    """安装 _cond_audio_rows 兜底 patch，返回恢复函数（try/finally 调用）。

    ComfyUI 0.33+ 的 PackedLayout 按 keyframe 的 audio_latent 声明 cond_audio
    段；共存 H3 插件（如 H3-Motion-Context 的 keyframe/ref 共存 patch）会
    重建 payload 并丢弃 keyframe 音频，导致 audio_embed 行数不足形状错位。
    行数与 keyframes+refs 声明不符时用 payload 里完好的素材在线重建
    （layout 段顺序：kf 音频在前、refs 音频在后）。续拍/分镜两节点共用。
    """
    orig = dit._cond_audio_rows

    def fixed(payload, device, _orig=orig):
        rows = _orig(payload, device)
        want = [z for z in (kf.get("audio_latent")
                            for kf in (payload.get("keyframes") or [])) if z is not None]
        want += [z for z in (r.get("audio_latent")
                             for r in (payload.get("refs") or [])) if z is not None]
        expected = sum(int(z.shape[-1]) * 2 for z in want)
        if want and (0 if rows is None else int(rows.shape[0])) != expected:
            rows = _orig({"cond_audio_latents": want,
                          "audio_cond_noise_aug": payload.get("audio_cond_noise_aug"),
                          "seed": payload.get("seed")}, device)
        return rows

    dit._cond_audio_rows = fixed
    return lambda: setattr(dit, "_cond_audio_rows", orig)


def step_cond_noise_guard(dit, aug_start, aug_end, fade_ratio):
    """递减锚定：visual_cond_noise_aug 随采样进度从强到弱递减到消失。

    patch dit.forward，每步读 sigma_v 计算进度，修改 minimax_payload 里的
    visual/audio cond_noise_aug。同时影响 _cond_video_rows（噪声混合比）和
    _forward（seg_t["cond"] 时间步钳位）——两者都从 payload 读同一个值。

    sigma_v: 1.0（采样开始）→ 0.0（采样结束）
    progress = (1 − sigma_v) / fade_ratio，clamp [0, 1]
    dynamic_aug = aug_start + (aug_end − aug_start) × progress

    aug_start: 初始值（接近 1.0 = 硬锚定，接缝吻合）
    aug_end: 终点值（0.0 = 纯噪声，锚定完全消失）
    fade_ratio: 递减占总步数的比例（0.3 = 前 30% 步数内递减完毕）
    """
    orig_forward = dit.forward

    def patched_forward(x, timestep, context, transformer_options={}, **kwargs):
        sigma_v = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
        progress = min((1.0 - sigma_v) / fade_ratio, 1.0) if fade_ratio > 0 else 1.0
        dynamic_aug = aug_start + (aug_end - aug_start) * progress

        payload = kwargs.get("minimax_payload")
        if payload is not None:
            payload = dict(payload)
            payload["visual_cond_noise_aug"] = dynamic_aug
            payload["audio_cond_noise_aug"] = max(0.0, dynamic_aug - 0.5)
            kwargs["minimax_payload"] = payload

        return orig_forward(x, timestep, context, transformer_options, **kwargs)

    dit.forward = patched_forward
    return lambda: setattr(dit, "forward", orig_forward)


def full_bridge_supported():
    """多 token keyframe 探测（进程内缓存一次）。

    旧版 PackedLayout（PR #15439 之前）对每个 keyframe 固定只分配 1 帧 latent
    的行数且仅认首/尾锚点，钉 22 帧桥（7 token，2835 行）会形状错位
    （[2835,96] 无法广播到 [405,96]）。构造一个 2 token 的微型 keyframe 布局
    实测 cond 行数：翻倍即支持完整桥，否则调用方降级为单帧桥。
    """
    global _full_bridge_cache
    if _full_bridge_cache is not None:
        return _full_bridge_cache
    ok = False
    if _minimax_model is not None:
        try:
            import torch
            layout = _minimax_model.PackedLayout(
                1, 2, 8, 8, 4,
                keyframes=[{"resolved_frame_index": 0,
                            "latent": torch.zeros(1, 24, 2, 8, 8)}])
            ok = int((~layout.img_update).sum()) == 32  # 2 token × 16 行（8×8 latent）
        except Exception:
            ok = False
    _full_bridge_cache = ok
    return ok


def _apply_anchor_noise(cond, aug):
    """锚定加噪：写入 H3 cond 噪声增强 kwargs（模型侧 aug<1 时按比例混噪）。

    SkyReels-V2 addnoise_condition 思路：干净锚定帧让模型逐帧复现（段首刹车/
    内容重演），加噪后锚定退化为软参考。音频加噪减半（音频桥窗口短，过噪伤听感）。
    续拍（桥锚定）与分镜（首尾帧锚定）两节点共用。
    """
    if aug <= 0.0:
        return cond
    return node_helpers.conditioning_set_values(cond, {
        "minimax_visual_cond_noise_aug": round(1.0 - aug, 4),
        "minimax_audio_cond_noise_aug": round(1.0 - aug * 0.5, 4),
    })


def _decode_audio(audio_vae, audio_latent, norm_skip_frac=0.0):
    # 与官方 VAEDecodeAudio（comfy_extras/nodes_audio.vae_decode_audio）保持一致；
    # norm_skip_frac>0 时归一化 std 只统计保留区——锚定区音频随后整段裁掉，
    # 若计入会抬高归一化分母、系统性压低本段响度，接缝处即响度跳变
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)
    if 0.0 < norm_skip_frac < 1.0:
        body = audio[..., round(audio.shape[-1] * norm_skip_frac):]
    else:
        body = audio
    std = torch.std(body, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sample_rate = getattr(audio_vae, "audio_sample_rate_output",
                           getattr(audio_vae, "audio_sample_rate", 44100))
    return audio, sample_rate


def _tail_keyframe(video_t, audio_t, ctx_frames, with_audio, end_tokens=None, full_bridge=True):
    """上段尾部 ctx 帧 latent 直切为 keyframe；end_tokens 为输出末端 token 边界。

    end_tokens=None（序章等外部源）取原始尾部；生成段必须传 kept 末端对齐值，
    保证锚定末端 == 输出末端——否则下段续拍点落在本段从未输出的网格填充帧上，
    每个接缝跳过最多 16 帧内容（观感即"接缝跳变"）。
    旧版 ComfyUI keyframe 协议只收单帧 latent：full_bridge=False 时只钉
    尾部最后 1 个 token（承载上段末尾画面），且不附音频。
    """
    vt = video_latent_t(ctx_frames) if full_bridge else 1
    end = video_t.shape[2] if end_tokens is None else min(end_tokens, video_t.shape[2])
    kf = {
        "resolved_frame_index": 0,
        "latent": video_t[:, :, end - vt:end, :, :].clone(),
    }
    if with_audio and full_bridge and audio_t is not None:
        at = audio_tokens_for_frames(ctx_frames)
        aend = min(audio_tokens_for_frames(latent_t_to_frames(end)), audio_t.shape[-1])
        if 0 < at <= aend:
            kf["audio_latent"] = audio_t[..., aend - at:aend].clone()
    return kf


def _center_cover(frames, width, height):
    """[F,H,W,3] -> [F,height,width,3]，与官方 _resize(..., "center") 同语义的 cover-crop。"""
    x = frames[..., :3].movedim(-1, 1).float()
    x = comfy.utils.common_upscale(x, width, height, "lanczos", "center")
    return x.movedim(1, -1)


def _autosave_dir(root):
    from folder_paths import get_output_directory
    return os.path.join(get_output_directory(), "h3_auto", os.path.basename(root))


def _autosave_copy_seg(root, idx):
    """把存档目录里刚落盘的分段 mp4 复制到自动保存目录（不重编码，秒级）。"""
    import shutil
    try:
        src = os.path.join(root, f"seg_{idx:03d}.mp4")
        if os.path.isfile(src):
            dst_dir = _autosave_dir(root)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(src, os.path.join(dst_dir, f"seg_{idx:03d}.mp4"))
    except Exception:
        pass


def _autosave_final(root, frames, wav, sample_rate, fps=24):
    """完整链（或审片已确认部分）PyAV 编码成片到自动保存目录，成功返回路径。"""
    from .media import save_av_mp4
    try:
        dst_dir = _autosave_dir(root)
        os.makedirs(dst_dir, exist_ok=True)
        path = os.path.join(dst_dir, f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        print(f"[H3自动保存] 编码完整成片：{int(frames.shape[0])} 帧 → {path}（编码期间 CPU 升高属正常）")
        t0 = time.time()
        ok = save_av_mp4(path, frames, wav, sample_rate, fps)
        print(f"[H3自动保存] 成片编码{'完成' if ok else '失败'}：{time.time() - t0:.0f}s")
        return path if ok else None
    except Exception as e:
        print(f"[H3自动保存] 成片编码异常：{type(e).__name__}: {e}")
        return None


def _encode_audio_latent(audio_vae, audio, tokens):
    """AUDIO dict -> [1,32,2,T] latent，裁到与画面等长的 token 数。

    仿官方 _encode_ref_audio 语义：重采样到音频 VAE 采样率后整段编码。
    torchaudio 延迟导入（ComfyUI 自带，无 ComfyUI 的结构单测不触达）。
    """
    import torchaudio

    waveform = audio["waveform"]
    sr = int(audio.get("sample_rate") or 0)
    vae_sr = getattr(audio_vae, "audio_sample_rate", 32000)
    if sr and sr != vae_sr:
        waveform = torchaudio.functional.resample(waveform, sr, vae_sr)
    z = audio_vae.encode(waveform[:1].movedim(1, -1))
    return z[..., :tokens].clone()


def _autogrow_items(group, prefix):
    """autogrow dict -> 按序号排序的非 None 值，压实为从 0 连续编号的官方 ref_* 键名。

    压实保证 <Picture 1>/<Video 1>/<Audio 1> 标签始终对应"按连接顺序的第 1 个"，
    与官方 note 的 "in the exact order they were connected" 语义一致，
    即使用户中间留了空输入框也不会错位。
    """
    if not group:
        return {}
    def num(k):
        try:
            return int(str(k).rsplit("_", 1)[-1])
        except ValueError:
            return 0
    vals = [v for _, v in sorted(group.items(), key=lambda kv: num(kv[0])) if v is not None]
    return {f"{prefix}{i}": v for i, v in enumerate(vals)}


class H3SeamlessChainSampler(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3SeamlessChainSampler",
            display_name="H3 Seamless Chain (段间引导续拍)",
            category="MiniMaxH3",
            description="MiniMax H3 多段视频段间引导：每段提示词单独输入，上段尾帧 latent 直切钉入下段头部，"
                        "无缝续拍后裁剪拼接，输出完整视频+音频（可另出每段分镜）。"
                        "参考素材用法与官方 Reference to Video 一致：<Picture i> / <Video k> / <Audio j>。",
            inputs=[
                io.Model.Input("模型"),
                io.Clip.Input("文本编码器"),
                io.Vae.Input("视频VAE"),
                io.Vae.Input("音频VAE"),
                io.Int.Input("宽度", default=864, min=32, max=16384, step=32),
                io.Int.Input("高度", default=480, min=32, max=16384, step=32),
                io.Int.Input("每段帧数", default=124, min=5, max=3600,
                             tooltip="每段可见帧数 @24fps（124≈5秒，训练范围约 124-362）。总时长 = 段数 × 每段帧数 ÷ 24"),
                io.Combo.Input("引导帧数", options=["5", "22", "39", "56"], default="22",
                               tooltip="段间引导重叠桥：钉入下段头部的上段尾帧数。越大衔接越顺、越慢越吃显存"),
                io.Int.Input("种子", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="第 i 段实际使用 种子+i"),
                io.Int.Input("步数", default=25, min=1, max=100),
                io.Float.Input("CFG", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("采样器", options=comfy.samplers.KSampler.SAMPLERS, default="res_multistep"),
                io.Combo.Input("调度器", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Combo.Input("自动存档", options=["关闭", "自动存档"], default="关闭",
                               tooltip="自动存档：每段采样后立即落盘 latent 存档（约5MB/段），中断后重跑自动跳过已完成段，结果与一次跑完一致"),
                io.String.Input("存档目录", default="",
                                tooltip="空=按参数指纹自动命名于 output/checkpoints/ 下；换链请填新名字或清空旧目录。"
                                        "填了名字即固定存档：中断重跑、改词重跑都续在这个目录。存档为中间产物，跑完可删"),
                io.Combo.Input("桥帧门控", options=["关闭", "标注", "自动回退"], default="标注",
                               tooltip="对将成为重叠桥的尾帧打分（Laplacian清晰度+曝光）：标注=只写报告；自动回退=尾帧低于阈值时向前回退17/34帧取好帧续拍（该段可见帧数随之减少）"),
                io.Float.Input("清晰度阈值", default=30.0, min=0.0, max=100.0, step=0.5,
                               tooltip="桥帧总分阈值，低于判定为坏尾。建议先跑「标注」档看报告里的分数分布再定"),
                io.Int.Input("回退上限", default=34, min=0, max=68, step=17,
                             tooltip="自动回退最多向前多少帧（17 的倍数，踩 17k+5 网格）"),
                io.Combo.Input("接缝处理", options=["潜空间精修", "smoothstep像素混合", "关闭"],
                               default="潜空间精修",
                               tooltip="拼接点衔接层（默认=旧「接缝混合」控件位）：潜空间精修=把上段尾+本段头"
                                       "各「精修窗口」帧的干净 latent 拼成跨缝窗口，整体加噪到「精修强度」后"
                                       "联合重去噪——缝两侧出自同一次去噪，缝差≈段内正常帧差，根治跳变/叠影；"
                                       "smoothstep像素混合=旧机制：上段尾帧硬锁为本段首帧+权重窗吸收前「混合帧数」"
                                       "帧偏差，偏差大时有叠影感；关闭=不处理。音频不做拼接期混合"
                                       "（两侧是不同内容，叠加会双声部重叠；音频连贯靠生成期桥锚定+响度对齐）。"
                                       "纯后处理不进存档指纹，改参数不触发重跑"),
                io.Int.Input("混合帧数", default=6, min=1, max=24,
                             tooltip="smoothstep 模式=像素混合窗帧数（两端权重导数为 0、中段过渡；"
                                     "运动越快窗应越短，6帧≈0.25s）；潜空间精修模式=精修区末端渐变回原帧的"
                                     "羽化帧数，防精修边界出现第二条微缝"),
                io.Float.Input("锚定加噪", default=0.0, min=0.0, max=0.5, step=0.05,
                               tooltip="对桥锚定帧注入噪声的比例（SkyReels-V2 addnoise_condition 思路）："
                                       "干净锚定帧会让模型起步「刹车」并在可见部分重演锚定内容；加噪让模型"
                                       "把锚定当「参考」而非「必须逐帧复现」。0=关闭（默认，保持现状）；"
                                       "0.1 微调；0.2 标准（SkyReels 同值）；0.3+ 干预强但画面细节会变软。"
                                       "仅影响带引导桥的段；不进存档指纹，改参数不触发重跑"),
                io.Combo.Input("审片模式", options=["关闭", "逐段确认"], default="关闭",
                               tooltip="逐段确认：每次运行只生成一个新的段落即返回，预览「分段图像」或自动保存目录里的"
                                       "分段视频后重新运行继续下一段；不满意可改该段提示词（自动从该段重跑）或设「重跑起始段」重摇。"
                                       "开启后存档自动启用"),
                io.Combo.Input("自动保存", options=["关闭", "分段+成片"], default="分段+成片",
                               tooltip="开启后无需任何下游接线：每段生成完自动存 mp4、链（或审片已确认部分）自动拼成"
                                       "完整成片，全部落在 output/h3_auto/<存档名>/ 下，报告注明路径。开启后存档自动启用"),
                io.Int.Input("重跑起始段", default=0, min=0, max=63,
                             tooltip="0=自动（沿用存档进度，改过提示词的段自动重做）；N=从第 N 段起丢弃存档重新生成"
                                     "（有序章时序章为第 1 段），配合改「种子」即可重摇该段及之后。用完记得改回 0"),
                io.Float.Input("精修强度", default=0.45, min=0.2, max=0.7, step=0.05,
                               tooltip="接缝精修的加噪/去噪强度（denoise）：窗口两侧各保留 (1-强度) 的原结构。"
                                       "0.2-0.3 改动小、调和弱；0.45 标准；0.6+ 过渡更顺但纹理细节改动大"),
                io.Combo.Input("精修窗口", options=["22", "39", "56"], default="39",
                               tooltip="接缝精修每侧帧数（缝前取上段尾、缝后取本段头，均为 token 网格点）。"
                                       "窗口越大过渡越从容、精修耗时越长（约为本段采样的 1/4-1/2）"),
                io.Combo.Input("接缝重摇", options=["关闭", "自动"], default="自动",
                               tooltip="自动：本段生成后若接缝帧差 > 重摇阈值，换种子重采本段（最多「重摇上限」次），"
                                       "排除抽卡坏段（同参数下缝差 0.02-0.17 波动大，重摇取达标结果）；"
                                       "回放段（存档载入）不参与重摇。坏段触发时每次重摇=一次完整段采样时长"),
                io.Float.Input("重摇阈值", default=0.06, min=0.02, max=0.3, step=0.01,
                               tooltip="接缝帧差超过此值触发自动重摇（实测好缝约 0.02-0.03，坏缝 0.08+）。"
                                       "调低更严格但更耗时"),
                io.Int.Input("重摇上限", default=1, min=0, max=3,
                             tooltip="自动重摇的额外尝试次数（0=等于关闭重摇）"),
                io.Combo.Input("智能切镜", options=["关闭", "自动"], default="关闭",
                               tooltip="自动在段尾找运动低谷（自然停顿点）作为切镜位置，多余帧丢弃。"
                                       "切镜发生在自然停顿处 → 观感无跳变。"
                                       "搜索范围=段尾1/3，最少保留50%内容。"
                                       "与桥帧门控互补：门控查尾帧是否糊，切镜找内容是否到了自然间歇"),
                io.Combo.Input("递减锚定", options=["关闭", "0.3", "0.5", "0.7"], default="关闭",
                               tooltip="锚定约束随采样进度递减：开始强（接缝吻合）→ 逐渐减弱（模型自然过渡）"
                                       "→ 完全消失（自由生成）。数值=递减占总步数比例（0.3=前30%步数递减完毕）。"
                                       "开启后「锚定加噪」的值作为递减起点，终点为 0（锚定消失）。"
                                       "治本：模型自己找到逻辑断点完成切镜，而非被强行拉住整段"),
                io.Image.Input("首帧图片", optional=True,
                               tooltip="第一段的起始帧（i2v）。用了它请用 fl2va UNET，且不能同时用任何参考素材"),
                io.Image.Input("尾帧锚定", optional=True,
                               tooltip="身份锚定帧（last_frame keyframe）：注入每段末尾位置作为人物/场景参考，"
                                       "与段首引导桥形成「隧道」——模型去噪全程被首尾双锚点约束，过了桥窗口"
                                       "（约2秒）也不会漂移。建议用角色正面清晰帧。不传则不锚定尾帧（保持现状）"),
                io.Image.Input("起始视频", optional=True,
                               tooltip="序章：上传视频（≥5 帧、24fps，超长只取前「每段帧数」内）编码为第 1 段存入存档，"
                                       "成片以它开头，生成段从其结尾续拍；经一次 VAE 重编码，不能与首帧图片同用"),
                io.Audio.Input("起始视频音轨", optional=True,
                               tooltip="序章原声（与起始视频配对，建议同源 LoadVideo 拆出；不接则序章按静音处理）"),
                io.Autogrow.Input("提示词组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.String.Input("提示词", multiline=True,
                                                            placeholder="这一段的画面描述，第 N 段会顺着第 N-1 段结尾继续"),
                                      prefix="提示词_", min=1, max=64)),
                io.Autogrow.Input("参考图片组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("参考图片",
                                                           tooltip="参考主体图，提示词用 <Picture i> 引用（i 从 1 开始按序号）。用了请接 ref2va UNET"),
                                      prefix="参考图片_", min=0, max=9)),
                io.Autogrow.Input("参考视频组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("参考视频",
                                                           tooltip="参考视频帧（24fps，2-15 秒），提示词用 <Video k> 引用"),
                                      prefix="参考视频_", min=0, max=3)),
                io.Autogrow.Input("参考视频音轨组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("参考视频音轨",
                                                           tooltip="同号参考视频的原声（参考视频音轨_1 配 参考视频_1），自动获得 <Audio j> 标签"),
                                      prefix="参考视频音轨_", min=0, max=3)),
                io.Autogrow.Input("参考音频组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("参考音频",
                                                           tooltip="独立参考音频（配乐/音效），提示词用 <Audio j> 引用"),
                                      prefix="参考音频_", min=0, max=3)),
            ],
            outputs=[
                io.Image.Output("图像"),
                io.Audio.Output("音频"),
                io.Int.Output("帧率"),
                io.String.Output("报告"),
                io.Image.Output("分段图像", is_output_list=True,
                                tooltip="每段裁剪后的可见帧（逐段展开，接 Create Video 可单独保存）"),
                io.Audio.Output("分段音频", is_output_list=True,
                                tooltip="与分段图像逐段配对的音轨"),
            ],
        )

    @classmethod
    def execute(cls, 模型, 文本编码器, 视频VAE, 音频VAE, 宽度, 高度, 每段帧数, 引导帧数,
                种子, 步数, CFG, 采样器, 调度器, 首帧图片=None, 尾帧锚定=None, 起始视频=None, 起始视频音轨=None,
                提示词组=None,
                参考图片组=None, 参考视频组=None, 参考视频音轨组=None, 参考音频组=None,
                自动存档="关闭", 存档目录="", 桥帧门控="标注", 清晰度阈值=30.0, 回退上限=34,
                接缝处理="潜空间精修", 混合帧数=6, 锚定加噪=0.0,
                审片模式="关闭", 自动保存="分段+成片", 重跑起始段=0,
                精修强度=0.45, 精修窗口="39", 接缝重摇="自动", 重摇阈值=0.06, 重摇上限=1,
                智能切镜="关闭", 递减锚定="关闭"):
        prompts = _autogrow_items(提示词组, "p")
        seg_prompts = [str(v).strip() for v in prompts.values() if str(v).strip()]
        if not seg_prompts:
            raise ValueError("提示词不能为空：请在「提示词组」里每段添加一个输入框并填写内容")

        refs = {
            "ref_images": _autogrow_items(参考图片组, "ref_image_"),
            "ref_videos": _autogrow_items(参考视频组, "ref_video_"),
            "ref_video_audios": _autogrow_items(参考视频音轨组, "ref_video_audio_"),
            "ref_audios": _autogrow_items(参考音频组, "ref_audio_"),
        }
        has_refs = any(refs.values())
        if 首帧图片 is not None and has_refs:
            raise ValueError("首帧图片（i2v，fl2va UNET）与参考素材（r2v，ref2va UNET）不能同时使用")
        if 起始视频 is not None and 首帧图片 is not None:
            raise ValueError("起始视频（序章）与首帧图片（i2v）不能同时使用：两者都定义第 1 段的视觉起点")

        length, width, height, seed = int(每段帧数), int(宽度), int(高度), int(种子)
        ctx = int(引导帧数)
        gate_limit = max(0, int(回退上限) // 17 * 17)
        from . import qc  # 桥帧打分 + 接缝测量共用（延迟导入：无 ComfyUI 环境下结构单测不触达）
        clip, video_vae, audio_vae = 文本编码器, 视频VAE, 音频VAE
        if has_refs:
            chain = "r2v（ref2va UNET）"
        elif 首帧图片 is not None:
            chain = "i2v 首段 + t2v 续段（fl2va UNET）"
        else:
            chain = "t2v（fl2va UNET）"

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))

        report = [f"H3 Seamless Chain：{len(seg_prompts)} 段，链路 {chain}，上下文 {ctx} 帧"]
        if has_refs:
            counts = " ".join(f"{k}={len(v)}" for k, v in refs.items() if v)
            report.append(f"参考素材：{counts}")
        if not KEYFRAME_AUDIO_SUPPORTED:
            report.append("注意：当前 ComfyUI 不含 PR #15439（Add Guide 协议），段间引导降级为仅视频锚定，音频不锚定"
                          + ("；r2v 链上引导会与参考素材冲突失效，强烈建议升级 ComfyUI" if has_refs else ""))
        full_bridge = full_bridge_supported()
        if not full_bridge:
            report.append("注意：当前 ComfyUI 的 keyframe 协议仅支持单帧锚定，段间引导已自动降级为单帧桥，"
                          "接缝质量受限；升级 ComfyUI 后无需改参数即自动恢复完整引导帧数")
        aug = min(max(float(锚定加噪), 0.0), 0.5)
        if aug > 0.0:
            report.append(f"锚定加噪 {aug:.2f}：桥锚定帧按参考而非逐帧复现注入（视觉 {1.0 - aug:.2f} / 音频 {1.0 - aug * 0.5:.2f} 保真）")
            if aug > 0.25:
                report.append(f"注意：锚定加噪 {aug:.2f} 偏高（>0.25 锚定偏软，段首偏差方差增大，建议 0.15-0.20）")

        tail_anchor_latent = None
        if 尾帧锚定 is not None:
            tail_anchor_latent = video_vae.encode(_center_cover(尾帧锚定[:1], width, height))
            report.append(f"尾帧锚定：身份锚点注入末帧 keyframe（{'视觉保真 ' + format(1.0 - aug, '.2f') if aug > 0 else '硬锚定'}）")
        if 接缝处理 not in ("潜空间精修", "smoothstep像素混合", "关闭"):
            # 旧工作流按控件位存的是「接缝混合」时代的值，按位回填
            接缝处理 = "关闭" if 接缝处理 == "关闭" else (
                "smoothstep像素混合" if 接缝处理 == "smoothstep" else "潜空间精修")

        fade_ratio = 0.0 if 递减锚定 == "关闭" else float(递减锚定)
        if fade_ratio > 0:
            aug_start = 1.0 - aug if aug > 0 else 0.999
            report.append(f"递减锚定：前 {fade_ratio*100:.0f}% 步数内锚定 {aug_start:.2f} → 0 递减消失"
                          + (f"（起点=锚定加噪 {aug:.2f}）" if aug > 0 else "（硬锚定起点）"))

        # 存档指纹只覆盖共享参数（不含提示词、不含种子）：改某段提示词仍指向同一条链，
        # 重跑起点由逐段提示词哈希比对定位；种子控件开着 control_after_generate 每次运行
        # 自动 +1，真正的种子序列由 manifest 权威记录（见下方续跑载入逻辑）
        resume = 自动存档 in ("自动存档", "自动续跑")  # 自动续跑=旧版工作流里的值，读档兼容
        review = 审片模式 == "逐段确认"
        autosave = 自动保存 == "分段+成片"
        reroll = max(0, int(重跑起始段))
        use_ckpt = resume or review or autosave   # 审片须落盘续接；自动保存须段落盘 mp4
        if review and not (resume or autosave):
            report.append("审片模式：存档自动启用（每段落盘 latent，跨次运行续接）")
        if autosave and not (resume or review):
            report.append("自动保存：存档自动启用（分段视频每段落盘）")
        elif reroll > 0 and not use_ckpt:
            report.append("注意：「重跑起始段」仅在自动存档/审片模式下生效，本次已忽略")
        ckpt_params = {
            "width": width, "height": height,
            "length": length, "ctx": ctx, "steps": int(步数), "cfg": float(CFG),
            "sampler": 采样器, "scheduler": 调度器, "chain": chain,
            "smart_cut": 智能切镜 == "自动", "fade_ratio": fade_ratio,
            "gate": {"mode": 桥帧门控, "threshold": float(清晰度阈值), "limit": gate_limit},
        }
        # 纯后处理参数只记录不进指纹（改值不触发重跑；报告回看用）
        seam_refine = {"mode": 接缝处理, "strength": float(精修强度),
                       "window": str(精修窗口), "blend": int(混合帧数),
                       "reroll": 接缝重摇, "reroll_th": float(重摇阈值),
                       "reroll_max": int(重摇上限), "anchor_aug": aug}
        seg_hashes = [checkpoint.prompt_hash(p) for p in seg_prompts]
        prologue_hash = None
        if 起始视频 is not None:
            f0 = 起始视频[0].detach().float().cpu()
            prologue_hash = checkpoint.prompt_hash(
                f"prologue:{int(起始视频.shape[0])}:{float(f0.mean()):.4f}:{float(f0.std()):.4f}")
        root, manifest, done, seeds = None, None, 0, []
        off = 1 if 起始视频 is not None else 0
        if use_ckpt:
            root = checkpoint.ckpt_dir(ckpt_params, 存档目录.strip())
            manifest = checkpoint.load_manifest(root)
            if manifest is not None:
                if manifest.get("schema") != checkpoint.SCHEMA:
                    raise ValueError(f"存档目录格式不认识（{manifest.get('schema')}），请换一个目录名；"
                                     "旧版 v1 存档不兼容本版本，请清空旧目录或换新名字")
                checkpoint.assert_match(manifest["params"], ckpt_params)
                if 起始视频 is None and manifest.get("has_prologue"):
                    off = 1  # 输入已断开仍沿用存档序章（LoadVideo 可 bypass），哈希校验跳过
                elif 起始视频 is not None and not manifest.get("has_prologue"):
                    manifest = checkpoint.truncate(root, manifest, 0)
                    report.append("存档续跑：检测到新接入的序章视频，整链重做")
                elif 起始视频 is not None and prologue_hash is not None:
                    stored = list(manifest.get("prompt_hashes", []))
                    if stored and stored[0] != prologue_hash:
                        manifest = checkpoint.truncate(root, manifest, 0)
                        report.append("存档续跑：序章视频已更换，整链重做")
                done = checkpoint.contiguous_done(root, int(manifest.get("done", 0)))
                full_hashes = ([prologue_hash] if off else []) + seg_hashes
                hashes = list(manifest.get("prompt_hashes", []))
                # 「重跑起始段」为 1-based 段号（与 tooltip 一致）：N=从第 N 段起重做
                start = min(max(reroll - 1, 0), done) if reroll > 0 else min(
                    checkpoint.reroll_start(hashes, full_hashes, done), done)
                if start < done:
                    manifest = checkpoint.truncate(root, manifest, start)
                    done = start
                    原因 = "手动指定「重跑起始段」" if reroll > 0 else "检测到该段提示词已修改"
                    report.append(f"存档续跑：{原因}，从段 {start + 1} 起重新生成"
                                  + (f"（段 1-{start} 沿用存档）" if start else "（整链重做）"))
                seeds = [int(s) for s in manifest.get("seeds", [])]
                report.append(f"存档续跑：载入已完成 {done}/{len(seg_prompts) + off} 段，目录 {root}")
                if reroll == 0 and seeds:
                    report.append("控件种子仅在「重跑起始段」> 0 时生效，当前沿用存档种子序列")
        full_hashes = ([prologue_hash] if off else []) + seg_hashes
        if review and autosave:
            report.append(f"审片：分段视频每段落盘 → output/h3_auto/{os.path.basename(root) or '…'}/seg_XXX.mp4，"
                          "成片为运行结束时的已确认部分")

        pbar = comfy.utils.ProgressBar(len(seg_prompts))
        total = len(seg_prompts) + off
        prompt_list = (["「序章（上传视频）」"] if off else []) + list(seg_prompts)
        thumbs, videos, seams, bridge_scores = [], [], [], []
        all_frames = []
        all_wav = None
        seg_frames = []
        seg_wavs = []
        trims = []
        guide = None
        prev_tail_frame = None
        prev_tail_wav = None
        prev_lat = None   # 上段 (video_t, audio_t, kept t0, kept t1, audio a0, audio a1)——接缝精修窗口切片用
        sample_rate = None

        if use_ckpt:  # 运行起点状态（面板据此定位当前链）
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll, "report": "",
                                   "updated_at": time.time()})

        if off:
            prologue_fresh = False
            if use_ckpt and done >= 1:
                pv, pa = checkpoint.load_segment(root, 0)
                pv = pv.to(video_vae.device)
                pa = pa.to(audio_vae.device)
                prologue_origin = "存档载入"
            else:
                raw_fc = int(起始视频.shape[0])
                fc = align_frame_count_down(min(raw_fc, length))
                if fc < 5:
                    raise ValueError("起始视频至少需要 5 帧（约 0.2 秒 @24fps），请换更长的视频")
                pv = video_vae.encode(_center_cover(起始视频[:fc], width, height))
                if 起始视频音轨 is not None:
                    pa = _encode_audio_latent(audio_vae, 起始视频音轨, audio_tokens_for_frames(fc))
                else:
                    pa = torch.zeros(1, 32, 2, audio_tokens_for_frames(fc), device=video_vae.device)
                prologue_origin = f"编码{fc}帧"
                prologue_fresh = True
                if use_ckpt:
                    checkpoint.save_segment(root, 0, pv, pa)
                    seeds = [0]
                    done = 1
                    checkpoint.save_manifest(root, {
                        "schema": checkpoint.SCHEMA, "done": 1, "has_prologue": True,
                        "seeds": [0], "trims": [0], "prompt_hashes": [prologue_hash],
                        "total": total, "thumbs": [], "videos": [], "prompts": prompt_list,
                        "seams": [None], "bridge_scores": [None], "params": ckpt_params,
                        "seam_refine": seam_refine})
                report.append(f"序章：上传视频编码为段 1/{total}（{fc} 帧"
                              + ("，超长仅取前段" if raw_fc > fc else "")
                              + "，经一次 VAE 重编码，按 24fps 处理"
                              + ("，未接音轨按静音处理" if 起始视频音轨 is None else "") + "）")
            pframes = video_vae.decode(pv)
            if len(pframes.shape) == 5:
                pframes = pframes.reshape(-1, pframes.shape[-3], pframes.shape[-2], pframes.shape[-1])
            pwav, sample_rate = _decode_audio(audio_vae, pa)
            thumbs.append(checkpoint.save_thumb(root, 0, pframes[0]) if use_ckpt else "")
            videos.append(checkpoint.save_segment_mp4(root, 0, pframes, pwav, sample_rate,
                                                      fresh=prologue_fresh) if use_ckpt else "")
            seams.append(None)
            bridge_scores.append(None)
            all_frames.append(pframes.cpu())
            seg_frames.append(pframes.cpu())
            seg_wavs.append({"waveform": pwav.cpu(), "sample_rate": sample_rate})
            all_wav = pwav.cpu()
            prev_tail_frame = pframes[-1].cpu()
            seam_n0 = max(1, int(sample_rate * 0.25))
            prev_tail_wav = pwav.cpu()[..., -seam_n0:]
            guide = _tail_keyframe(pv, pa, ctx, KEYFRAME_AUDIO_SUPPORTED and full_bridge,
                                   full_bridge=full_bridge)
            prev_lat = (pv, pa, 0, pv.shape[2], 0, pa.shape[-1])
            report.append(f"段1/{total}：{prologue_origin} 序章 留{pframes.shape[0]}帧 · 种子 — | guide=无（序章）")

        def _decode_crop(i, video_t, audio_t, skip_f):
            """解码 → 桥帧门控 → 尾切 token 对齐 → 裁剪到保留区（重摇与正常路径共用）。

            返回 (frames, wav, sample_rate, end_t, vis_len, 桥帧总分, 门控报告行)；
            报告行只取最终采用的尝试（重摇的中间尝试整组丢弃）。
            """
            lines = []
            gi = i + off
            sampled_fc = latent_t_to_frames(video_t.shape[2])
            vis_len = length
            frames = video_vae.decode(video_t)
            if len(frames.shape) == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            # 音频归一化只统计保留区（见 _decode_audio）：锚定区音频不计入 std
            wav, sample_rate = _decode_audio(audio_vae, audio_t,
                                             norm_skip_frac=skip_f / sampled_fc if skip_f else 0.0)
            seg_bridge_score = None
            if 桥帧门控 != "关闭" and i + 1 < len(seg_prompts):
                window = frames[max(skip_f, frames.shape[0] - (ctx + gate_limit)):]
                score = qc.frame_scores(window)
                tail_score = float(score[-1])
                seg_bridge_score = round(tail_score, 2)
                if 桥帧门控 == "标注":
                    flag = " ↓ 低于阈值" if tail_score < 清晰度阈值 else ""
                    lines.append(f"段{gi + 1} 桥帧总分 {tail_score:.1f}{flag}")
                else:  # 自动回退
                    back, hit = qc.pick_backtrack(score, gate_limit, float(清晰度阈值))
                    if back:
                        vis_len = length - back
                        lines.append(f"段{gi + 1} 尾帧低质（{tail_score:.1f} < {清晰度阈值:g}），"
                                     f"回退 {back} 帧续拍（回退点 {hit:.1f}）")
            # 智能切镜：在段尾搜索运动低谷（自然停顿点），多余帧丢弃。
            # 运动低谷 = 动作完成/暂停 = 自然切镜点；与桥帧门控互补。
            if 智能切镜 == "自动" and i + 1 < len(seg_prompts):
                cut = qc.find_cut_point(frames, skip_f, vis_len)
                if cut is not None:
                    cut_f, cut_motion, cut_quality = cut
                    new_vis_len = cut_f - skip_f
                    if 0 < new_vis_len < vis_len:
                        trimmed = vis_len - new_vis_len
                        vis_len = new_vis_len
                        lines.append(f"段{gi + 1} 智能切镜：运动低谷 @帧{cut_f}"
                                     f"（运动 {cut_motion:.4f} 清晰 {cut_quality:.1f}），"
                                     f"丢弃尾部 {trimmed} 帧")
            # 尾切对齐 token 网格：kept 末端必须与 guide 锚定末端重合。此前 guide
            # 取采样 latent 原始尾部（含 17k+5 网格填充帧，从未输出），下段续拍点
            # 落在本段输出末尾之后——ctx=56 时每个接缝跳过 15 帧（0.6s）内容。
            # 门控回退时向下对齐（不回吐坏帧），否则向上（不丢内容，每段至多多留几帧）
            end_t = frames_to_latent_t(skip_f + vis_len, up=(length - vis_len) == 0)
            vis_len = latent_t_to_frames(end_t) - skip_f
            frames = frames[skip_f:skip_f + vis_len]
            wav_total = wav.shape[-1]
            skip_s = round(wav_total * skip_f / sampled_fc)
            take_s = round(wav_total * vis_len / sampled_fc)
            wav = wav[..., skip_s:skip_s + take_s]
            return frames, wav, sample_rate, end_t, vis_len, seg_bridge_score, lines

        for i, prompt in enumerate(seg_prompts):
            g = i + off  # 全局段下标（有序章时序章占 0 号）
            replay = use_ckpt and g < done
            skip_f = 0 if (i == 0 and off == 0) else ctx
            if replay:
                video_t, audio_t = checkpoint.load_segment(root, g)
                video_t = video_t.to(video_vae.device)
                audio_t = audio_t.to(audio_vae.device)
                dt = 0.0
                cur_seed = seeds[g] if g < len(seeds) else None
                frames, wav, sample_rate, end_t, vis_len, seg_bridge_score, gate_lines = \
                    _decode_crop(i, video_t, audio_t, skip_f)
            else:
                # 种子规则：重摇（重跑起始段>0）用控件种子；否则延续断点种子序列的等差，
                # 使审片多轮运行与一次跑完逐帧一致；断点无生成段种子（仅序章/新链）才用控件种子
                if reroll > 0:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                elif len(seeds) > off:
                    cur_seed = (seeds[-1] + g - len(seeds) + 1) % 0xffffffffffffffff
                else:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                seg_len = length + (0 if skip_f == 0 else ctx)
                if has_refs:
                    out = MiniMaxH3ReferenceToVideo.execute(
                        clip=clip, vae=video_vae, audio_vae=audio_vae,
                        prompt=prompt, width=width, height=height, length=seg_len,
                        ref_image_size="match", **refs)
                else:
                    out = MiniMaxH3ImageToVideo.execute(
                        clip=clip, vae=video_vae, prompt=prompt,
                        width=width, height=height, length=seg_len,
                        first_frame=首帧图片 if i == 0 else None)

                cond, latent = out[0], out[1]
                if guide is not None or tail_anchor_latent is not None:
                    cond = cls._apply_guide(
                        cond, guide, latent_t_to_frames(latent["samples"].tensors[0].shape[2]),
                        tail_kf_latent=tail_anchor_latent)
                    # 锚定加噪（SkyReels-V2 addnoise_condition 思路）：H3 模型 payload
                    # 原生支持 cond 噪声增强（extra_conds 从 cond dict 任意键取参），
                    # aug=1.0 即不加噪；值越小锚定越「软」，缓解段首刹车/内容重演
                    if aug > 0.0:
                        cond = _apply_anchor_noise(cond, aug)

                # 接缝自动重摇：本段生成后若缝差超阈值，换种子重采本段（上限内），
                # 排除抽卡坏段（同参数下缝差 0.02-0.17 波动大）。cond/latent 与种子
                # 无关只构造一次；回放段不参与。各次尝试取缝差最小的一组（CPU 快照，
                # 不占显存；末次反而更差时回切）。最终种子进 manifest（重放可复现）
                reroll_max = max(0, int(重摇上限)) if (接缝重摇 == "自动" and skip_f) else 0
                attempt = 0
                d_raw = None
                best = None   # (缝差, 结果快照)
                while True:
                    # 共存 H3 插件可能丢 keyframe/refs 音频导致 cond_audio 行数错位，
                    # 采样期挂模型层兜底（见 cond_audio_rows_guard），完成后恢复
                    restore_audio_rows = cond_audio_rows_guard(模型.model.diffusion_model)
                    # 递减锚定：visual_cond_noise_aug 随采样进度从强到弱递减到消失。
                    # 开启时覆盖 _apply_anchor_noise 写入的固定值——每步动态计算
                    restore_step_aug = None
                    if fade_ratio > 0 and (guide is not None or tail_anchor_latent is not None):
                        aug_start = 1.0 - aug if aug > 0 else 0.999
                        restore_step_aug = step_cond_noise_guard(
                            模型.model.diffusion_model, aug_start, 0.0, fade_ratio)
                    try:
                        t0 = time.perf_counter()
                        sampled = nodes.common_ksampler(
                            模型, cur_seed, 步数, CFG,
                            采样器, 调度器, cond, negative, latent, denoise=1.0)[0]
                    finally:
                        restore_audio_rows()
                        if restore_step_aug:
                            restore_step_aug()
                    dt = time.perf_counter() - t0
                    video_t, audio_t = sampled["samples"].unbind()
                    frames, wav, sample_rate, end_t, vis_len, seg_bridge_score, gate_lines = \
                        _decode_crop(i, video_t, audio_t, skip_f)
                    d_raw = None
                    if prev_tail_frame is not None:
                        d_raw = qc.seam_metrics(prev_tail_frame, frames[0])[0]
                    if best is None or (d_raw is not None and (best[0] is None or d_raw < best[0])):
                        best = (d_raw, (frames.cpu(), wav.cpu(), sample_rate,
                                        video_t, audio_t, end_t, vis_len,
                                        seg_bridge_score, list(gate_lines), cur_seed))
                    if d_raw is None or d_raw <= float(重摇阈值) or attempt >= reroll_max:
                        break
                    attempt += 1
                    cur_seed = (cur_seed + 7919) % 0xffffffffffffffff
                    report.append(f"段{g + 1} 接缝 {d_raw:.3f} > {重摇阈值:g}，自动换种子重摇（{attempt}/{reroll_max}）")
                if attempt:
                    if best[0] is not None and (d_raw is None or best[0] < d_raw):
                        (frames, wav, sample_rate, video_t, audio_t, end_t, vis_len,
                         seg_bridge_score, gate_lines, cur_seed) = best[1]
                        report.append(f"段{g + 1} 重摇 {attempt} 次取缝差最小（{best[0]:.3f} < 末次 {d_raw:.3f}）")
                        d_raw = best[0]
                    report.append(f"段{g + 1} 重摇后接缝 {d_raw:.3f} · 种子 {cur_seed}")
                # 维持不变量 seeds[g] = 该段种子（段文件缺失导致 done 回退时，
                # 重做段的种子可能与 manifest 残留记录相同，按下标赋值避免列表重复错位）
                if g < len(seeds):
                    seeds[g] = cur_seed
                else:
                    seeds.append(cur_seed)
                if use_ckpt:
                    checkpoint.save_segment(root, g, video_t, audio_t)
            report.extend(gate_lines)
            bridge_scores.append(seg_bridge_score)
            trims.append(length - vis_len)

            # 段首响度对齐（与分镜链同款）：增益匹配上段尾 RMS（±6dB 钳制 + 1s 渐出），
            # 增益不沿链累积；归一化已排除锚定区，此处兜住内容本身的响度差
            if (i > 0 or off) and prev_tail_wav is not None:
                wav, gain_db = qc.loudness_align_head(wav, prev_tail_wav, rate=sample_rate)
                if gain_db is not None:
                    report.append(f"段{g + 1} 响度对齐：段首 {gain_db:+.1f} dB（1s 渐出）")

            # 接缝后验测量（测而不干预）：上一段最后可见帧 vs 本段首帧
            seam_n = int(sample_rate * 0.25)
            seam_d, seam_db = None, None
            if (i > 0 or off) and prev_tail_frame is not None:
                seam_d, seam_db = qc.seam_metrics(prev_tail_frame, frames[0],
                                                  prev_tail_wav, wav[..., :seam_n], rate=sample_rate)
                db_txt = f"{seam_db:+.1f} dB" if seam_db is not None else "—"
                flag = " ↑ 建议人工检查" if seam_d > 0.08 or (seam_db is not None and abs(seam_db) > 6.0) else ""
                report.append(f"段{g + 1} 接缝：帧差 {seam_d:.3f} · 响度跳变 {db_txt}{flag}")

            # 接缝处理（生成后、拼接前）：潜空间精修 / smoothstep 像素混合 / 关闭。
            # 精修=跨缝窗口联合重去噪（refine.py）：缝两侧出自同一次去噪，根治
            # 段首偏差；只替换本段头部缝后侧帧，上段帧/存档/分段 mp4 均不动。
            # 纯后处理不进断点指纹（改此参数不触发重跑，replay 段结果仍一致）
            refine_used = False
            if (i > 0 or off) and prev_tail_frame is not None and prev_lat is not None \
                    and 接缝处理 != "关闭":
                if 接缝处理 == "潜空间精修":
                    ck0 = video_latent_t(skip_f)
                    cur_lat_ctx = (video_t, audio_t, ck0, end_t,
                                   audio_tokens_for_frames(skip_f),
                                   audio_tokens_for_frames(skip_f + vis_len))
                    win = refine.build_seam_window(prev_lat, cur_lat_ctx, int(精修窗口))
                    if win is None:
                        report.append(f"段{g + 1} 精修窗口不足（保留区不足每侧 {精修窗口} 帧），回退像素平滑")
                    else:
                        win_v, win_a, vt_p, vt_c, wf = win
                        # 缝前侧（上段尾）全部 video latent 作为 keyframe 注入 cond
                        # （与 _tail_keyframe 同模式：resolved_frame_index=0，锚定窗口
                        # 前 vt_p 帧=上段尾内容）。模型去噪时知道缝后侧应延续缝前侧，
                        # 否则只靠 prompt+refs 自由生成——段2无效(0.170→0.170)、
                        # 段3变差(0.032→0.055)都是因为 cond 缺 keyframe 引导
                        seam_kf_lat = win_v[:, :, :vt_p].clone()
                        try:
                            t1 = time.perf_counter()
                            refined_v = refine.refine_seam(
                                模型, negative, prompt, refs if has_refs else None,
                                clip, video_vae, audio_vae, win_v, win_a, wf,
                                width, height, 精修强度,
                                (cur_seed + 1) % 0xffffffffffffffff if cur_seed is not None else 1,
                                步数, CFG, 采样器, 调度器,
                                seam_kf_latent=seam_kf_lat, seam_kf_index=0)
                            wframes = video_vae.decode(refined_v)
                            if len(wframes.shape) == 5:
                                wframes = wframes.reshape(-1, wframes.shape[-3],
                                                          wframes.shape[-2], wframes.shape[-1])
                            # 丢弃缝前侧（上段侧），只保留本段替换区
                            wframes = wframes[latent_t_to_frames(vt_p):]
                            replace_n = min(wframes.shape[0], frames.shape[0])
                            feather = min(max(1, int(混合帧数)), replace_n)
                            if feather > 1:
                                # 精修区末端 smoothstep 渐变回原帧，防精修边界出现第二条微缝
                                # （w: 0→1 沿羽化推进，末帧=原帧与未精修区无缝衔接）
                                w = qc.smoothstep_weights(feather, device=wframes.device,
                                                          dtype=wframes.dtype).view(-1, 1, 1, 1)
                                wframes[-feather:] = wframes[-feather:] * (1.0 - w) + \
                                    frames[:replace_n][-feather:].to(wframes.device) * w
                            frames[:replace_n] = wframes[:replace_n].to(frames.device)
                            refine_used = True
                            d2, _ = qc.seam_metrics(prev_tail_frame, frames[0])
                            if seam_d is not None:
                                report.append(f"段{g + 1} 接缝精修：{seam_d:.3f} → {d2:.3f}"
                                              f" · 窗口 {wf} 帧 · {time.perf_counter() - t1:.0f}s")
                            seam_d = d2
                        except Exception as e:
                            report.append(f"段{g + 1} 接缝精修异常（{type(e).__name__}），回退像素平滑")
                if not refine_used:
                    # smoothstep 像素兜底：锚帧硬锁 + 权重窗吸收（偏差大时有叠影感）
                    span = min(max(1, int(混合帧数)), frames.shape[0])
                    frames = qc.smoothstep_blend_head(frames, prev_tail_frame, span)
                    report.append(f"段{g + 1} 接缝平滑：首帧硬锁锚帧 + smoothstep {span} 帧过渡")
            if (i > 0 or off) and prev_tail_frame is not None:
                seams.append([round(seam_d, 4), None if seam_db is None else round(seam_db, 2)])
            else:
                seams.append(None)

            all_frames.append(frames.cpu())
            seg_frames.append(frames.cpu())
            seg_wav = wav.cpu()
            seg_wavs.append({"waveform": seg_wav, "sample_rate": sample_rate})
            all_wav = seg_wav if all_wav is None else torch.cat([all_wav, seg_wav], dim=-1)
            prev_tail_frame = frames[-1].cpu()
            prev_tail_wav = seg_wav[..., -seam_n:]
            prev_lat = (video_t, audio_t, video_latent_t(skip_f), end_t,
                        audio_tokens_for_frames(skip_f),
                        audio_tokens_for_frames(skip_f + vis_len))
            if use_ckpt:
                thumbs.append(checkpoint.save_thumb(root, g, frames[0]))
                videos.append(checkpoint.save_segment_mp4(root, g, frames, wav, sample_rate,
                                                          fresh=not replay))
                if autosave:
                    _autosave_copy_seg(root, g)
            else:
                thumbs.append("")
                videos.append("")

            if guide is not None:
                note = (f"guide=上段尾{ctx}帧" if full_bridge else "guide=单帧桥(旧协议降级)") \
                    + ("+音频" if "audio_latent" in guide else "")
            else:
                note = "guide=无（首段）"
            origin = "存档载入" if replay else f"采样{latent_t_to_frames(video_t.shape[2])}帧"
            seed_txt = cur_seed if cur_seed is not None else "—"
            report.append(f"段{g + 1}/{total}：{origin} 裁头{skip_f}帧 留{frames.shape[0]}帧"
                          f" · 种子 {seed_txt}" + ("" if replay else f" · 采样 {dt:.0f}s") + f" | {note}")

            if i + 1 < len(seg_prompts):
                # end_tokens=kept 末端：锚定末端与输出末端重合（回退量已含在 vis_len 里）
                guide = _tail_keyframe(video_t, audio_t, ctx, KEYFRAME_AUDIO_SUPPORTED and full_bridge,
                                       end_tokens=end_t, full_bridge=full_bridge)

            if use_ckpt and not replay:
                done = g + 1
                checkpoint.save_manifest(root, {
                    "schema": checkpoint.SCHEMA, "done": done, "has_prologue": bool(off),
                    "seeds": list(seeds[:done]), "trims": list(trims[:done]),
                    "prompt_hashes": full_hashes[:done],
                    "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                    "prompts": prompt_list[:done],
                    "seams": seams[:done], "bridge_scores": bridge_scores[:done],
                    "params": ckpt_params, "seam_refine": seam_refine,
                })

            pbar.update(1)

            if review and not replay and i + 1 < len(seg_prompts):
                report.append(f"审片：段 {g + 1} 已完成并落盘 → 看自动保存目录里的 seg_{g:03d}.mp4；"
                              f"满意请直接重新运行继续段 {g + 2}；不满意：改该段提示词后运行（自动从本段重跑），"
                              f"或设「重跑起始段={g + 1}」+ 换种子重摇")
                break

        if use_ckpt:
            # 兜底回写：纯回放运行（无新段）也会重解码全部段，把缩略图/分段视频/指标
            # 等增量键补齐——旧版本存档的 manifest 缺这些键时由此自愈，无需重跑整链
            checkpoint.save_manifest(root, {
                "schema": checkpoint.SCHEMA, "done": done, "has_prologue": bool(off),
                "seeds": list(seeds[:done]), "trims": list(trims[:done]),
                "prompt_hashes": full_hashes[:done],
                "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                "prompts": prompt_list[:done],
                "seams": seams[:done], "bridge_scores": bridge_scores[:done],
                "params": ckpt_params, "seam_refine": seam_refine,
            })
            if review:
                if len(seg_frames) == total:
                    report.append("审片：本链已全部完成")
                if reroll > 0:
                    report.append(f"注意：「重跑起始段」={reroll} 已生效，确认无误后请改回 0")
            report.append(f"存档目录（含中间 latent，跑完可删）：{root}")
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll,
                                   "report": "\n".join(report), "updated_at": time.time()})

        images = torch.cat(all_frames, dim=0)
        # 自动保存：完整链（或审片已确认部分）编码成片到 output/h3_auto/<存档名>/
        if autosave and use_ckpt:
            final_name = _autosave_final(root, images, all_wav, sample_rate)
            if final_name:
                report.append(f"自动保存：分段与成片已就绪 → output/h3_auto/{os.path.basename(root)}/"
                              f"（seg_XXX.mp4 逐段，{os.path.basename(final_name)} 完整成片）")
            else:
                report.append("自动保存：成片编码失败（缺 PyAV 或编码异常），分段 mp4 不受影响")
        # 链路总结：接缝指标一览（seams[i] = [帧差, 响度dB] 或 None）
        measured = [(g, s[0], s[1]) for g, s in enumerate(seams) if s]
        if measured:
            avg_d = sum(m[1] for m in measured) / len(measured)
            worst = max(measured, key=lambda m: m[1])
            line = f"链路完成：{total} 段 · 接缝平均帧差 {avg_d:.3f} · 最差接缝 段{worst[0] + 1}（{worst[1]:.3f}"
            if worst[2] is not None:
                line += f"，{worst[2]:+.1f} dB"
            report.append(line + "）")
        return io.NodeOutput(
            images,
            {"waveform": all_wav, "sample_rate": sample_rate},
            24,
            "\n".join(report),
            seg_frames,
            seg_wavs,
        )

    @classmethod
    def IS_CHANGED(cls, 审片模式="关闭", 自动存档="关闭", 自动保存="分段+成片", **kwargs):
        if 审片模式 == "逐段确认" or 自动存档 in ("自动存档", "自动续跑") or 自动保存 == "分段+成片":
            return float("nan")   # 存档/审片/自动保存激活时输入不变也强制真正执行（重读最新 manifest）
        return ""

    @staticmethod
    def _apply_guide(cond, guide, sampled_fc, tail_kf_latent=None):
        """把 keyframe 注入 conditioning（官方 minimax_keyframes 协议）。

        guide: 首帧引导桥 keyframe（上段尾 latent 切片），None=首段无桥。
        tail_kf_latent: 尾帧身份锚定的 VAE latent，注入到 resolved_frame_index=
        sampled_fc-1 位置——与首帧桥形成「隧道」，模型去噪全程被首尾双锚点约束。
        合并 cond 里已有的 keyframes（如 i2v 首帧图片的 first_frame keyframe）。
        """
        existing = cond[0][1].get("minimax_keyframes", [])
        keyframes = list(existing)
        if guide is not None:
            keyframes.append(guide)
        if tail_kf_latent is not None:
            keyframes.append({"resolved_frame_index": sampled_fc - 1,
                              "latent": tail_kf_latent})
        if not keyframes:
            return cond
        return node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": keyframes,
            "minimax_frame_count": sampled_fc,
        })
