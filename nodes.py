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
「自动保存=分段+成片」时每段 mp4 与完整成片自动落盘项目文件夹
output/h3_projects/<项目名>/（游戏式存档：一项目一文件夹，导演台可读档/删除）。

兼容性：不 monkey-patch；conditioning/latent 构造直接调用官方
MiniMaxH3ImageToVideo / MiniMaxH3ReferenceToVideo 节点类，采样走官方
common_ksampler。多 token 桥与 keyframe 音频需要 ComfyUI 含 PR #15439
（2026-08-09 之后构建）；旧版 PackedLayout 每个 keyframe 只分配单帧 latent
行数，钉多 token 桥会形状错位，运行时自动探测并降级为单帧桥（报告说明）。
"""

import os
import re
import time
import json

import torch
import nodes
import node_helpers
import folder_paths
import comfy.utils
import comfy.samplers
from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo
from comfy_api.latest import io

from . import checkpoint
from . import metrics
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


def _autosave_final(root, frames, wav, sample_rate, fps=24):
    """完整链（或审片已确认部分）PyAV 编码成片到项目文件夹，成功返回路径。

    失败时返回 (None, error_msg) 供调用方写入报告；成功返回 (path, None)。
    """
    from . import media
    try:
        path = os.path.join(root, f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        print(f"[H3自动保存] 编码完整成片：{int(frames.shape[0])} 帧 → {path}（编码期间 CPU 升高属正常）")
        t0 = time.time()
        ok = media.save_av_mp4(path, frames, wav, sample_rate, fps)
        print(f"[H3自动保存] 成片编码{'完成' if ok else '失败'}：{time.time() - t0:.0f}s")
        return (path, None) if ok else (None, media.last_error)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
        print(f"[H3自动保存] 成片编码异常：{err}")
        return None, err


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


def _load_input_image(filename):
    """从 ComfyUI input 目录加载图片为 IMAGE 张量（与 LoadImage 节点同格式：[1,H,W,C] float32 0-1）。"""
    from PIL import Image, ImageOps
    import numpy as np
    image_path = folder_paths.get_annotated_filepath(str(filename))
    img = node_helpers.pillow(Image.open, image_path)
    img = node_helpers.pillow(ImageOps.exif_transpose, img)
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)


def _load_input_video(filename):
    """input 目录视频 -> (帧张量, 音轨)：与 LoadVideo+GetVideoComponents 画布链同格式。

    音轨与帧同源自同一文件，即「参考视频音轨」自动配对，无需单独上传。
    """
    from comfy_api.latest import InputImpl
    video_path = folder_paths.get_annotated_filepath(str(filename))
    comps = InputImpl.VideoFromFile(video_path).get_components()
    if comps.images is None or not getattr(comps.images, "shape", None):
        raise ValueError(f"参考视频「{filename}」解不出画面帧：请确认是可解码的 24fps 视频（2-15 秒）")
    if comps.audio is None:
        import torch as _t
        comps.audio = {"waveform": _t.zeros(1, 1, 0), "sample_rate": 24000}
    return comps.images, comps.audio


def _load_input_audio(filename):
    """input 目录音频 -> AUDIO dict（与 LoadAudio 节点输出同格式）。"""
    from comfy_extras.nodes_audio import load
    audio_path = folder_paths.get_annotated_filepath(str(filename))
    waveform, sample_rate = load(audio_path)
    return {"waveform": waveform.unsqueeze(0), "sample_rate": sample_rate}


def _parse_director_state(raw):
    """解析导演台状态 JSON。空/非法时返回空 dict，不影响原有画布接线工作流。"""
    if not raw:
        return {}
    try:
        d = json.loads(str(raw)) if isinstance(raw, str) else raw
        return d if isinstance(d, dict) else {}
    except (TypeError, ValueError):
        return {}


# ---- 官方 Resolution Selector 同款画布换算 + 时长网格吸附 ----

_AR_RATIO = {"21:9": 21 / 9, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1.0}
_MP_OPTIONS = [round(i * 0.1, 1) for i in range(1, 21)]   # 0.1–2.0MP，0.1 步进（官方箭头微调同款）
_LABEL_TOKEN = re.compile(r"\[\[([^\[\]]{1,24})\]\]")


def _resolve_canvas(ar, mp):
    """宽高比+百万像素 -> 画布（官方 Resolution Selector 公式逐位移植，comfy_extras/nodes_resolution.py）。

    1MP = 1024×1024 = 1048576 px（官方口径，非 1e6）；两侧各自 round 对齐 32 倍数，无上限收敛。
    16:9 档位对照官方节点：0.2→608×352 / 0.5→960×544 / 0.98→1344×768（H3 原生）/ 1.0→1376×768 /
    1.2→1504×832 / 1.5→1664×928 / 2.0→1920×1088；9:16+1.0 → 768×1376。
    """
    r = _AR_RATIO[str(ar)]
    total = float(mp) * 1024 * 1024
    w0, h0 = (total * r) ** 0.5, (total / r) ** 0.5
    return round(w0 / 32) * 32, round(h0 / 32) * 32


def _snap_seconds(seconds):
    """秒 -> 就近的 17k+5 帧网格（@24fps，≥5 帧）。5.0s→124，与旧默认帧数一致。"""
    f = max(5, int(round(float(seconds) * 24)))
    k = max(0, round((f - 5) / 17))
    return 17 * k + 5


REF_CAPS = {"image": 9, "video": 3, "audio": 3}   # 官方单段参考上限（nodes_minimax_h3 autogrow max）
_KIND_NAME = {"image": "图片", "video": "视频", "audio": "音频"}


def _normalize_order(label_order):
    """段引用顺序 -> [(kind, 标签)]；纯字符串元素按旧格式视为 image（兼容旧调用/旧测试）。"""
    out = []
    for x in label_order:
        out.append(("image", x) if isinstance(x, str) else (str(x[0]), str(x[1])))
    return out


def _kind_tokens(label_order):
    """[(kind, 标签)] -> 按类别独立编号的 {标签: token}（图 <Picture k> / 视 <Video k> / 音 <Audio j>）。"""
    counters = {"image": 0, "video": 0, "audio": 0}
    mapping = {}
    for kind, lbl in label_order:
        counters[kind] = counters.get(kind, 0) + 1
        token = {"image": "<Picture {}>", "video": "<Video {}>", "audio": "<Audio {}>"}.get(kind)
        if token is None:
            raise ValueError(f"未知素材类别「{kind}」：必须是 image / video / audio")
        mapping[lbl] = token.format(counters[kind])
    return mapping


def _apply_label_tokens(prompt, label_order):
    """把提示词中的 [[标签]] 替换为该段压实编号后的 <Picture k>/<Video k>/<Audio j>。

    label_order 为该段按序引用的 (kind, 标签)（或旧格式纯标签=image）；各类别独立从 1
    编号；长标签先替换，防「角色1」吃掉「角色10」前缀；剩余未知 [[..]] 报错并列出可用标签。
    """
    mapping = _kind_tokens(_normalize_order(label_order))
    out = prompt
    for lbl in sorted(mapping, key=len, reverse=True):
        out = out.replace(f"[[{lbl}]]", mapping[lbl])
    unknown = _LABEL_TOKEN.findall(out)
    if unknown:
        raise ValueError(f"提示词引用了未知素材标签「{unknown[0].strip()}」：可用标签 {list(mapping)}")
    return out


def _reference_header(label_order):
    """段首自动注入的参考定义（官方 Ref2VA subject_definitions 风格），三类 token 各自编号。"""
    mapping = _kind_tokens(_normalize_order(label_order))
    role = {"Picture": "reference image", "Video": "reference video", "Audio": "reference audio"}
    lines = [f"{tok} is the {role[tok.split()[0][1:].capitalize()]} for {lbl}, "
             f"used as a generation anchor of this segment."
             for lbl, tok in mapping.items()]
    return "subject_definitions:\n" + "\n".join(lines)


# 官方 h3-prompt-writing 三字段标签：主提示词含任一即视为已是官方格式，直通不包装
_OFFICIAL_FIELD_RE = re.compile(
    r"integrated_multimodal_description\s*:|overall_soundscape\s*:|detailed_description\s*:")


def _wrap_dialogue(text):
    """中文对白「……」→ 官方 <d>[中文] ……</d>。

    只动「」（中文对白惯例）；英文双引号是官方可见屏幕文字记法，不动。
    """
    return re.sub(r"「([^「」]*)」", r"<d>[中文] \1</d>", text)


def _compose_official(scene, char, prompt, soundscape, music):
    """按官方 h3-prompt-writing 结构组装单段提示词。

    - prompt 已是官方格式（三字段任一标签）→ 原样直通（含 [[标签]] 等待后续替换）
    - 否则组装三字段：描述体 = 场景。角色。主提示词（句号连接，各剥尾部标点），
      主提示词已以 [Shot 开头时不重复加 [Shot 1] 前缀
    - 环境音/配乐留空则整个字段省略（不写 N/A——官方 N/A = 明确请求无声，
      省略 = 不约束）
    - 主提示词未含 <d> 时自动把「……」转 <d>[中文] …</d>
    返回 (组装文本, 对白转换次数)。
    """
    prompt = prompt or ""
    if _OFFICIAL_FIELD_RE.search(prompt):
        return prompt, 0
    wrapped = prompt
    dialogues = 0
    if "<d>" not in prompt:
        wrapped = _wrap_dialogue(prompt)
        dialogues = len(re.findall(r"<d>", wrapped))
    parts = [str(p).strip().rstrip("。.；;，, ") for p in (scene, char, wrapped) if str(p).strip()]
    # 主提示词已带 [Shot N] 开头标签：剥掉防重复，统一在描述行首放一个前缀
    shot_tag = re.match(r"\s*(\[Shot\s*\d+\])\s*", str(wrapped))
    if shot_tag:
        parts = [str(p).strip().rstrip("。.；;，, ") for p in (scene, char) if str(p).strip()]
        parts.append(wrapped[shot_tag.end():].strip().rstrip("。.；;，, "))
        parts = [p for p in parts if p]
    body = "。".join(parts)
    lines = []
    if body:
        prefix = f"{shot_tag.group(1)} " if shot_tag else "[Shot 1] "
        lines.append(f"integrated_multimodal_description: {prefix}{body}")
    if str(soundscape or "").strip():
        lines.append(f"overall_soundscape: {str(soundscape).strip()}")
    if str(music or "").strip():
        lines.append(f"non_diegetic_music: {str(music).strip()}")
    return "\n".join(lines), dialogues


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
                io.Combo.Input("宽高比", options=["自定义", "21:9", "16:9", "9:16", "4:3", "3:4", "1:1"], default="16:9",
                               tooltip="官方 Resolution Selector 同款：与「百万像素」共同换算画布（1MP=1024×1024，32 倍数对齐）。"
                                       "选「自定义」时直接用下方宽度/高度"),
                io.Float.Input("百万像素", default=0.5, min=0.1, max=2.0, step=0.1,
                               tooltip="目标总像素（MP），0.1–2.0 步进 0.1，箭头微调（官方 Resolution Selector 同款口径）："
                                       "0.2 草稿（608×352）/ 0.5 快速预览（960×544）/ 0.98 H3 官方原生（1344×768）/ "
                                       "1.0（1376×768）/ 2.0 超采样（1920×1088）"),
                io.Int.Input("宽度", default=864, min=32, max=16384, step=32, advanced=True,
                             tooltip="「宽高比=自定义」时直接生效；其余模式由 宽高比+百万像素 换算覆盖"),
                io.Int.Input("高度", default=480, min=32, max=16384, step=32, advanced=True,
                             tooltip="「宽高比=自定义」时直接生效；其余模式由 宽高比+百万像素 换算覆盖"),
                io.Float.Input("每段时长", default=5.0, min=0.5, max=15.0, step=0.1,
                               tooltip="每段可见时长（秒）@24fps，内部自动吸附 H3 的 17k+5 帧网格："
                                       "5.0s→124帧、6.0s→141帧。全链默认值，导演台每段可单独覆盖"),
                io.Combo.Input("引导帧数", options=["5", "22", "39", "56"], default="22",
                               tooltip="段间引导重叠桥：钉入下段头部的上段尾帧数。越大衔接越顺、越慢越吃显存"),
                io.Int.Input("种子", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="第 i 段实际使用 种子+i"),
                io.Int.Input("步数", default=25, min=1, max=100),
                io.Float.Input("CFG", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("采样器", options=comfy.samplers.KSampler.SAMPLERS, default="res_multistep"),
                io.Combo.Input("调度器", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Combo.Input("自动存档", options=["关闭", "自动存档"], default="关闭", advanced=True,
                               tooltip="自动存档：每段采样后立即落盘 latent 存档（约5MB/段），中断后重跑自动跳过已完成段，结果与一次跑完一致。"
                                       "注意：「自动保存」默认开（分段+成片）且「审片模式」都会自动启用存档，"
                                       "本开关仅在两者全关时才有独立作用（纯 latent 续跑、不出 mp4 的调试场景）"),
                io.String.Input("存档目录", default="",
                                tooltip="项目名：output/h3_projects/<项目名>/ 一个项目一个文件夹（视频/提示词/成片/latent 全在内）。"
                                        "空=按参数指纹自动命名；填了名字即固定项目：中断重跑、改词重跑都续在这个文件夹"),
                io.Combo.Input("桥帧门控", options=["关闭", "标注", "自动回退"], default="标注",
                               tooltip="对将成为重叠桥的尾帧打分（Laplacian清晰度+曝光）：标注=只写报告；自动回退=尾帧低于阈值时向前回退17/34帧取好帧续拍（该段可见帧数随之减少）"),
                io.Float.Input("清晰度阈值", default=30.0, min=0.0, max=100.0, step=0.5,
                               tooltip="桥帧总分阈值，低于判定为坏尾。建议先跑「标注」档看报告里的分数分布再定"),
                io.Int.Input("回退上限", default=34, min=0, max=68, step=17,
                             tooltip="自动回退最多向前多少帧（17 的倍数，踩 17k+5 网格）"),
                io.Combo.Input("接缝处理",
                               options=["标准", "轻量", "强力", "自定义",
                                        "潜空间精修", "smoothstep像素混合", "关闭"],
                               default="标准",
                               tooltip="统一接缝处理入口（一个下拉决定全部衔接参数，细粒度控件仅「自定义/旧值档」生效）："
                                       "标准=跨缝窗口联合重去噪+双端锚定+按缝差自适应强度(0.30/0.45/0.55)，"
                                       "窗口39帧、羽化6帧，锚定加噪与递减锚定一律关闭（画质优先，推荐）；"
                                       "轻量=更保守的分档(0.22/0.30/0.40)+窗口22帧+羽化4帧，好缝几乎不动；"
                                       "强力=更强分档(0.40/0.55/0.65)+羽化8帧，坏缝多的链用；"
                                       "自定义=读下方「锚定加噪/递减锚定/精修强度/精修窗口/混合帧数/自适应精修」"
                                       "（自由组合易伤画质，仅供实验）；"
                                       "潜空间精修/smoothstep像素混合=旧版值（兼容旧工作流，行为=自定义档/纯像素混合）。"
                                       "纯后处理不进存档指纹，改参数不触发重跑"),
                io.Int.Input("混合帧数", default=6, min=1, max=24,
                             tooltip="仅接缝处理=自定义/旧值档时生效（预设档内部固化）："
                                     "smoothstep 模式=像素混合窗帧数（两端权重导数为 0、中段过渡；"
                                     "运动越快窗应越短，6帧≈0.25s）；潜空间精修模式=精修区末端渐变回原帧的"
                                     "羽化帧数，防精修边界出现第二条微缝"),
                io.Float.Input("锚定加噪", default=0.0, min=0.0, max=0.5, step=0.05,
                               tooltip="仅接缝处理=自定义/旧值档时生效（预设档一律关闭——实测对画质是净伤害）。"
                                       "对桥锚定帧注入噪声的比例（SkyReels-V2 addnoise_condition 思路）："
                                       "干净锚定帧会让模型起步「刹车」并在可见部分重演锚定内容；加噪让模型"
                                       "把锚定当「参考」而非「必须逐帧复现」。0=关闭（默认）；"
                                       "0.2 标准（SkyReels 同值）；0.3+ 干预强但画面细节会变软。"
                                       "仅影响带引导桥的段；不进存档指纹，改参数不触发重跑"),
                io.Combo.Input("审片模式", options=["关闭", "逐段确认"], default="关闭",
                               tooltip="逐段确认：每次运行只生成一个新的段落即返回，预览「分段图像」或项目文件夹里的"
                                       "分段视频后重新运行继续下一段；不满意可改该段提示词（自动从该段重跑）或设「重跑起始段」重摇。"
                                       "开启后存档自动启用"),
                io.Combo.Input("自动保存", options=["关闭", "分段+成片"], default="分段+成片",
                               tooltip="开启后无需任何下游接线：每段生成完自动存 mp4、链（或审片已确认部分）自动拼成"
                                       "完整成片，全部直接落在项目文件夹 output/h3_projects/<项目名>/ 内，报告注明路径。开启后存档自动启用"),
                io.Int.Input("重跑起始段", default=0, min=0, max=63,
                             tooltip="0=自动（沿用存档进度，改过提示词的段自动重做）；N=从第 N 段起丢弃存档重新生成"
                                     "（有序章时序章为第 1 段），配合改「种子」即可重摇该段及之后。用完记得改回 0"),
                io.Float.Input("精修强度", default=0.45, min=0.2, max=0.7, step=0.05,
                               tooltip="仅接缝处理=自定义/旧值档、且「自适应精修」关闭时生效（预设档内部固化）："
                                       "接缝精修的加噪/去噪强度（denoise），窗口两侧各保留 (1-强度) 的原结构。"
                                       "0.2-0.3 改动小、调和弱；0.45 标准；0.6+ 过渡更顺但纹理细节改动大"),
                io.Combo.Input("精修窗口", options=["22", "39", "56"], default="39",
                               tooltip="仅接缝处理=自定义/旧值档时生效（预设档内部固化）："
                                       "接缝精修每侧帧数（缝前取上段尾、缝后取本段头，均为 token 网格点）。"
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
                               tooltip="仅接缝处理=自定义/旧值档时生效（预设档一律关闭——实测对画质是净伤害）。"
                                       "锚定约束随采样进度递减：开始强（接缝吻合）→ 逐渐减弱（模型自然过渡）"
                                       "→ 完全消失（自由生成）。数值=递减占总步数比例（0.3=前30%步数递减完毕）。"
                                       "开启后「锚定加噪」的值作为递减起点，终点为 0（锚定消失）"),
                io.Int.Input("切镜最多丢帧", default=17, min=0, max=120,
                             tooltip="智能切镜的单段丢帧上限（0=只标注切镜点不裁剪）。"
                                     "此前切镜在段尾 1/3 无上限搜索，多段累计可丢数秒内容"
                                     "（30 秒视频被删 7 秒的主因）。进存档指纹，改值触发重跑"),
                io.Int.Input("全链丢弃预算", default=48, min=0, max=3600,
                             tooltip="整链累计丢弃帧数预算（智能切镜+门控回退+网格对齐合计，48帧=2秒）。"
                                     "超预算后智能切镜自动降级为只标注不裁剪，门控回退仍生效但记 WARN。"
                                     "报告末尾有累计丢弃汇总行。进存档指纹"),
                io.Combo.Input("自适应精修", options=["开启", "关闭"], default="开启",
                               tooltip="仅接缝处理=自定义/旧值档时生效（预设档内部固化分档）："
                                       "按缝差分档精修强度：<0.04 轻修0.30（保细节）；0.04-0.08 标准0.45；"
                                       ">0.08 强调和0.55。关闭=固定用「精修强度」值。固定 0.45 整窗重去噪"
                                       "会把好缝也重新去噪一遍——接缝发糊的主因之一。不进存档指纹"),
                io.Combo.Input("生成模式", options=["文生视频", "首帧视频", "多参视频"], default="文生视频",
                               tooltip="导演台一体化控制用：文生=纯文本（fl2va UNET，不接任何图片）；"
                                       "首帧=首帧图片起手（fl2va UNET，仅接「首帧图片」）；"
                                       "多参=参考图片（ref2va UNET，仅接「参考图片」组）。"
                                       "UI 据此互斥首帧/参考图；后端校验模式与素材一致性，不匹配时报错。"
                                       "实际 UNET 由「模型」输入决定，本控件只做模式声明与素材互斥。"),
                io.String.Input("导演台状态", multiline=True, default="", socketless=True, advanced=True,
                               tooltip="导演台前端写入的 JSON 状态（模式/提示词/素材文件名/分段处理）。"
                                       "有值时优先于画布接线：提示词从 JSON 读取，图片从 input 目录加载。"
                                       "first_frame=首帧图片、last_frame=尾帧锚定（身份锚点，任意模式可用）；"
                                       "ref_assets 为标签素材池 [{file,label}]；segments 数组为每段提供 "
                                       "scene_prompt / character_prompt / soundscape（环境音）/"
                                       "music（配乐，均留空省略）/ seconds（本段秒数）/"
                                       "refs（本段引用的素材标签，缺省=全部），提示词用 [[标签]] 引用素材，"
                                       "后端按官方 h3-prompt-writing 三字段结构组装（含对白「」→<d> 自动转换）。"
                                       "空值或旧工作流走原有画布输入（同样获得官方结构组装，向后兼容）。"
                                       "只存相对输入文件名，不存媒体内容、密钥或绝对路径。"),
                io.Image.Input("首帧图片", optional=True,
                               tooltip="第一段的起始帧（i2v）。用了它请用 fl2va UNET，且不能同时用任何参考素材"),
                io.Image.Input("尾帧锚定", optional=True,
                               tooltip="身份锚定帧（last_frame keyframe）：注入每段末尾位置作为人物/场景参考，"
                                       "与段首引导桥形成「隧道」——模型去噪全程被首尾双锚点约束，过了桥窗口"
                                       "（约2秒）也不会漂移。建议用角色正面清晰帧（导演台素材池可上传，"
                                       "任意模式可用）。不传则不锚定尾帧（保持现状）"),
                io.Image.Input("起始视频", optional=True,
                               tooltip="序章：上传视频（≥5 帧、24fps，超长只取前「每段时长」内）编码为第 1 段存入存档，"
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
    def execute(cls, 模型, 文本编码器, 视频VAE, 音频VAE, 宽高比, 百万像素, 宽度, 高度, 每段时长, 引导帧数,
                种子, 步数, CFG, 采样器, 调度器, 首帧图片=None, 尾帧锚定=None, 起始视频=None, 起始视频音轨=None,
                提示词组=None,
                参考图片组=None, 参考视频组=None, 参考视频音轨组=None, 参考音频组=None,
                自动存档="关闭", 存档目录="", 桥帧门控="标注", 清晰度阈值=30.0, 回退上限=34,
                接缝处理="标准", 混合帧数=6, 锚定加噪=0.0,
                审片模式="关闭", 自动保存="分段+成片", 重跑起始段=0,
                精修强度=0.45, 精修窗口="39", 接缝重摇="自动", 重摇阈值=0.06, 重摇上限=1,
                智能切镜="关闭", 递减锚定="关闭", 切镜最多丢帧=17, 全链丢弃预算=48,
                自适应精修="开启", 生成模式="文生视频", 导演台状态=""):
        # 运行期路由兜底：导入期注册因时序失败时，首次执行后前端删除/列表即可用
        try:
            from .routes import ensure_registered
            ensure_registered()
        except Exception:
            pass
        # ---- 导演台状态驱动：有 JSON 状态时优先于画布接线 ----
        ds = _parse_director_state(导演台状态)
        ds_used = bool(ds)
        if ds.get("mode"):
            _m = str(ds["mode"])
            if _m not in ("文生视频", "首帧视频", "多参视频"):
                raise ValueError(f"导演台状态中的模式「{_m}」无效：必须是「文生视频」「首帧视频」或「多参视频」")
            生成模式 = _m
        # 提示词：导演台 JSON 优先，空则回退画布 autogrow 输入
        if ds.get("prompts") and isinstance(ds["prompts"], list):
            seg_prompts = [str(p).strip() for p in ds["prompts"] if str(p).strip()]
        else:
            prompts = _autogrow_items(提示词组, "p")
            seg_prompts = [str(v).strip() for v in prompts.values() if str(v).strip()]
        if not seg_prompts:
            raise ValueError("提示词不能为空：请在导演台填写提示词，或在「提示词组」里每段添加一个输入框并填写内容")

        # 素材池（标签->文件，分 图/视/音 三类）：ref_assets 为 v2 真源；旧 ref_images 视为图片
        pool_files = []   # [(kind, label, file)]
        _kind_n = {"image": 0, "video": 0, "audio": 0}
        if isinstance(ds.get("ref_assets"), list) and ds["ref_assets"]:
            for item in ds["ref_assets"]:
                if not (isinstance(item, dict) and item.get("file")):
                    continue
                _k = str(item.get("kind") or "image")
                if _k not in _KIND_NAME:
                    _k = "image"
                label = str(item.get("label") or "").strip()
                if not label:
                    # 缺省标签只对未命名素材按类别计数（与前端 getDs 一致）
                    _kind_n[_k] += 1
                    label = f"{_KIND_NAME[_k]}{_kind_n[_k]}"
                pool_files.append((_k, label[:24], str(item["file"])))
        elif ds.get("ref_images") and isinstance(ds["ref_images"], list):
            pool_files = [("image", f"图片{i + 1}", str(fn)) for i, fn in enumerate(ds["ref_images"]) if fn]
        _seen = set()
        for _i, (_k, _lbl, _fn) in enumerate(pool_files):
            _base, _n = _lbl, 2
            while _lbl in _seen:
                _lbl = f"{_base}{_n}"
                _n += 1
            _seen.add(_lbl)
            pool_files[_i] = (_k, _lbl, _fn)
        pool_labels = [lbl for _, lbl, _ in pool_files]
        pool_kind = {lbl: k for k, lbl, _ in pool_files}
        pool_file_of = {lbl: fn for _, lbl, fn in pool_files}

        segments = ds.get("segments") if isinstance(ds.get("segments"), list) else []

        # 分段处理中心：官方三字段组装（场景/角色/环境音/配乐）；[[标签]] -> <Picture>/<Video>/<Audio>；
        # 段首自动 subject_definitions。画布模式（无 JSON 状态）同样获得官方结构。
        seg_composed = 0
        seg_custom_refs = 0
        seg_dialogues = 0
        seg_label_orders = [[] for _ in seg_prompts]   # 每段 [(kind, 标签)]，按勾选顺序
        composed_prompts = []
        for i, prompt in enumerate(seg_prompts):
            seg = segments[i] if i < len(segments) and isinstance(segments[i], dict) else {}
            scene = str(seg.get("scene_prompt", "")).strip()
            char = str(seg.get("character_prompt", "")).strip()
            soundscape = str(seg.get("soundscape", "")).strip()   # 旧 JSON 无此键 = 空
            music = str(seg.get("music", "")).strip()
            full, dlg = _compose_official(scene, char, prompt, soundscape, music)
            seg_dialogues += dlg
            if scene or char or soundscape or music:
                seg_composed += 1
            if pool_files:
                refs_sel = seg.get("refs")
                order = []
                if isinstance(refs_sel, list) and refs_sel:
                    for r in refs_sel:
                        lbl = str(r).strip()
                        if lbl not in pool_labels:
                            raise ValueError(f"段{i + 1} 引用了未知素材标签「{lbl}」：可用标签 {pool_labels}")
                        item = (pool_kind[lbl], lbl)
                        if item not in order:
                            order.append(item)
                    seg_custom_refs += 1
                    # 提示词里 [[标签]] 提到但没勾选的素材：按出现顺序并入，防勾选/文本失配报错
                    for tok in _LABEL_TOKEN.findall(full):
                        lbl = tok.strip()
                        if lbl in pool_labels and (pool_kind[lbl], lbl) not in order:
                            order.append((pool_kind[lbl], lbl))
                else:
                    # refs 缺省/显式空 = 引用全部素材（与旧行为一致，避免 ref2va 链空参考段）
                    order = [(k, lbl) for k, lbl, _ in pool_files]
                # 官方单段上限：图 9 / 视 3 / 音 3（超了报错并指出勾选明细）
                for _k, _cap in REF_CAPS.items():
                    _picked = [lbl for k, lbl in order if k == _k]
                    if len(_picked) > _cap:
                        raise ValueError(f"段{i + 1} 引用{_KIND_NAME[_k]}素材 {len(_picked)} 个，"
                                         f"超过官方单段上限 {_cap} 个（{_picked}）：请在段卡片少勾几个")
                seg_label_orders[i] = order
                if order:
                    has_pic = "<Picture" in full or "<Video" in full or "<Audio" in full
                    full = _apply_label_tokens(full, order)
                    if not has_pic:
                        full = _reference_header(order) + "\n" + full
            composed_prompts.append(full)
        seg_prompts = composed_prompts

        # 每段时长原始值：seg.seconds（None=跟随全局默认），吸附网格在 length 确定后统一做
        seg_secs = []
        for i in range(len(seg_prompts)):
            seg = segments[i] if i < len(segments) and isinstance(segments[i], dict) else {}
            try:
                v = seg.get("seconds")
                seg_secs.append(None if v in (None, "") else _snap_seconds(float(v)))
            except (TypeError, ValueError):
                seg_secs.append(None)

        # 段级独立镜头（断链）：与上段毫无关联的镜头一键全断——不接上段桥、不裁头、
        # 不做接缝处理/测量/重摇、上段不做桥帧门控与智能切镜、不做响度对齐（硬切转场）
        seg_unlink = []
        for i in range(len(seg_prompts)):
            seg = segments[i] if i < len(segments) and isinstance(segments[i], dict) else {}
            seg_unlink.append(bool(seg.get("unlink")))

        # 潜空间放大二采（导演台面板控制）：每段采样定稿后立即「神经放大 latent →
        # 低强度重采样 → 解码」，分段视频与成片直接保存二采结果（同名覆盖，不再
        # 产出 upseg_* 副本）。模型在主循环前预加载：缺失/不匹配当场降级为基础
        # 分辨率整链运行（报告注明），不让每段反复撞同一个错。关闭时 up_cfg=None
        from . import upscale
        up_cfg = upscale.parse_state(ds)
        up_net = None
        _up_err = None
        if up_cfg:
            try:
                up_net = upscale.load_net(up_cfg)
            except ValueError as e:
                _up_err = str(e)
                up_cfg = None

        # 插入视频段（导演台状态 inserts）：按链位混排进执行序列——画面+原声进成片，
        # 尾帧 latent 桥接指导下一段生成（序章机制的任意段间推广）。
        # 链位 pos = 提示词段+插入段混排后的 1-based 位置（不含序章），与前端 planFromDs 一致
        insert_specs = []
        _raw_inserts = ds.get("inserts") if isinstance(ds.get("inserts"), list) else []
        for it in _raw_inserts:
            if not (isinstance(it, dict) and str(it.get("file") or "").strip()):
                continue
            try:
                pos = int(it.get("pos"))
            except (TypeError, ValueError):
                raise ValueError(f"插入视频位置无效：{it.get('pos')!r}（须为正整数链位）")
            if not 1 <= pos <= len(seg_prompts) + len(_raw_inserts):
                raise ValueError(f"插入视频链位 {pos} 越界：有效范围 1-{len(seg_prompts) + len(_raw_inserts)}")
            insert_specs.append((pos, str(it["file"]).strip()))
        _seen_pos = set()
        for pos, _f in insert_specs:
            if pos in _seen_pos:
                raise ValueError(f"插入视频链位 {pos} 重复：每个链位只能插入一个视频")
            _seen_pos.add(pos)
        insert_specs.sort(key=lambda x: x[0])
        exec_items = []   # ("prompt", 提示词段索引) | ("insert", 文件名)，链位序
        _pi = 0
        for pos, fname in insert_specs:
            while len(exec_items) + 1 < pos and _pi < len(seg_prompts):
                exec_items.append(("prompt", _pi))
                _pi += 1
            exec_items.append(("insert", fname))
        while _pi < len(seg_prompts):
            exec_items.append(("prompt", _pi))
            _pi += 1

        # 首帧图片：导演台 JSON 优先，空则用画布连接
        if ds.get("first_frame"):
            首帧图片 = _load_input_image(ds["first_frame"])

        # 尾帧锚定（身份锚点，任意模式可用）：导演台 JSON 优先，空则用画布连接
        if ds.get("last_frame"):
            尾帧锚定 = _load_input_image(ds["last_frame"])

        # i2v 首段官方指令行（官方 I2VA 固定格式：声明首帧 = <Picture 1> 锚）；
        # 已是官方格式的段不注入（用户自管的完整结构里可能自带指令行）
        if 首帧图片 is not None and seg_prompts and not _OFFICIAL_FIELD_RE.search(seg_prompts[0]):
            seg_prompts[0] = ("For the target video, at 0.00 seconds into the target video, "
                              "<Picture 1> (from [Shot 1]) is fully referenced.\n" + seg_prompts[0])

        # 素材池张量（按类别）：JSON 池各类别独立加载；图片池空时回退画布 autogrow（全段共用）
        pool_tensors = {"image": {}, "video": {}, "audio": {}}
        for _k, _lbl, _fn in pool_files:
            if _k == "image":
                pool_tensors["image"][_lbl] = _load_input_image(_fn)
            elif _k == "video":
                pool_tensors["video"][_lbl] = _load_input_video(_fn)   # (帧张量, 同源音轨)
            else:
                pool_tensors["audio"][_lbl] = _load_input_audio(_fn)
        refs = {
            "ref_videos": _autogrow_items(参考视频组, "ref_video_"),
            "ref_video_audios": _autogrow_items(参考视频音轨组, "ref_video_audio_"),
            "ref_audios": _autogrow_items(参考音频组, "ref_audio_"),
        }
        if not pool_tensors["image"]:
            _canvas_imgs = _autogrow_items(参考图片组, "ref_image_")
            pool_tensors["image"] = {f"图片{j + 1}": v for j, v in enumerate(_canvas_imgs.values())}
            if pool_tensors["image"]:
                pool_labels = list(pool_tensors["image"].keys())
                seg_label_orders = [[("image", l) for l in pool_labels] for _ in seg_prompts]
        # 状态池某类非空时该类覆盖画布接线（视频/音频与图片同规则），另一类画布接线仍生效
        if pool_tensors["video"]:
            refs["ref_videos"], refs["ref_video_audios"] = {}, {}
        if pool_tensors["audio"]:
            refs["ref_audios"] = {}
        has_refs = any(pool_tensors.values()) or any(refs.values())
        if 首帧图片 is not None and has_refs:
            raise ValueError("首帧图片（i2v，fl2va UNET）与参考素材（r2v，ref2va UNET）不能同时使用")
        if 起始视频 is not None and 首帧图片 is not None:
            raise ValueError("起始视频（序章）与首帧图片（i2v）不能同时使用：两者都定义第 1 段的视觉起点")
        # 生成模式一致性校验：导演台据此互斥首帧/参考图，后端复验，不匹配直接报错（避免跑废）
        _mode = str(生成模式)
        if _mode == "文生视频" and (首帧图片 is not None or has_refs):
            raise ValueError("「文生视频」模式下不能接首帧图片或参考素材：请在导演台切到首帧/多参模式后再接图")
        if _mode == "首帧视频" and has_refs:
            raise ValueError("「首帧视频」模式下不能接参考素材（首帧与多参互斥）：请切到「多参视频」模式用参考图")
        if _mode == "多参视频" and 首帧图片 is not None:
            raise ValueError("「多参视频」模式下不能接首帧图片（多参与首帧互斥）：请切到「首帧视频」模式用首帧")

        if str(宽高比) != "自定义":
            宽度, 高度 = _resolve_canvas(宽高比, 百万像素)
        width, height, seed = int(宽度), int(高度), int(种子)
        length = _snap_seconds(每段时长)
        seg_lengths = [length if s is None else s for s in seg_secs]
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

        report = [f"H3 Seamless Chain：{len(seg_prompts)} 段，链路 {chain}，模式 {_mode}，上下文 {ctx} 帧"]
        if insert_specs:
            _ins_txt = "、".join(f"链位{p}={f}" for p, f in insert_specs)
            report.append(f"插入视频：{len(insert_specs)} 段（{_ins_txt}）——画面+原声进成片，尾帧桥指导下一段")
        _unlink_pos = [str(item_i + 1) for item_i, it in enumerate(exec_items)
                       if it[0] == "prompt" and seg_unlink[it[1]]]
        if _unlink_pos:
            report.append(f"独立镜头：{len(_unlink_pos)} 段与上段断链（无桥接/硬切，段 {'、'.join(_unlink_pos)}）")
        if up_cfg:
            report.append(f"潜空间放大二采：{up_cfg['mode']} · {up_cfg['arch']} {up_cfg['scale']:g}× · "
                          f"强度 {up_cfg['denoise']:g} · 步数 {up_cfg['steps']}——每段采样定稿后立即"
                          "渲染高清，分段视频与成片直接保存二采结果")
        elif _up_err:
            report.append(f"潜空间放大二采：面板已开启但本次跳过——{_up_err}")
        if str(宽高比) != "自定义":
            report.append(f"画布：{宽高比} · {float(百万像素):g}MP → {width}×{height}（官方换算，1MP=1024×1024，32 倍数对齐）")
        else:
            report.append(f"画布：自定义 {width}×{height}")
        _custom_len = [i for i in range(len(seg_lengths)) if seg_lengths[i] != length]
        if _custom_len:
            report.append("每段时长：" + " ".join(f"段{i + 1}={seg_lengths[i] / 24:.1f}s({seg_lengths[i]}帧)" for i in _custom_len)
                          + f"（默认 {length / 24:.1f}s={length}帧）")
        if ds_used:
            report.append("素材来源：导演台状态（JSON）")
        if seg_composed:
            report.append(f"分段处理：{seg_composed}/{len(seg_prompts)} 段含场景/角色/声音提示词（官方三字段组装）")
        if seg_dialogues:
            report.append(f"对白格式：{seg_dialogues} 处「」已转 <d>[中文]")
        if has_refs:
            _k_short = {"image": "图", "video": "视", "audio": "音"}
            _parts = []
            if pool_files:
                _stat = "、".join(f"{_k_short[k]}×{sum(1 for kk, _, _ in pool_files if kk == k)}"
                                  for k in ("image", "video", "audio")
                                  if any(kk == k for kk, _, _ in pool_files))
                _names = "、".join(f"{_k_short[k]}·{lbl}（{pool_file_of.get(lbl, '画布')}）"
                                   for k, lbl, _f in pool_files)
                _parts.append(f"池 {_stat}（{_names}）")
            counts = " ".join(f"{k}={len(v)}" for k, v in refs.items() if v)
            if counts:
                _parts.append(f"画布接线 {counts}")
            if seg_custom_refs:
                _parts.append(f"段级子集 {seg_custom_refs}/{len(seg_prompts)} 段自定义")
            report.append("参考素材：" + " | ".join(_parts))
        if not KEYFRAME_AUDIO_SUPPORTED:
            report.append("注意：当前 ComfyUI 不含 PR #15439（Add Guide 协议），段间引导降级为仅视频锚定，音频不锚定"
                          + ("；r2v 链上引导会与参考素材冲突失效，强烈建议升级 ComfyUI" if has_refs else ""))
        full_bridge = full_bridge_supported()
        if not full_bridge:
            report.append("注意：当前 ComfyUI 的 keyframe 协议仅支持单帧锚定，段间引导已自动降级为单帧桥，"
                          "接缝质量受限；升级 ComfyUI 后无需改参数即自动恢复完整引导帧数")
        # 统一接缝后端：模式一个下拉决定全部衔接参数（refine.resolve_profile）。
        # 预设档（标准/轻量/强力）固化保守参数——锚定加噪/递减锚定一律关闭；
        # 自定义/旧值档透传下方细粒度控件。生成期参数（aug/fade）从 profile 取
        if 接缝处理 not in ("标准", "轻量", "强力", "自定义",
                            "潜空间精修", "smoothstep像素混合", "关闭"):
            # 旧工作流按控件位存的是「接缝混合」时代的值，按位回填
            接缝处理 = "关闭" if 接缝处理 == "关闭" else (
                "smoothstep像素混合" if 接缝处理 == "smoothstep" else "潜空间精修")
        seam = refine.resolve_profile(
            接缝处理,
            anchor_aug=float(锚定加噪),
            fade_ratio=0.0 if 递减锚定 == "关闭" else float(递减锚定),
            strength=float(精修强度), window=int(精修窗口), blend=int(混合帧数),
            adaptive=自适应精修 == "开启")
        if seam["mode"] in ("标准", "轻量", "强力"):
            t = seam["tiers"]
            report.append(f"接缝处理：{seam['mode']}（自适应强度 {t[0]:.2f}/{t[1]:.2f}/{t[2]:.2f}"
                          f" · 窗口 {seam['window']} 帧 · 羽化 {seam['blend']} 帧"
                          f" · 锚定加噪/递减锚定关闭）")
        elif seam["use_refine"]:
            strength_txt = "自适应分档" if seam["tiers"] else f"{seam['fixed']:.2f} 固定"
            report.append(f"接缝处理：{seam['mode']}（细粒度控件：强度{strength_txt}"
                          f" · 窗口 {seam['window']} · 羽化 {seam['blend']} · 加噪 {seam['anchor_aug']:.2f}"
                          f" · 递减 {seam['fade_ratio']:.1f}）")

        aug = min(max(seam["anchor_aug"], 0.0), 0.5)
        if aug > 0.0:
            report.append(f"锚定加噪 {aug:.2f}：桥锚定帧按参考而非逐帧复现注入（视觉 {1.0 - aug:.2f} / 音频 {1.0 - aug * 0.5:.2f} 保真）")
            if aug > 0.25:
                report.append(f"注意：锚定加噪 {aug:.2f} 偏高（>0.25 锚定偏软，段首偏差方差增大，建议 0.15-0.20）")

        tail_anchor_latent = None
        if 尾帧锚定 is not None:
            tail_anchor_latent = video_vae.encode(_center_cover(尾帧锚定[:1], width, height))
            report.append(f"尾帧锚定：身份锚点注入末帧 keyframe（{'视觉保真 ' + format(1.0 - aug, '.2f') if aug > 0 else '硬锚定'}）")

        fade_ratio = seam["fade_ratio"]
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
        if up_cfg and not use_ckpt:
            # 二采产物落项目文件夹（manifest 记录 + seg mp4 覆盖），无存档就没有挂载点
            report.append("潜空间放大二采：需开启自动保存（或审片/自动存档）建立项目存档，本次跳过")
            up_cfg = None
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
            "smart_cut_max": max(0, int(切镜最多丢帧)), "drop_budget": max(0, int(全链丢弃预算)),
            "gate": {"mode": 桥帧门控, "threshold": float(清晰度阈值), "limit": gate_limit},
        }
        # 纯后处理参数只记录不进指纹（改值不触发重跑；报告回看用）
        seam_refine = {"mode": seam["mode"],
                       "tiers": list(seam["tiers"]) if seam["tiers"] else None,
                       "fixed": seam["fixed"], "window": seam["window"],
                       "blend": seam["blend"],
                       "reroll": 接缝重摇, "reroll_th": float(重摇阈值),
                       "reroll_max": int(重摇上限), "anchor_aug": aug}
        if seam["pixel_blend"]:
            report.append("提示：smoothstep像素混合是加权平均（运动残影/糊感来源），建议仅调试对比用；"
                          "正式出片建议「标准」档")
        # 逐段哈希：默认时长且未自定义段级引用的段哈希与旧公式一致（旧存档续跑不受影响）；
        # 改过时长/引用/断链开关的段把对应标记并入哈希 → 自动从该段重做（否则只改勾选不触发重做）
        _hash_refs = bool(pool_files)   # 仅状态池段级引用进哈希；画布回退不进（保旧档兼容）
        seg_hashes = []
        for i, p in enumerate(seg_prompts):
            tag = p
            if seg_lengths[i] != length:
                tag = f"{seg_lengths[i]}|{tag}"
            if _hash_refs and seg_label_orders[i]:
                tag = f"{','.join(lbl for _k, lbl in seg_label_orders[i])}|{tag}"
            if seg_unlink[i]:
                tag = f"unlink|{tag}"
            seg_hashes.append(checkpoint.prompt_hash(tag))

        # 插入段哈希 = 文件指纹（mtime+size）：换文件/同名覆盖上传 → 从该段起重跑
        def _insert_hash(fname):
            try:
                st = os.stat(folder_paths.get_annotated_filepath(fname))
                return checkpoint.prompt_hash(f"insert|{fname}|{st.st_mtime_ns}|{st.st_size}")
            except OSError:
                return checkpoint.prompt_hash(f"insert|{fname}|missing")

        exec_hashes = [seg_hashes[it[1]] if it[0] == "prompt" else _insert_hash(it[1])
                       for it in exec_items]
        prologue_hash = None
        if 起始视频 is not None:
            f0 = 起始视频[0].detach().float().cpu()
            prologue_hash = checkpoint.prompt_hash(
                f"prologue:{int(起始视频.shape[0])}:{float(f0.mean()):.4f}:{float(f0.std()):.4f}")
        root, manifest, done, seeds = None, None, 0, []
        proj_title, proj_created, proj_finals = "", None, []
        proj_inserts = []
        off = 1 if 起始视频 is not None else 0
        if use_ckpt:
            root = checkpoint.ckpt_dir(ckpt_params, 存档目录.strip())
            manifest = checkpoint.load_manifest(root)
            # 项目元数据：title 首次=目录名、created_at 首次=本次、finals/inserts 沿用存档
            proj_title = os.path.basename(root)
            proj_created = time.time()
            if manifest is not None:
                proj_title = str(manifest.get("title") or proj_title)
                proj_created = manifest.get("created_at") or proj_created
                proj_finals = list(manifest.get("finals") or [])
                proj_inserts = [dict(x) for x in (manifest.get("inserts") or [])
                                if isinstance(x, dict)]
            if manifest is not None:
                if manifest.get("schema") != checkpoint.SCHEMA:
                    raise ValueError(f"存档目录格式不认识（{manifest.get('schema')}），请换一个目录名；"
                                     "旧版 v1 存档不兼容本版本，请清空旧目录或换新名字")
                checkpoint.assert_match(manifest["params"], ckpt_params)
                if 起始视频 is None and manifest.get("has_prologue"):
                    off = 1  # 输入已断开仍沿用存档序章（LoadVideo 可 bypass），哈希校验跳过
                elif 起始视频 is not None and not manifest.get("has_prologue"):
                    manifest = checkpoint.truncate(root, manifest, 0)
                    proj_inserts = []
                    report.append("存档续跑：检测到新接入的序章视频，整链重做")
                elif 起始视频 is not None and prologue_hash is not None:
                    stored = list(manifest.get("prompt_hashes", []))
                    if stored and stored[0] != prologue_hash:
                        manifest = checkpoint.truncate(root, manifest, 0)
                        proj_inserts = []
                        report.append("存档续跑：序章视频已更换，整链重做")
                done = checkpoint.contiguous_done(root, int(manifest.get("done", 0)))
                full_hashes = ([prologue_hash] if off else []) + exec_hashes
                hashes = list(manifest.get("prompt_hashes", []))
                # 「重跑起始段」为 1-based 段号（与 tooltip 一致）：N=从第 N 段起重做
                start = min(max(reroll - 1, 0), done) if reroll > 0 else min(
                    checkpoint.reroll_start(hashes, full_hashes, done), done)
                if start < done:
                    manifest = checkpoint.truncate(root, manifest, start)
                    done = start
                    proj_inserts = [x for x in proj_inserts
                                    if int(x.get("slot", -1)) < start]
                    原因 = "手动指定「重跑起始段」" if reroll > 0 else "检测到该段提示词已修改"
                    report.append(f"存档续跑：{原因}，从段 {start + 1} 起重新生成"
                                  + (f"（段 1-{start} 沿用存档）" if start else "（整链重做）"))
                seeds = [int(s) for s in manifest.get("seeds", [])]
                report.append(f"存档续跑：载入已完成 {done}/{len(exec_items) + off} 段，目录 {root}")
                if reroll == 0 and seeds:
                    report.append("控件种子仅在「重跑起始段」> 0 时生效，当前沿用存档种子序列")
        full_hashes = ([prologue_hash] if off else []) + exec_hashes
        # 二采记录沿用：主循环的全量 manifest 覆写必须带上 upscale 键，
        # 否则每次段落盘都会把之前清扫写入的二采记录清掉（truncate 已联动截断）
        proj_upscale = (manifest.get("upscale") if isinstance(manifest, dict) else None) \
            if use_ckpt else None
        if review and autosave:
            report.append(f"审片：分段视频每段落盘 → output/h3_projects/{os.path.basename(root) or '…'}/seg_XXX.mp4，"
                          "成片为运行结束时的已确认部分")

        pbar = comfy.utils.ProgressBar(len(exec_items))
        total = len(exec_items) + off
        prompt_list = (["「序章（上传视频）」"] if off else []) + [
            (f"[插入视频] {it[1]}" if it[0] == "insert" else seg_prompts[it[1]]) for it in exec_items]
        thumbs, videos, seams, bridge_scores = [], [], [], []
        all_frames = []
        all_wav = None
        seg_frames = []
        seg_wavs = []
        trims = []
        guide = None
        prev_tail_frame = None
        prev_tail_wav = None
        prev_tail_clip = None   # 上段尾 24 帧（接缝指标局部基线用）
        prev_lat = None   # 上段 (video_t, audio_t, kept t0, kept t1, audio a0, audio a1)——接缝精修窗口切片用
        prev_hi_tail = None   # 上段高清尾帧 CPU [H,W,3]（二采段首接缝平滑锚；本段未二采时为 None）
        sample_rate = None
        dropped_total = 0        # 全链累计丢弃帧（切镜+门控回退+网格对齐），预算门控用
        seam_metrics_rows = []   # 每缝五维 z-score（与 seams 列表对齐；无缝/指标不可用为 None）

        def _up_hi(g, video_t, audio_t, kind, idx, guide_kf, tail_kf, cur_seed,
                   skip_f, vis_len, blend_span, wav, rate):
            """二采渲染段 g 并落盘高清产物（序章/插入段/生成段统一入口）。

            返回 (ready, tried)：ready=True 表示高清产物已在盘（本次渲染成功或
            记录有效沿用——调用方跳过基础分辨率保存，段视频就是二采结果）；
            ready=False 且 tried=True 表示渲染失败（调用方基础保存强制重编码，
            覆盖可能写坏的 mp4 自愈）；范围外/关闭时 (False, False) 正常基础保存。
            proj_upscale / prev_hi_tail 原地更新（后者供下一段高清接缝平滑）。
            """
            nonlocal proj_upscale, prev_hi_tail
            if not (up_cfg and use_ckpt):
                return False, False
            if not upscale.in_scope(up_cfg, g):
                return False, False
            if g < len(full_hashes) and cur_seed is not None:
                bh = f"{full_hashes[g]}|{cur_seed}"   # 本段当前身份（回放段与 manifest 记录一致）
            else:
                bh = upscale.base_hash(manifest, g) if isinstance(manifest, dict) else ""
            if not upscale.record_stale(manifest, root, up_cfg, g, bh):
                files = checkpoint.upscale_files(g)
                prev_hi_tail = upscale._load_frame_png(os.path.join(root, files["last"]))
                return True, False
            try:
                prev_hi_tail, proj_upscale = upscale.render_segment(
                    模型, clip, video_vae, audio_vae, negative, up_cfg, up_net,
                    root, g, video_t, audio_t, kind, idx,
                    seg_prompts, seg_label_orders, pool_tensors, refs,
                    首帧图片, guide_kf, tail_kf, cur_seed,
                    skip_f, vis_len, blend_span, prev_hi_tail,
                    wav, rate, bh, report, 采样器, 调度器)
                return True, False
            except upscale.UpscaleAbortError:
                raise   # 预检/二采显存致命：报告已 append，终止整链，不降级
            except Exception as e:   # 单段偶发失败只降级该段，整链照常
                oom = "out of memory" in str(e).lower()
                report.append(f"段{g + 1} 二采失败：{type(e).__name__}: {e}"
                              "——本段按基础分辨率保存（基础链产物不受影响）"
                              + ("；显存不够：可把放大倍率降到 1.5× 或关闭重采样只超分（本段"
                                 "会自动补渲染，基础链不重做）" if oom else ""))
                prev_hi_tail = None
                # 释放二采残留（放大 latent / 解码帧），给后续基础采样腾空间；
                # 新方案重采样在基础分辨率跑、放大网络 1-2GB，不触发重型卸载
                try:
                    import comfy.model_management
                    comfy.model_management.soft_empty_cache()
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                return False, True

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
                        "seam_metrics": [None], "seam_refine": seam_refine,
                        "inserts": list(proj_inserts), "upscale": proj_upscale})
                report.append(f"序章：上传视频编码为段 1/{total}（{fc} 帧"
                              + ("，超长仅取前段" if raw_fc > fc else "")
                              + "，经一次 VAE 重编码，按 24fps 处理"
                              + ("，未接音轨按静音处理" if 起始视频音轨 is None else "") + "）")
            pframes = video_vae.decode(pv)
            if len(pframes.shape) == 5:
                pframes = pframes.reshape(-1, pframes.shape[-3], pframes.shape[-2], pframes.shape[-1])
            pwav, sample_rate = _decode_audio(audio_vae, pa)
            _hi_ready, _hi_tried = _up_hi(0, pv, pa, "prologue", None, None, None, 0,
                                          0, pframes.shape[0], 0, pwav, sample_rate)
            if not _hi_ready:
                thumbs.append(checkpoint.save_thumb(root, 0, pframes[0]) if use_ckpt else "")
                videos.append(checkpoint.save_segment_mp4(root, 0, pframes, pwav, sample_rate,
                                                          fresh=prologue_fresh or _hi_tried) if use_ckpt else "")
            else:
                _u_files = checkpoint.upscale_files(0)
                thumbs.append(_u_files["thumb"])
                videos.append(_u_files["mp4"])
            seams.append(None)
            bridge_scores.append(None)
            seam_metrics_rows.append(None)
            all_frames.append(pframes.cpu())
            seg_frames.append(pframes.cpu())
            seg_wavs.append({"waveform": pwav.cpu(), "sample_rate": sample_rate})
            all_wav = pwav.cpu()
            prev_tail_frame = pframes[-1].cpu()
            prev_tail_clip = pframes[-24:].cpu()
            seam_n0 = max(1, int(sample_rate * 0.25))
            prev_tail_wav = pwav.cpu()[..., -seam_n0:]
            guide = _tail_keyframe(pv, pa, ctx, KEYFRAME_AUDIO_SUPPORTED and full_bridge,
                                   full_bridge=full_bridge)
            prev_lat = (pv, pa, 0, pv.shape[2], 0, pa.shape[-1])
            report.append(f"段1/{total}：{prologue_origin} 序章 留{pframes.shape[0]}帧 · 种子 — | guide=无（序章）")

        def _decode_crop(i, video_t, audio_t, skip_f, gi, next_bridge):
            """解码 → 桥帧门控 → 尾切 token 对齐 → 裁剪到保留区（重摇与正常路径共用）。

            gi=全局段号（混排链位+off，报告用）；next_bridge=下一段是否接收本段尾帧桥
            （下段是独立镜头/插入视频/末段时为 False：门控与智能切镜都是为下段桥服务的，
            下段不要桥就不必牺牲本段帧数）。
            返回 (frames, wav, sample_rate, end_t, vis_len, 桥帧总分, 门控报告行)；
            报告行只取最终采用的尝试（重摇的中间尝试整组丢弃）。
            """
            lines = []
            sampled_fc = latent_t_to_frames(video_t.shape[2])
            vis_len = seg_lengths[i]
            frames = video_vae.decode(video_t)
            if len(frames.shape) == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            # 音频归一化只统计保留区（见 _decode_audio）：锚定区音频不计入 std
            wav, sample_rate = _decode_audio(audio_vae, audio_t,
                                             norm_skip_frac=skip_f / sampled_fc if skip_f else 0.0)
            seg_bridge_score = None
            if 桥帧门控 != "关闭" and next_bridge:
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
                        vis_len = seg_lengths[i] - back
                        over = f"，超全链预算（剩余 {max(0, int(全链丢弃预算)) - dropped_total} 帧）WARN" \
                            if back > max(0, int(全链丢弃预算)) - dropped_total else ""
                        lines.append(f"段{gi + 1} 尾帧低质（{tail_score:.1f} < {清晰度阈值:g}），"
                                     f"回退 {back} 帧续拍（回退点 {hit:.1f}）{over}")
            # 智能切镜：在段尾搜索运动低谷（自然停顿点），多余帧丢弃。
            # 运动低谷 = 动作完成/暂停 = 自然切镜点；与桥帧门控互补。
            # 丢帧止损：单段上限（切镜最多丢帧）+ 全链预算（含门控回退与网格对齐，
            # dropped_total 在主循环逐段累计）——超预算只标注不裁剪，防"30秒删7秒"
            if 智能切镜 == "自动" and next_bridge:
                cut_cap = max(0, int(切镜最多丢帧))
                cut = qc.find_cut_point(frames, skip_f, vis_len,
                                        max_trim_frames=cut_cap if cut_cap > 0 else None)
                if cut is not None:
                    cut_f, cut_motion, cut_quality = cut
                    new_vis_len = cut_f - skip_f
                    trimmed = vis_len - new_vis_len
                    budget_left = max(0, int(全链丢弃预算)) - dropped_total
                    if trimmed <= 0:
                        pass
                    elif cut_cap == 0:
                        lines.append(f"段{gi + 1} 智能切镜点位 @帧{cut_f}"
                                     f"（运动 {cut_motion:.4f} 清晰 {cut_quality:.1f}）——"
                                     f"「切镜最多丢帧=0」，仅标注不裁剪")
                    elif trimmed > budget_left:
                        lines.append(f"段{gi + 1} 智能切镜 @帧{cut_f} 需丢 {trimmed} 帧"
                                     f" > 剩余预算 {budget_left} 帧，本次仅标注不裁剪")
                    else:
                        vis_len = new_vis_len
                        lines.append(f"段{gi + 1} 智能切镜：运动低谷 @帧{cut_f}"
                                     f"（运动 {cut_motion:.4f} 清晰 {cut_quality:.1f}），"
                                     f"丢弃尾部 {trimmed} 帧")
            # 尾切对齐 token 网格：kept 末端必须与 guide 锚定末端重合。此前 guide
            # 取采样 latent 原始尾部（含 17k+5 网格填充帧，从未输出），下段续拍点
            # 落在本段输出末尾之后——ctx=56 时每个接缝跳过 15 帧（0.6s）内容。
            # 门控回退时向下对齐（不回吐坏帧），否则向上（不丢内容，每段至多多留几帧）
            end_t = frames_to_latent_t(skip_f + vis_len, up=(seg_lengths[i] - vis_len) == 0)
            vis_len = latent_t_to_frames(end_t) - skip_f
            frames = frames[skip_f:skip_f + vis_len]
            wav_total = wav.shape[-1]
            skip_s = round(wav_total * skip_f / sampled_fc)
            take_s = round(wav_total * vis_len / sampled_fc)
            wav = wav[..., skip_s:skip_s + take_s]
            return frames, wav, sample_rate, end_t, vis_len, seg_bridge_score, lines

        def _next_wants_bridge(item_i):
            """混排序列中 item_i 的下一段是否接收本段尾帧桥：下段不存在/是插入段/
            是独立镜头提示词段 → False（插入段不吃 guide，独立镜头不要桥）"""
            nxt = exec_items[item_i + 1] if item_i + 1 < len(exec_items) else None
            return bool(nxt) and nxt[0] == "prompt" and not seg_unlink[nxt[1]]

        for item_i, item in enumerate(exec_items):
            # ---- 插入视频段：画面+原声进成片，尾帧 latent 桥接指导下一段 ----
            if item[0] == "insert":
                ins_file = item[1]
                g = item_i + off
                ins_replay = use_ckpt and g < done
                if ins_replay:
                    pv, pa = checkpoint.load_segment(root, g)
                    pv = pv.to(video_vae.device)
                    pa = pa.to(audio_vae.device)
                    report.append(f"段{g + 1}/{total}：插入视频「{ins_file}」存档载入 · guide=尾帧桥（指导下段）")
                else:
                    imgs, aud = _load_input_video(ins_file)
                    raw_fc = int(imgs.shape[0])
                    fc = align_frame_count_down(min(raw_fc, length))
                    if fc < 5:
                        raise ValueError(f"插入视频「{ins_file}」至少需要 5 帧（约 0.2 秒 @24fps），请换更长的视频")
                    pv = video_vae.encode(_center_cover(imgs[:fc], width, height))
                    _wav_in = aud.get("waveform") if isinstance(aud, dict) else None
                    if _wav_in is not None and int(_wav_in.shape[-1]) > 0:
                        pa = _encode_audio_latent(audio_vae, aud, audio_tokens_for_frames(fc))
                        _mute = ""
                    else:
                        pa = torch.zeros(1, 32, 2, audio_tokens_for_frames(fc), device=video_vae.device)
                        _mute = "，无音轨按静音处理"
                    if use_ckpt:
                        checkpoint.save_segment(root, g, pv, pa)
                    report.append(f"段{g + 1}/{total}：插入视频「{ins_file}」编码 {fc} 帧"
                                  + ("，超长仅取前段" if raw_fc > fc else "") + _mute
                                  + " · guide=尾帧桥（指导下段）")
                pframes = video_vae.decode(pv)
                if len(pframes.shape) == 5:
                    pframes = pframes.reshape(-1, pframes.shape[-3], pframes.shape[-2], pframes.shape[-1])
                pwav, sample_rate = _decode_audio(audio_vae, pa)
                _hi_ready, _hi_tried = _up_hi(g, pv, pa, "insert", None, None, None, 0,
                                              0, pframes.shape[0], 0, pwav, sample_rate)
                if not _hi_ready:
                    thumbs.append(checkpoint.save_thumb(root, g, pframes[0]) if use_ckpt else "")
                    videos.append(checkpoint.save_segment_mp4(root, g, pframes, pwav, sample_rate,
                                                              fresh=not ins_replay or _hi_tried) if use_ckpt else "")
                else:
                    _u_files = checkpoint.upscale_files(g)
                    thumbs.append(_u_files["thumb"])
                    videos.append(_u_files["mp4"])
                # 插入段与上段硬切（外部素材画面固定，混合/精修会重影）：指标记 None
                seams.append(None)
                bridge_scores.append(None)
                seam_metrics_rows.append(None)
                if g < len(seeds):
                    seeds[g] = 0
                else:
                    seeds.append(0)
                trims.append(0)
                all_frames.append(pframes.cpu())
                seg_frames.append(pframes.cpu())
                seg_wavs.append({"waveform": pwav.cpu(), "sample_rate": sample_rate})
                all_wav = pwav.cpu() if all_wav is None else torch.cat([all_wav, pwav.cpu()], dim=-1)
                prev_tail_frame = pframes[-1].cpu()
                prev_tail_clip = pframes[-24:].cpu()
                _seam_n0 = max(1, int(sample_rate * 0.25))
                prev_tail_wav = pwav.cpu()[..., -_seam_n0:]
                guide = _tail_keyframe(pv, pa, ctx, KEYFRAME_AUDIO_SUPPORTED and full_bridge,
                                       full_bridge=full_bridge)
                prev_lat = (pv, pa, 0, pv.shape[2], 0, pa.shape[-1])
                if not ins_replay:
                    proj_inserts.append({"slot": g, "file": ins_file})
                pbar.update(1)
                if use_ckpt and not ins_replay:
                    done = g + 1
                    checkpoint.save_manifest(root, {
                        "schema": checkpoint.SCHEMA, "done": done, "has_prologue": bool(off),
                        "seeds": list(seeds[:done]), "trims": list(trims[:done]),
                        "prompt_hashes": full_hashes[:done],
                        "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                        "prompts": prompt_list[:done],
                        "seams": seams[:done], "bridge_scores": bridge_scores[:done],
                        "seam_metrics": seam_metrics_rows[:done],
                        "params": ckpt_params, "seam_refine": seam_refine,
                        "inserts": list(proj_inserts),
                        "title": proj_title, "created_at": proj_created,
                        "updated_at": time.time(), "finals": list(proj_finals),
                        "upscale": proj_upscale,
                    })
                if review and not ins_replay and item_i + 1 < len(exec_items):
                    report.append(f"审片：段 {g + 1}（插入视频）已完成并落盘 → seg_{g:03d}.mp4；"
                                  "满意请直接重新运行继续下一段")
                    break
                continue

            i = item[1]   # 提示词段索引（seg_lengths/seg_unlink/seg_label_orders 均按此索引）
            prompt = seg_prompts[i]
            g = item_i + off  # 全局段下标（有序章时序章占 0 号）
            replay = use_ckpt and g < done
            next_wants_bridge = _next_wants_bridge(item_i)
            skip_f = 0 if (item_i == 0 and off == 0) or seg_unlink[i] else ctx
            if replay:
                video_t, audio_t = checkpoint.load_segment(root, g)
                video_t = video_t.to(video_vae.device)
                audio_t = audio_t.to(audio_vae.device)
                dt = 0.0
                cur_seed = seeds[g] if g < len(seeds) else None
                frames, wav, sample_rate, end_t, vis_len, seg_bridge_score, gate_lines = \
                    _decode_crop(i, video_t, audio_t, skip_f, gi=g, next_bridge=next_wants_bridge)
            else:
                # 种子规则：重摇（重跑起始段>0）用控件种子；否则延续断点种子序列的等差，
                # 使审片多轮运行与一次跑完逐帧一致；断点无生成段种子（仅序章/新链）才用控件种子
                if reroll > 0:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                elif len(seeds) > off:
                    cur_seed = (seeds[-1] + g - len(seeds) + 1) % 0xffffffffffffffff
                else:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                seg_len = seg_lengths[i] + (0 if skip_f == 0 else ctx)
                if has_refs:
                    # 段级注入：只把本段勾选的素材压实进 conditioning（未勾选的根本不进本段），
                    # 三类各自独立编号 ref_image_/ref_video_/ref_audio_（<Picture>/<Video>/<Audio>）
                    seg_refs = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
                    _n = {"image": 0, "video": 0, "audio": 0}
                    for _k, _lbl in seg_label_orders[i]:
                        if _lbl not in pool_tensors.get(_k, {}):
                            continue
                        if _k == "image":
                            seg_refs["ref_images"][f"ref_image_{_n['image']}"] = pool_tensors["image"][_lbl]
                        elif _k == "video":
                            _imgs, _aud = pool_tensors["video"][_lbl]
                            seg_refs["ref_videos"][f"ref_video_{_n['video']}"] = _imgs
                            seg_refs["ref_video_audios"][f"ref_video_audio_{_n['video']}"] = _aud
                        else:
                            seg_refs["ref_audios"][f"ref_audio_{_n['audio']}"] = pool_tensors["audio"][_lbl]
                        _n[_k] += 1
                    # 未被状态池接管的画布接线类别：沿用旧行为全段共用，编号接在状态素材之后
                    for j, v in enumerate(refs["ref_videos"].values()):
                        seg_refs["ref_videos"][f"ref_video_{_n['video'] + j}"] = v
                    for j, v in enumerate(refs["ref_video_audios"].values()):
                        seg_refs["ref_video_audios"][f"ref_video_audio_{_n['video'] + j}"] = v
                    for j, v in enumerate(refs["ref_audios"].values()):
                        seg_refs["ref_audios"][f"ref_audio_{_n['audio'] + j}"] = v
                    out = MiniMaxH3ReferenceToVideo.execute(
                        clip=clip, vae=video_vae, audio_vae=audio_vae,
                        prompt=prompt, width=width, height=height, length=seg_len,
                        ref_image_size="match", **seg_refs)
                else:
                    out = MiniMaxH3ImageToVideo.execute(
                        clip=clip, vae=video_vae, prompt=prompt,
                        width=width, height=height, length=seg_len,
                        first_frame=首帧图片 if i == 0 else None)

                cond, latent = out[0], out[1]
                # 独立镜头段：上段桥不注入（guide 屏蔽为 None）；尾帧锚定是用户主动
                # 设定的本段结尾身份锚点，与段间衔接无关，不受断链影响
                eff_guide = None if seg_unlink[i] else guide
                if eff_guide is not None or tail_anchor_latent is not None:
                    cond = cls._apply_guide(
                        cond, eff_guide, latent_t_to_frames(latent["samples"].tensors[0].shape[2]),
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
                    if fade_ratio > 0 and (eff_guide is not None or tail_anchor_latent is not None):
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
                        _decode_crop(i, video_t, audio_t, skip_f, gi=g, next_bridge=next_wants_bridge)
                    d_raw = None
                    if prev_tail_frame is not None and not seg_unlink[i]:
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
            trims.append(seg_lengths[i] - vis_len)
            dropped_total += seg_lengths[i] - vis_len   # 切镜+门控回退+网格对齐合计，预算门控用

            # 段首响度对齐（与分镜链同款）：增益匹配上段尾 RMS（±6dB 钳制 + 1s 渐出），
            # 增益不沿链累积；归一化已排除锚定区，此处兜住内容本身的响度差。
            # 独立镜头段跳过（独立镜头常配独立声音设计）
            if (item_i > 0 or off) and prev_tail_wav is not None and not seg_unlink[i]:
                wav, gain_db = qc.loudness_align_head(wav, prev_tail_wav, rate=sample_rate)
                if gain_db is not None:
                    report.append(f"段{g + 1} 响度对齐：段首 {gain_db:+.1f} dB（1s 渐出）")

            # 接缝后验测量（测而不干预）：上一段最后可见帧 vs 本段首帧
            seam_n = int(sample_rate * 0.25)
            seam_d, seam_db = None, None
            if (item_i > 0 or off) and prev_tail_frame is not None and not seg_unlink[i]:
                seam_d, seam_db = qc.seam_metrics(prev_tail_frame, frames[0],
                                                  prev_tail_wav, wav[..., :seam_n], rate=sample_rate)
                db_txt = f"{seam_db:+.1f} dB" if seam_db is not None else "—"
                flag = " ↑ 建议人工检查" if seam_d > 0.08 or (seam_db is not None and abs(seam_db) > 6.0) else ""
                report.append(f"段{g + 1} 接缝：帧差 {seam_d:.3f} · 响度跳变 {db_txt}{flag}")

            # 接缝处理（生成后、拼接前）：统一由 seam profile 驱动（refine.resolve_profile）。
            # 精修=跨缝窗口联合重去噪（refine.py）：缝两侧出自同一次去噪，根治
            # 段首偏差；只替换本段头部缝后侧帧，上段帧/存档/分段 mp4 均不动。
            # 纯后处理不进断点指纹（改此参数不触发重跑，replay 段结果仍一致）。
            # 独立镜头段不做任何接缝处理（硬切转场）
            # 二采段首平滑跨度：镜像基础链的链接缝处理意图（精修/混合任一开=同跨度
            # smoothstep；全关=硬切）。跨缝精修帧无法在高清路径复用（重采样后是新解码），
            # 桥锚 keyframe + 平滑兜连续性（见 upscale.render_segment）
            hi_blend = seam["blend"] if ((item_i > 0 or off) and prev_tail_frame is not None
                                         and prev_lat is not None and not seg_unlink[i]
                                         and (seam["use_refine"] or seam["pixel_blend"])) else 0
            refine_used = False
            seam_skipped = False   # 窗口不足=完全跳过（像素混合本身是糊感来源，不作兜底）
            if (item_i > 0 or off) and prev_tail_frame is not None and prev_lat is not None \
                    and not seg_unlink[i] and (seam["use_refine"] or seam["pixel_blend"]):
                if seam["use_refine"]:
                    ck0 = video_latent_t(skip_f)
                    cur_lat_ctx = (video_t, audio_t, ck0, end_t,
                                   audio_tokens_for_frames(skip_f),
                                   audio_tokens_for_frames(skip_f + vis_len))
                    win = refine.build_seam_window(prev_lat, cur_lat_ctx, seam["window"])
                    if win is None:
                        seam_skipped = True
                        report.append(f"段{g + 1} 精修窗口不足（保留区不足每侧 {seam['window']} 帧），本次不处理接缝")
                    else:
                        win_v, win_a, vt_p, vt_c, wf = win
                        # 缝前侧（上段尾）全部 video latent 作为 keyframe 注入 cond
                        # （与 _tail_keyframe 同模式：resolved_frame_index=0，锚定窗口
                        # 前 vt_p 帧=上段尾内容）。模型去噪时知道缝后侧应延续缝前侧，
                        # 否则只靠 prompt+refs 自由生成——段2无效(0.170→0.170)、
                        # 段3变差(0.032→0.055)都是因为 cond 缺 keyframe 引导。
                        # 双端锚定：本段侧窗口末端 2 token（5 帧）锚到窗口末帧——
                        # 模型只能改写中间区，不再自由改写无锚端后靠羽化硬接
                        seam_kf_lat = win_v[:, :, :vt_p].clone()
                        tail_kf_lat = win_v[:, :, -2:].clone()
                        eff_strength = refine.adaptive_strength(seam_d, seam["tiers"]) \
                            if seam["tiers"] else seam["fixed"]
                        try:
                            t1 = time.perf_counter()
                            refined_v = refine.refine_seam(
                                模型, negative, prompt, refs if has_refs else None,
                                clip, video_vae, audio_vae, win_v, win_a, wf,
                                width, height, eff_strength,
                                (cur_seed + 1) % 0xffffffffffffffff if cur_seed is not None else 1,
                                步数, CFG, 采样器, 调度器,
                                seam_kf_latent=seam_kf_lat, seam_kf_index=0,
                                tail_kf_latent=tail_kf_lat, tail_kf_index=wf - 5)
                            wframes = video_vae.decode(refined_v)
                            if len(wframes.shape) == 5:
                                wframes = wframes.reshape(-1, wframes.shape[-3],
                                                          wframes.shape[-2], wframes.shape[-1])
                            # 丢弃缝前侧（上段侧），只保留本段替换区
                            wframes = wframes[latent_t_to_frames(vt_p):]
                            replace_n = min(wframes.shape[0], frames.shape[0])
                            feather = min(max(1, seam["blend"]), replace_n)
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
                                worse = " · 缝差略升，已保留精修结果（不再回退像素混合）" if d2 > seam_d else ""
                                report.append(f"段{g + 1} 接缝精修：{seam_d:.3f} → {d2:.3f}"
                                              f" · 强度 {eff_strength:.2f}{'（自适应）' if seam['tiers'] else ''}"
                                              f" · 双端锚定 · 窗口 {wf} 帧 · {time.perf_counter() - t1:.0f}s{worse}")
                            seam_d = d2
                        except Exception as e:
                            report.append(f"段{g + 1} 接缝精修异常（{type(e).__name__}），回退像素平滑")
                if not refine_used and not seam_skipped and seam["use_refine"]:
                    # smoothstep 像素兜底：仅精修异常时使用（加权平均有叠影/糊感）
                    span = min(max(1, seam["blend"]), frames.shape[0])
                    frames = qc.smoothstep_blend_head(frames, prev_tail_frame, span)
                    report.append(f"段{g + 1} 接缝平滑：首帧硬锁锚帧 + smoothstep {span} 帧过渡")
                elif seam["pixel_blend"]:
                    # 纯像素混合档（旧值兼容）：非精修路径的主动选择
                    span = min(max(1, seam["blend"]), frames.shape[0])
                    frames = qc.smoothstep_blend_head(frames, prev_tail_frame, span)
                    report.append(f"段{g + 1} 接缝平滑：首帧硬锁锚帧 + smoothstep {span} 帧过渡")
            if (item_i > 0 or off) and prev_tail_frame is not None and not seg_unlink[i]:
                seams.append([round(seam_d, 4), None if seam_db is None else round(seam_db, 2)])
            else:
                seams.append(None)
            # 接缝五维 z-score（光流/加速度/LPIPS/嵌入/相机）：上段尾 24 帧 +
            # 本段头 48 帧的局部基线评测，处理（精修/混合）后的最终成帧上测。
            # 写入 manifest seam_metrics（tools/ab_report.py 的 A/B 对比数据源）
            z_row = None
            if (item_i > 0 or off) and prev_tail_clip is not None and not seg_unlink[i]:
                try:
                    z_row = metrics.evaluate_local(prev_tail_clip, frames[:48].cpu())
                    z_txt = metrics.fmt_seam_z(z_row)
                    if z_txt and "无可用" not in z_txt:
                        report.append(f"段{g + 1} 接缝基准：{z_txt}（|z|<2 合格）")
                except Exception:
                    z_row = None
            seam_metrics_rows.append(z_row if ((item_i > 0 or off) and not seg_unlink[i]) else None)

            all_frames.append(frames.cpu())
            seg_frames.append(frames.cpu())
            seg_wav = wav.cpu()
            seg_wavs.append({"waveform": seg_wav, "sample_rate": sample_rate})
            all_wav = seg_wav if all_wav is None else torch.cat([all_wav, seg_wav], dim=-1)
            prev_tail_frame = frames[-1].cpu()
            prev_tail_clip = frames[-24:].cpu()
            prev_tail_wav = seg_wav[..., -seam_n:]
            prev_lat = (video_t, audio_t, video_latent_t(skip_f), end_t,
                        audio_tokens_for_frames(skip_f),
                        audio_tokens_for_frames(skip_f + vis_len))
            if use_ckpt:
                # 二采在段视频落盘前接管：成功/记录沿用则段视频即高清结果（同名覆盖），
                # 失败才回退基础分辨率保存（fresh 强制重编码，覆盖可能写坏的 mp4）
                hi_ready, hi_tried = _up_hi(g, video_t, audio_t, "prompt", i,
                                            None if seg_unlink[i] else guide, tail_anchor_latent,
                                            cur_seed, skip_f, frames.shape[0], hi_blend,
                                            wav, sample_rate)
                if not hi_ready:
                    thumbs.append(checkpoint.save_thumb(root, g, frames[0]))
                    videos.append(checkpoint.save_segment_mp4(root, g, frames, wav, sample_rate,
                                                              fresh=not replay or hi_tried))
                else:
                    _u_files = checkpoint.upscale_files(g)
                    thumbs.append(_u_files["thumb"])
                    videos.append(_u_files["mp4"])
            else:
                thumbs.append("")
                videos.append("")

            if guide is not None:
                note = (f"guide=上段尾{ctx}帧" if full_bridge else "guide=单帧桥(旧协议降级)") \
                    + ("+音频" if "audio_latent" in guide else "")
            else:
                note = "guide=无（首段）"
            if seg_unlink[i]:
                note = "guide=无（独立镜头断链）"
            origin = "存档载入" if replay else f"采样{latent_t_to_frames(video_t.shape[2])}帧"
            seed_txt = cur_seed if cur_seed is not None else "—"
            report.append(f"段{g + 1}/{total}：{origin} 裁头{skip_f}帧 留{frames.shape[0]}帧({frames.shape[0] / 24:.1f}s)"
                          f" · 种子 {seed_txt}" + ("" if replay else f" · 采样 {dt:.0f}s") + f" | {note}"
                          + (" · 独立镜头（断链）" if seg_unlink[i] else ""))

            if next_wants_bridge:
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
                    "seam_metrics": seam_metrics_rows[:done],
                    "params": ckpt_params, "seam_refine": seam_refine,
                    "inserts": list(proj_inserts),
                    "title": proj_title, "created_at": proj_created,
                    "updated_at": time.time(), "finals": list(proj_finals),
                    "upscale": proj_upscale,
                })

            pbar.update(1)

            if review and not replay and item_i + 1 < len(exec_items):
                report.append(f"审片：段 {g + 1} 已完成并落盘 → 看项目文件夹里的 seg_{g:03d}.mp4；"
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
                "seam_metrics": seam_metrics_rows[:done],
                "params": ckpt_params, "seam_refine": seam_refine,
                "inserts": list(proj_inserts),
                "title": proj_title, "created_at": proj_created,
                "updated_at": time.time(), "finals": list(proj_finals),
                "upscale": proj_upscale,
            })
            if review:
                if len(seg_frames) == total:
                    report.append("审片：本链已全部完成")
                if reroll > 0:
                    report.append(f"注意：「重跑起始段」={reroll} 已生效，确认无误后请改回 0")
            report.append(f"项目文件夹（视频/提示词/latent 全在此，删项目=删整个文件夹）：{root}")
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll,
                                   "report": "\n".join(report), "updated_at": time.time()})

        images = torch.cat(all_frames, dim=0)
        # 自动保存：完整链（或审片已确认部分）编码成片，直接落项目文件夹。
        # 二采开启且全链高清记录齐时优先流式拼接高清分段（单份产物，成片=二采结果）；
        # 链未完成/记录不齐/尺寸混排/拼接失败回退内存帧编码基础分辨率成片
        if autosave and use_ckpt:
            if not (up_cfg and upscale.try_final(root, up_cfg, report)):
                final_name, enc_err = _autosave_final(root, images, all_wav, sample_rate)
                if final_name:
                    proj_finals.append(os.path.basename(final_name))
                    mf = checkpoint.load_manifest(root) or {}
                    mf["finals"] = list(proj_finals)
                    checkpoint.save_manifest(root, mf)
                    report.append(f"自动保存：分段与成片已就绪 → {root}"
                                  f"（seg_XXX.mp4 逐段，{os.path.basename(final_name)} 完整成片）")
                else:
                    err_detail = f"（{enc_err}）" if enc_err else ""
                    report.append(f"自动保存：成片编码失败{err_detail}，分段 mp4 不受影响")
            # 状态指针重写：带上自动保存/二采成片的报告行（此前的 save_state 在其之前）
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll,
                                   "report": "\n".join(report), "updated_at": time.time()})
        # 链路总结：接缝指标一览（seams[i] = [帧差, 响度dB] 或 None）
        measured = [(g, s[0], s[1]) for g, s in enumerate(seams) if s]
        if measured:
            avg_d = sum(m[1] for m in measured) / len(measured)
            worst = max(measured, key=lambda m: m[1])
            line = f"链路完成：{total} 段 · 接缝平均帧差 {avg_d:.3f} · 最差接缝 段{worst[0] + 1}（{worst[1]:.3f}"
            if worst[2] is not None:
                line += f"，{worst[2]:+.1f} dB"
            report.append(line + "）")
        z_rows = [z for z in seam_metrics_rows if z]
        if z_rows and any(v is not None for z in z_rows for v in
                          (z.get("flow_z"), z.get("lpips_z"), z.get("emb_z"), z.get("cam_z"))):
            fz = [z["flow_z"] for z in z_rows if z.get("flow_z") is not None]
            if fz:
                report.append(f"接缝基准汇总：光流 z 均值 {sum(fz) / len(fz):+.1f} · 最差 {max(fz):+.1f}σ"
                              f"（|z|<2 合格；明细 manifest.seam_metrics）")
        drop_total = sum(t for t in trims if t)
        if drop_total > 0:
            chain_frames = sum(f.shape[0] for f in all_frames)
            share = drop_total / max(1, chain_frames + drop_total) * 100.0
            report.append(f"累计丢弃 {drop_total} 帧（{drop_total / 24:.1f}s，约占计划时长 {share:.1f}%）"
                          f"——含智能切镜/门控回退/网格对齐，逐段明细见 manifest trims")
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
