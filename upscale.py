"""潜空间放大二次采样（二采）——主循环内渲染通道。

上游两个仓库的整合：
- 放大网络：LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler——H3 24 通道 latent
  神经放大（2D 残差骨干 / 纯 3D 卷积，网络与权重加载见 upscale_net.py）。
  时间维 T 绝对不变（17k+5 token 网格硬约束），只放大 H×W，偶数对齐
  （latent 偶数 = 像素 32 倍数，官方画布口径）。
- 二采范式：wjluoxiao/ComfyUI-JZL-MiniMax-H3 的 CondSync 思路——latent 放大后
  用官方 common_ksampler 以 denoise<1 低强度重去噪补回高频细节；条件里的
  keyframe（生成期桥锚）同步放大到目标尺寸，避免按原尺寸重编码。

与主链的关系：每段采样完成后（重摇定稿、latent 存档之后、分段 mp4 落盘之前）
立即「神经放大 → 低强度重采样 → 解码」——分段视频 seg_NNN.mp4 与成片直接
保存二采后的高清结果，不再产出 upseg_* 双份副本（旧版独立后处理通道已废弃）。
段内机制（重摇/门控/智能切镜/接缝精修/指标）仍跑在基础分辨率帧上，语义零漂移；
二采只接管「落盘的帧」。音频不重采样：分段与成片音轨=原轨（零音频回归）。

两种触发模式（导演台面板）：
- 跟随生成：每次运行对待做段（新增/失效）二采，逐段审片时即"生成一段二采一段"
- 手动选择：勾选任意段（含插入视频段/序章）随下次运行二采

重做规则（manifest.upscale 记录，二采参数不进 ckpt_params 指纹）：
- 二采参数变（hash 变）→ include 范围内段全部重渲染高清，基础 latent 不动
- 基础链从段 k 重做（truncate）→ ≥ k 的记录与锚文件自动清除，重跑时补渲染
- 某段基础 prompt/seed 变（base_hash 变）→ 仅该段高清记录失效
- 全链记录齐且尺寸一致 → 成片改为流式拼接高清分段（单份产物）
"""

import os
import time

import torch

from . import checkpoint
from . import grid

MODES = ("跟随生成", "手动选择")
PRECISIONS = ("fp32", "fp16", "bf16")
_PARAM_KEYS = ("model", "arch", "scale", "denoise", "steps", "cfg", "precision")


# ---- 状态解析与重做判定（纯逻辑，无 ComfyUI 依赖，可单测） ----

def parse_state(ds):
    """导演台状态 ds -> 归一化二采配置；关闭/缺失返回 None。

    mode: 关闭（None）/ 跟随生成 / 手动选择；include 仅手动模式有值
    （全局槽位 0-based 列表，空列表=本次无可做段）；跟随生成 include=None=全部段。
    """
    if not isinstance(ds, dict):
        return None
    up = ds.get("upscale")
    if not isinstance(up, dict) or up.get("on") is False:
        return None
    mode = str(up.get("mode") or "").strip()
    if mode in ("", "关闭", "off"):
        return None
    if mode not in MODES:
        mode = "跟随生成"

    def _num(key, default, lo, hi):
        try:
            v = float(up.get(key))
        except (TypeError, ValueError):
            v = default
        if v != v:            # NaN 兜底
            v = default
        return min(max(v, lo), hi)

    precision = str(up.get("precision") or "fp32")
    if precision not in PRECISIONS:
        precision = "fp32"
    include = None
    if mode == "手动选择":
        vals = set()
        if isinstance(up.get("include"), list):
            for x in up["include"]:
                try:
                    vals.add(int(x))
                except (TypeError, ValueError):
                    continue
        include = sorted(vals)
    return {
        "mode": mode,
        "model": str(up.get("model") or "").strip(),
        "arch": "3D" if str(up.get("arch") or "").strip().upper() == "3D" else "2D",
        "scale": _num("scale", 2.0, 1.0, 4.0),
        "denoise": _num("denoise", 0.45, 0.05, 1.0),
        "steps": int(_num("steps", 15, 1, 100)),
        "cfg": _num("cfg", 1.0, 0.0, 100.0),
        "precision": precision,
        "include": include,
    }


def params_hash(cfg):
    """二采参数 -> 8 位指纹（mode/include 不进：不影响单段输出）。"""
    return checkpoint.fingerprint({k: cfg[k] for k in _PARAM_KEYS})


def base_hash(manifest, g):
    """基础段身份指纹：prompt_hashes[g] + seeds[g]（基础链改词/换种子即失效）。"""
    hashes = list(manifest.get("prompt_hashes") or [])
    seeds = list(manifest.get("seeds") or [])
    return f"{hashes[g] if g < len(hashes) else ''}|{seeds[g] if g < len(seeds) else ''}"


def _records(manifest):
    up = manifest.get("upscale")
    segs = up.get("segs") if isinstance(up, dict) else None
    return list(segs) if isinstance(segs, list) else []


def _record_valid(segs, root, g, ph, bh):
    """记录有效 = hash/base_hash 匹配且产物文件齐全（高清分段 mp4 + 缩略图 + 尾帧锚）。"""
    rec = segs[g] if g < len(segs) else None
    if not (isinstance(rec, dict) and rec.get("done")):
        return False
    if rec.get("hash") != ph or rec.get("base_hash") != bh:
        return False
    files = rec.get("files") or checkpoint.upscale_files(g)
    for key in ("mp4", "thumb", "last"):
        f = files.get(key)
        if not f or not os.path.isfile(os.path.join(root, f)):
            return False
    return True


def record_stale(manifest, root, cfg, g, bh=None):
    """段 g 的高清渲染是否待做（供主循环逐段判定；manifest 可为 None=全待做）。"""
    if cfg is None:
        return False
    if not in_scope(cfg, g):
        return False
    if manifest is None:
        return True
    ph = params_hash(cfg)
    if bh is None:
        bh = base_hash(manifest, g)
    return not _record_valid(_records(manifest), root, g, ph, bh)


def in_scope(cfg, g):
    """段 g 是否在二采范围内（跟随生成=全部段；手动选择=勾选段）。"""
    if cfg is None:
        return False
    if cfg["include"] is None:
        return True
    return g in cfg["include"]


# ---- latent 放大数学（纯 torch，可单测） ----

def target_hw(h, w, scale):
    """latent 目标尺寸：round(H×scale) 向上取偶（latent 偶数 = 像素 32 对齐）。"""
    h2 = max(2, int(round(h * float(scale))))
    w2 = max(2, int(round(w * float(scale))))
    return h2 + h2 % 2, w2 + w2 % 2


def upscale_video(video_t, net, scale, arch="2D"):
    """[B,24,T,H,W] latent -> 神经放大（T 不变，H/W×scale 偶数对齐），float32。

    归一化口径与上游训练一致：放大前 (x-μ)/σ，放大后反变换；网络精度由
    load_model 的 precision 决定，输入输出统一 float32。scale 使目标等于
    原尺寸时原样返回克隆（等价纯二采不放大）。
    """
    h2, w2 = target_hw(video_t.shape[-2], video_t.shape[-1], scale)
    if (h2, w2) == (video_t.shape[-2], video_t.shape[-1]):
        return video_t.detach().to(torch.float32).clone()
    x = video_t.detach().to(torch.float32)
    nd = next(net.parameters()).dtype
    mean, std = _norm_tensors(x.device, nd)
    xn = (x.to(nd) - mean) / std
    if arch == "3D":
        y = net(xn, scale=float(scale), target_size=(x.shape[2], h2, w2))
    else:
        y = net(xn, scale=float(scale), target_hw=(h2, w2))
    mean32, std32 = _norm_tensors(x.device, torch.float32)
    return y.to(torch.float32) * std32 + mean32


def _norm_tensors(device, dtype):
    from . import upscale_net
    return upscale_net._make_norm_tensors(device, dtype)


def resize_latent_bilinear(z, h, w):
    """latent 空间双线性缩放（CondSync 兜底路径，JZL 同款）：T/C 不变。

    5D -> 4D reshape -> bilinear -> 还原；目标等于原尺寸时原样返回。
    """
    b, c, t, hh, ww = z.shape
    if (h, w) == (hh, ww):
        return z
    flat = z.reshape(b * t, c, hh, ww)
    out = torch.nn.functional.interpolate(flat, size=(h, w), mode="bilinear",
                                          align_corners=False)
    return out.reshape(b, c, t, h, w)


# ---- 运行期（需 ComfyUI 环境，单测不触达） ----

def load_net(cfg):
    """按配置加载放大网络（upscale_net 按 名称+架构+设备+精度 缓存）。

    未选模型 / 权重文件缺失 / 架构不匹配 -> ValueError（带可操作的指引），
    调用方（nodes.py）在主循环前调用一次：当场报错降级为基础分辨率整链运行，
    不让每段反复撞同一个错。
    """
    import comfy.model_management

    from . import upscale_net

    if not cfg["model"]:
        raise ValueError("未选择放大模型——请从 HuggingFace "
                         "LBH-123-AI/Minimax_h3_latent_Upscaler 下载权重放入 "
                         "models/latent_upscale_models/，刷新导演台后在二采面板选择")
    dev = comfy.model_management.intermediate_device()
    try:
        return upscale_net.load_model(cfg["model"], dev, cfg["precision"], cfg["arch"])
    except FileNotFoundError as e:
        raise ValueError(f"放大模型加载失败：{e}") from None


def _build_seg_refs(i, seg_label_orders, pool_tensors, refs):
    """段级参考素材压实（与主循环同式：状态池勾选 + 未接管画布接线类别接编号）。"""
    seg_refs = {"ref_images": {}, "ref_videos": {}, "ref_video_audios": {}, "ref_audios": {}}
    n = {"image": 0, "video": 0, "audio": 0}
    for k, lbl in seg_label_orders[i]:
        if lbl not in pool_tensors.get(k, {}):
            continue
        if k == "image":
            seg_refs["ref_images"][f"ref_image_{n['image']}"] = pool_tensors["image"][lbl]
        elif k == "video":
            imgs, aud = pool_tensors["video"][lbl]
            seg_refs["ref_videos"][f"ref_video_{n['video']}"] = imgs
            seg_refs["ref_video_audios"][f"ref_video_audio_{n['video']}"] = aud
        else:
            seg_refs["ref_audios"][f"ref_audio_{n['audio']}"] = pool_tensors["audio"][lbl]
        n[k] += 1
    for j, v in enumerate(refs["ref_videos"].values()):
        seg_refs["ref_videos"][f"ref_video_{n['video'] + j}"] = v
    for j, v in enumerate(refs["ref_video_audios"].values()):
        seg_refs["ref_video_audios"][f"ref_video_audio_{n['video'] + j}"] = v
    for j, v in enumerate(refs["ref_audios"].values()):
        seg_refs["ref_audios"][f"ref_audio_{n['audio'] + j}"] = v
    return seg_refs


def render_latent(模型, clip, video_vae, audio_vae, negative, cfg, net,
                  video_t, audio_t, kind, idx, seg_prompts, seg_label_orders,
                  pool_tensors, refs, first_frame, guide, tail_kf_latent, cur_seed,
                  采样器, 调度器):
    """基础段 AV latent -> 高清视频 latent（放大 + 低强度重采样）。

    kind: "prompt"（提示词段，cond 带本段提示词/参考素材/首帧）
          / "insert"|"prologue"（外部素材段，空提示词轻精修）。
    guide: 本段生成时用的上段尾帧桥（None=首段/断链/插入段）——视频 latent
    神经放大后注入重采样 cond（CondSync：锚住段首与上段高清尾的连续性）。
    tail_kf_latent: 尾帧身份锚定的基础 latent（与主循环同语义，同步放大注入）。
    返回 (up_out_v 高清视频 latent, tw, th, 二采种子, 是否桥锚)。
    """
    import comfy.model_management
    import comfy.nested_tensor
    import nodes as comfy_nodes
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

    from . import nodes as plugin_nodes

    dev = next(net.parameters()).device
    if dev.type == 'cpu':
        # 上段二采异常后放大网络留在 CPU，装回 GPU
        dev = comfy.model_management.intermediate_device()
        net.to(dev)
    up_v = upscale_video(video_t, net, cfg["scale"], cfg["arch"])
    tw, th = int(up_v.shape[4]) * 16, int(up_v.shape[3]) * 16
    length = grid.latent_t_to_frames(video_t.shape[2])

    # cond 构造：官方节点按目标分辨率（refs/首帧由官方节点重编码到目标尺寸）
    if kind == "prompt":
        has_refs = any(pool_tensors.values()) or any(refs.values())
        if has_refs:
            out = MiniMaxH3ReferenceToVideo.execute(
                clip=clip, vae=video_vae, audio_vae=audio_vae,
                prompt=seg_prompts[idx], width=tw, height=th, length=length,
                ref_image_size="match",
                **_build_seg_refs(idx, seg_label_orders, pool_tensors, refs))
        else:
            out = MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=video_vae, prompt=seg_prompts[idx],
                width=tw, height=th, length=length,
                first_frame=first_frame if idx == 0 else None)
    else:
        out = MiniMaxH3ImageToVideo.execute(
            clip=clip, vae=video_vae, prompt="", width=tw, height=th, length=length)
    cond, latent = out[0], out[1]

    # 桥锚 CondSync：上段尾帧桥的 video latent 同步神经放大后注入 keyframe——
    # 只依赖上段基础 latent + scale，与处理顺序/模式无关（单段可独立重做）；
    # 尾帧身份锚定同式放大注入（首尾双锚隧道与主循环同构）
    bridged = False
    if kind == "prompt" and (guide is not None or tail_kf_latent is not None):
        up_guide = None
        if guide is not None:
            up_guide = dict(guide)
            up_guide["latent"] = upscale_video(guide["latent"].to(dev, torch.float32),
                                               net, cfg["scale"], cfg["arch"])
            bridged = True
        up_tail = None
        if tail_kf_latent is not None:
            up_tail = upscale_video(tail_kf_latent.to(dev, torch.float32),
                                    net, cfg["scale"], cfg["arch"])
        cond = plugin_nodes.H3SeamlessChainSampler._apply_guide(
            cond, up_guide, length, tail_kf_latent=up_tail)

    # 放大网络不再需要：卸至 CPU 释放显存给重采样（重采样结束后再装回，
    # 供下一段二采的放大步骤）。32GB 卡上这一步能腾出 ~1-2GB，恰好够用
    net.cpu()
    torch.cuda.empty_cache()

    # 低强度重采样：初始 latent = 放大后的视频 latent + 原音频 latent；
    # 采样输出的音频丢弃，分段/成片音轨 = 原轨（零音频回归）
    seed = (int(cur_seed) + 1) % 0xffffffffffffffff if cur_seed is not None else 1
    latent["samples"] = comfy.nested_tensor.NestedTensor(
        (up_v.to(dev, torch.float32), audio_t.to(dev, torch.float32)))
    restore_rows = plugin_nodes.cond_audio_rows_guard(模型.model.diffusion_model)
    try:
        sampled = comfy_nodes.common_ksampler(
            模型, seed, cfg["steps"], cfg["cfg"], 采样器, 调度器,
            cond, negative, latent, denoise=cfg["denoise"])[0]
    finally:
        restore_rows()

    # 先释放重采样中间张量，再装回放大网络——顺序反了会因重采样缓存还在而 OOM。
    # try/finally 确保异常时 net 也能装回（下段二采的放大步骤需要 GPU 上的 net）
    up_out_v, _discarded_audio = sampled["samples"].unbind()
    del cond, latent, sampled
    torch.cuda.empty_cache()
    return up_out_v, tw, th, seed, bridged


def render_segment(模型, clip, video_vae, audio_vae, negative, cfg, net,
                   root, g, video_t, audio_t, kind, idx,
                   seg_prompts, seg_label_orders, pool_tensors, refs,
                   first_frame, guide, tail_kf_latent, cur_seed,
                   skip_f, vis_len, blend_span, prev_hi_tail,
                   wav, sample_rate, bh, report, 采样器, 调度器):
    """基础段 AV latent -> 高清分段直接落盘（放大→重采样→解码→裁剪→接缝平滑）。

    主循环逐段调用（采样定稿/回放载入之后、基础段落盘之前）：分段视频与
    缩略图沿用基础段同名（单份产物——seg_NNN.mp4 即高清结果），另存尾帧锚
    uplast_NNN.png（下一段高清接缝平滑用）并原子写 manifest.upscale 记录。
    裁剪口径与主循环 _decode_crop 完全一致：skip_f/vis_len 由调用方按基础
    分辨率帧的门控/切镜决策传入（机制零漂移，二采只接管落盘的帧）；
    音频沿用基础段原轨（零音频回归）。blend_span>0 且锚帧尺寸匹配时对段首
    做 smoothstep 平滑（高清路径无跨缝精修，靠桥锚 keyframe+平滑兜连续性）。
    返回 (高清尾帧 CPU tensor, 最新 upscale 存档状态)；异常向上抛，由调用方
    降级为基础分辨率保存（基础链产物不受影响）。
    """
    from . import qc
    import comfy.model_management

    t0 = time.perf_counter()
    purge_legacy(root, g)

    # 释放基础链采样残留（KV cache / conditioning / 推理缓存），给二采腾显存——
    # 基础采样刚结束显存占用 ~28GB，二采放大需额外 ~2GB，不清理必 OOM
    comfy.model_management.soft_empty_cache()
    torch.cuda.empty_cache()

    up_v, tw, th, up_seed, bridged = render_latent(
        模型, clip, video_vae, audio_vae, negative, cfg, net,
        video_t, audio_t, kind, idx, seg_prompts, seg_label_orders,
        pool_tensors, refs, first_frame, guide, tail_kf_latent, cur_seed,
        采样器, 调度器)
    frames = video_vae.decode(up_v)
    del up_v
    torch.cuda.empty_cache()
    if len(frames.shape) == 5:
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    frames = frames[skip_f:skip_f + vis_len]
    blended = False
    if blend_span and prev_hi_tail is not None \
            and tuple(prev_hi_tail.shape[-2:]) == tuple(frames.shape[-2:]):
        frames = qc.smoothstep_blend_head(frames, prev_hi_tail,
                                          min(int(blend_span), frames.shape[0]))
        blended = True
    if not checkpoint.save_segment_mp4(root, g, frames, wav, sample_rate, fresh=True):
        raise RuntimeError("高清分段编码失败（save_av_mp4 返回失败，详见 ComfyUI 控制台输出）")
    checkpoint.save_thumb(root, g, frames[0])
    files = checkpoint.upscale_files(g)
    _save_png(os.path.join(root, files["last"]), frames[-1])
    up_state = write_record(root, g, cfg, up_seed, (tw, th), bh)
    report.append(f"段{g + 1} 二采：{tw}×{th} · {cfg['arch']} {cfg['scale']:g}× · "
                  f"强度 {cfg['denoise']:g} · {cfg['steps']}步 · {time.perf_counter() - t0:.0f}s"
                  + (" · 桥锚" if bridged else "") + (" · 接缝平滑" if blended else ""))
    # 收尾：释放二采残留（放大 latent / 解码帧 / 重采样缓存），给下段基础采样腾显存
    torch.cuda.empty_cache()
    return frames[-1].detach().float().cpu(), up_state


def write_record(root, g, cfg, seed, size, bh):
    """段 g 高清渲染记录原子写盘（重读 manifest 防竞态覆盖并发进度）。

    bh=该段基础身份指纹（调用方用本地 full_hashes/seeds 现算，不依赖磁盘
    manifest 的写入时机）；返回最新 upscale dict（调用方回填 proj_upscale，
    后续主循环的 manifest 快照写盘才不会把记录冲掉）。
    """
    ph = params_hash(cfg)
    rec = {"hash": ph, "base_hash": bh, "seed": seed, "done": True,
           "files": checkpoint.upscale_files(g), "size": list(size)}
    fresh = checkpoint.load_manifest(root) or {}
    up_state = dict(fresh["upscale"]) if isinstance(fresh.get("upscale"), dict) else {}
    segs = list(up_state.get("segs") or [])
    while len(segs) <= g:
        segs.append(None)
    segs[g] = rec
    up_state["segs"] = segs
    up_state["hash"] = ph
    up_state["params"] = {k: cfg[k] for k in _PARAM_KEYS}
    fresh["upscale"] = up_state
    fresh["updated_at"] = time.time()
    checkpoint.save_manifest(root, fresh)
    return up_state


def purge_legacy(root, g):
    """清掉旧版二采独立产物（upseg_*/upthumb_*，防新旧文件混淆）。"""
    for f in checkpoint.upscale_legacy_files(g):
        try:
            os.remove(os.path.join(root, f))
        except OSError:
            pass


def _save_png(path, frame, long_edge=None):
    """单帧落盘 PNG（失败返回 False，不阻断主流程）。"""
    try:
        from PIL import Image
        arr = (frame.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        img = Image.fromarray(arr)
        if long_edge:
            w, h = img.size
            s = float(long_edge) / max(w, h)
            if s < 1.0:
                img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
        img.save(path)
        return True
    except Exception:
        return False


def _load_frame_png(path, device=None):
    """读回 PNG 锚帧 -> [H,W,3] float 0-1 tensor（失败返回 None）。"""
    try:
        from PIL import Image
        import numpy as np
        img = Image.open(path).convert("RGB")
        arr = np.asarray(img).astype("float32") / 255.0
        t = torch.from_numpy(arr)
        return t.to(device) if device is not None else t
    except Exception:
        return None


def try_final(root, cfg, report):
    """全链段高清记录齐且尺寸一致时，流式拼接 seg_*.mp4 -> final_时间戳.mp4。

    返回 True 表示已产出高清成片（调用方跳过基础分辨率成片编码，单份产物）；
    链未完成 / 记录不齐 / 尺寸不一致（混排手动二采）/ 编码失败均返回 False
    （回退主循环内存帧编码基础成片，分段高清不受影响）。
    """
    mf = checkpoint.load_manifest(root) or {}
    total = int(mf.get("total") or 0)
    done = int(mf.get("done") or 0)
    if total <= 0 or done < total:
        return False
    ph = params_hash(cfg)
    segs = _records(mf)
    if len(segs) < total:
        report.append("二采成片：分段高清记录不全（手动模式未全选），成片按基础分辨率编码")
        return False
    for g in range(total):
        if not _record_valid(segs, root, g, ph, base_hash(mf, g)):
            report.append("二采成片：部分段无有效高清记录（手动模式未全选或记录失效），"
                          "成片按基础分辨率编码")
            return False
    size0 = segs[0].get("size") or [None, None]
    if any((segs[g].get("size") or [None, None]) != size0 for g in range(total)):
        report.append("二采成片：分段高清尺寸不一致（手动模式混排），成片按基础分辨率编码")
        return False
    sources = [os.path.join(root, segs[g]["files"]["mp4"]) for g in range(total)]
    out_name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
    try:
        from . import media
    except ImportError:
        import media
    if media.concat_av_mp4(sources, os.path.join(root, out_name),
                           width=size0[0], height=size0[1]):
        fresh = checkpoint.load_manifest(root) or dict(mf)
        fresh.setdefault("finals", []).append(out_name)
        up_state = dict(fresh.get("upscale") or {})
        up_state.setdefault("finals", []).append(out_name)
        fresh["upscale"] = up_state
        fresh["updated_at"] = time.time()
        checkpoint.save_manifest(root, fresh)
        report.append(f"二采成片：{total} 段高清流式拼接 → {out_name}"
                      f"（{size0[0]}×{size0[1]}，音轨沿用原声）")
        return True
    report.append(f"二采成片编码失败（{media.last_error}）——回退编码基础分辨率成片，"
                  "分段高清不受影响")
    return False
