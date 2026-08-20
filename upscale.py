"""潜空间放大二次采样（二采）独立后端。

上游两个仓库的整合：
- 放大网络：LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler——H3 24 通道 latent
  神经放大（2D 残差骨干 / 纯 3D 卷积，网络与权重加载见 upscale_net.py）。
  时间维 T 绝对不变（17k+5 token 网格硬约束），只放大 H×W，偶数对齐
  （latent 偶数 = 像素 32 倍数，官方画布口径）。
- 二采范式：wjluoxiao/ComfyUI-JZL-MiniMax-H3 的 CondSync 思路——latent 放大后
  用官方 common_ksampler 以 denoise<1 低强度重去噪补回高频细节；条件里的
  keyframe（生成期桥锚）同步放大到目标尺寸，避免按原尺寸重编码。

与主链的关系：完全独立的后处理通道。主循环（生成/桥接/断点/重跑）零改动；
本模块在主循环结束后清扫：读基础段 latent → 神经放大 → 低强度重采样 →
解码裁剪（复刻主循环 skip_f/trims/end_t 口径）→ 接缝平滑 → 落盘 upseg_*。
节点主输出仍是基础链结果；高清产物在项目文件夹（避免高清全帧驻留内存）。
音频不重采样：成片音轨=原轨（零音频回归，拼接不动音频的项目约束）。

两种触发模式（导演台面板）：
- 跟随生成：每次运行后自动对新增/失效段二采（逐段审片时即"生成一段二采一段"）
- 手动选择：勾选任意已完成段（含插入视频段/序章）一键二采

重做规则（manifest.upscale 记录，二采参数不进 ckpt_params 指纹）：
- 二采参数变（hash 变）→ include 范围内段全部重做二采，基础链不动
- 基础链从段 k 重做（truncate）→ upseg ≥ k 的记录与文件自动清除，链完成后补做
- 某段基础 prompt/seed 变（base_hash 变）→ 仅该段二采记录失效
"""

import os
import time

import torch

from . import checkpoint
from . import grid
from . import qc

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
    """记录有效 = hash/base_hash 匹配且关键产物文件在（mp4 成片 + last 接缝锚）。"""
    rec = segs[g] if g < len(segs) else None
    if not (isinstance(rec, dict) and rec.get("done")):
        return False
    if rec.get("hash") != ph or rec.get("base_hash") != bh:
        return False
    files = rec.get("files") or {}
    for key in ("mp4", "last"):
        f = files.get(key)
        if not f or not os.path.isfile(os.path.join(root, f)):
            return False
    return True


def pending_slots(manifest, root, cfg):
    """待二采全局槽位列表（链序）：已完成段中 include 范围内、记录失效或缺失的。"""
    done = int(manifest.get("done") or 0)
    total = int(manifest.get("total") or 0)
    ph = params_hash(cfg)
    segs = _records(manifest)
    out = []
    for g in range(min(done, total)):
        if cfg["include"] is not None and g not in cfg["include"]:
            continue
        if _record_valid(segs, root, g, ph, base_hash(manifest, g)):
            continue
        out.append(g)
    return out


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

def _slot_layout(exec_items, off, seg_unlink, seg_lengths, trims, g, ctx):
    """全局槽位 g -> (kind, idx, item_i, skip_f, vis_len, end_t)。

    kind: "prologue"（序章）/"insert"（插入视频）/"prompt"；vis_len/end_t 仅
    prompt 段有意义（复刻主循环 _decode_crop 口径：trims 来自 manifest，
    end_t 按 trims==0 决定向上/向下对齐 token 网格）。
    """
    if off and g == 0:
        return ("prologue", None, None, 0, None, None)
    item_i = g - off
    item = exec_items[item_i]
    if item[0] != "prompt":
        return ("insert", item[1], item_i, 0, None, None)
    i = item[1]
    skip_f = 0 if (item_i == 0 and off == 0) or seg_unlink[i] else ctx
    trim = trims[g] if g < len(trims) else 0
    vis0 = seg_lengths[i] - trim
    end_t = grid.frames_to_latent_t(skip_f + vis0, up=trim == 0)
    return ("prompt", i, item_i, skip_f, grid.latent_t_to_frames(end_t) - skip_f, end_t)


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


def run_pass(模型, clip, video_vae, audio_vae, negative, root, cfg,
             exec_items, off, seg_prompts, seg_lengths, seg_unlink,
             seg_label_orders, pool_tensors, refs, first_frame, ctx,
             采样器, 调度器, seam_blend, full_bridge, report):
    """主循环后的二采清扫：待做段逐个「放大→重采样→解码→落盘」。

    只读基础链的段 latent 与 manifest 记录，不改任何基础链产物；二采记录写
    manifest.upscale（每段完成即原子落盘，崩溃安全，写回前重读防竞态）。
    全链记录齐且链完成时流式拼接高清成片 final_up_*.mp4。
    ValueError=可读的配置错误（模型缺失/编码失败），由调用方写进报告。
    """
    import comfy.model_management
    import comfy.nested_tensor
    import nodes as comfy_nodes
    from comfy_extras.nodes_minimax_h3 import MiniMaxH3ImageToVideo, MiniMaxH3ReferenceToVideo

    from . import nodes as plugin_nodes
    from . import upscale_net

    mf = checkpoint.load_manifest(root) or {}
    total = int(mf.get("total") or 0)
    done = int(mf.get("done") or 0)
    if total <= 0 or done <= 0:
        report.append("二采：链尚无已完成段，本次跳过")
        return
    trims = [int(t or 0) for t in (mf.get("trims") or [])]
    seeds = list(mf.get("seeds") or [])
    segs = _records(mf)
    while len(segs) < total:
        segs.append(None)
    ph = params_hash(cfg)
    pending = pending_slots(mf, root, cfg)
    mode_txt = (f"{cfg['mode']} · {cfg['model'] or '未选模型'}（{cfg['arch']}）"
                f" · 倍率 {cfg['scale']:g} · 强度 {cfg['denoise']:.2f} · 步数 {cfg['steps']}")
    if pending:
        report.append(f"二采（{mode_txt}）：待做 {len(pending)} 段"
                      f"（段号 {'、'.join(str(g + 1) for g in pending)}）")
    else:
        report.append(f"二采（{mode_txt}）：{done}/{total} 段均为最新（hash {ph}），无待做段")

    if not cfg["model"]:
        raise ValueError("潜空间放大二采：未选择放大模型——请从 HuggingFace "
                         "LBH-123-AI/Minimax_h3_latent_Upscaler 下载权重放入 "
                         "models/latent_upscale_models/，刷新面板后选择")
    dev = comfy.model_management.intermediate_device()
    try:
        net = upscale_net.load_model(cfg["model"], dev, cfg["precision"], cfg["arch"])
    except FileNotFoundError as e:
        raise ValueError(f"潜空间放大模型加载失败：{e}") from None

    has_refs = any(pool_tensors.values()) or any(refs.values())
    pending_set = set(pending)
    chain_end = max(pending_set) + 1 if pending_set else 0
    prev_tail_wav = None    # 上段对齐后尾部 0.25s（响度对齐链状态，逐段推进）
    prev_has_up = False     # 上段是否有二采产物（接缝平滑锚存在性）
    did_any = False
    t_all = time.perf_counter()

    for g in range(chain_end):
        kind, idx, item_i, skip_f, vis_len, _end_t = _slot_layout(
            exec_items, off, seg_unlink, seg_lengths, trims, g, ctx)
        base_v, base_a = checkpoint.load_segment(root, g)
        base_a = base_a.to(audio_vae.device)
        sampled_fc = grid.latent_t_to_frames(base_v.shape[2])

        # 音频链状态（无论是否待做都推进——后续待做段的响度对齐依赖上段尾音）：
        # 复刻主循环口径：归一化排除锚定区 + skip/vis 比例裁剪 + 段首响度对齐
        if kind == "prompt":
            wav, sr = plugin_nodes._decode_audio(
                audio_vae, base_a, norm_skip_frac=skip_f / sampled_fc if skip_f else 0.0)
            wav_total = wav.shape[-1]
            skip_s = round(wav_total * skip_f / sampled_fc)
            take_s = round(wav_total * vis_len / sampled_fc)
            wav = wav[..., skip_s:skip_s + take_s]
            if g > 0 and prev_tail_wav is not None and not seg_unlink[idx]:
                wav, _gain_db = qc.loudness_align_head(wav, prev_tail_wav, rate=sr)
        else:
            wav, sr = plugin_nodes._decode_audio(audio_vae, base_a)
        seam_n = max(1, int(sr * 0.25))
        next_tail_wav = wav.cpu()[..., -seam_n:]

        if g not in pending_set:
            prev_tail_wav = next_tail_wav
            prev_has_up = isinstance(segs[g], dict) and bool(segs[g].get("done"))
            continue

        did_any = True
        t0 = time.perf_counter()
        bw, bh = int(base_v.shape[4]) * 16, int(base_v.shape[3]) * 16
        up_v = upscale_video(base_v.to(dev, torch.float32), net, cfg["scale"], cfg["arch"])
        tw, th = int(up_v.shape[4]) * 16, int(up_v.shape[3]) * 16
        if tw * th > 2_621_440:   # 2.5MP（1024² 二进制口径）
            report.append(f"二采注意：段{g + 1} 目标画布 {tw}×{th} 超过 2.5MP，"
                          "显存/内存压力大，建议降低倍率或缩小基础画布")

        # cond 构造：官方节点按目标分辨率（refs/首帧由官方节点重编码到目标尺寸）
        length = sampled_fc
        if kind == "prompt":
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
            # 插入视频/序章：空提示词轻精修（外部素材段是否二采由面板勾选决定）
            out = MiniMaxH3ImageToVideo.execute(
                clip=clip, vae=video_vae, prompt="", width=tw, height=th, length=length)
        cond, latent = out[0], out[1]

        # 桥锚 CondSync：上段「基础 latent 尾切片」神经放大后注入 keyframe——
        # 只依赖上段基础 latent + scale，与处理顺序/模式无关（单段可独立重做）
        bridged = False
        if skip_f > 0:
            pv, pa = checkpoint.load_segment(root, g - 1)
            pv, pa = pv.to(dev), pa.to(audio_vae.device)
            _, _, _, _, _, end_tokens_prev = _slot_layout(
                exec_items, off, seg_unlink, seg_lengths, trims, g - 1, ctx)
            kf = plugin_nodes._tail_keyframe(
                pv, pa, ctx, plugin_nodes.KEYFRAME_AUDIO_SUPPORTED and full_bridge,
                end_tokens=end_tokens_prev, full_bridge=full_bridge)
            kf["latent"] = upscale_video(kf["latent"].to(dev, torch.float32),
                                         net, cfg["scale"], cfg["arch"])
            cond = plugin_nodes.H3SeamlessChainSampler._apply_guide(cond, kf, length)
            bridged = True

        # 低强度重采样：初始 latent = 放大后的视频 latent + 原音频 latent；
        # 采样输出的音频丢弃，成片音轨 = 原轨（零音频回归）
        seed = (int(seeds[g]) + 1) % 0xffffffffffffffff if g < len(seeds) else 1
        latent["samples"] = comfy.nested_tensor.NestedTensor(
            (up_v.to(dev, torch.float32), base_a.to(dev, torch.float32)))
        restore_rows = plugin_nodes.cond_audio_rows_guard(模型.model.diffusion_model)
        try:
            sampled = comfy_nodes.common_ksampler(
                模型, seed, cfg["steps"], cfg["cfg"], 采样器, 调度器,
                cond, negative, latent, denoise=cfg["denoise"])[0]
        finally:
            restore_rows()
        up_out_v, _discarded_audio = sampled["samples"].unbind()

        # 解码 + 复刻主循环裁剪口径（skip_f 裁头；插入/序章全量）
        frames = video_vae.decode(up_out_v.to(video_vae.device))
        if len(frames.shape) == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        if kind == "prompt":
            frames = frames[skip_f:skip_f + vis_len]

        # 接缝平滑：上段有二采产物时，锚定上段高清尾帧 + smoothstep 窗吸收段首偏差
        blended = False
        if kind == "prompt" and skip_f > 0 and prev_has_up:
            anchor = _load_frame_png(
                os.path.join(root, checkpoint.upseg_paths(root, g - 1)["last"]),
                frames.device)
            if anchor is not None:
                span = min(max(1, int(seam_blend)), frames.shape[0])
                frames = qc.smoothstep_blend_head(frames, anchor, span)
                blended = True

        # 落盘四件套 + manifest 增量记录（每段完成即写盘，崩溃安全）
        files = checkpoint.upseg_paths(root, g)
        checkpoint.save_upsegment(root, g, up_out_v, base_a)
        try:
            from . import media
        except ImportError:
            import media
        if not media.save_av_mp4(os.path.join(root, files["mp4"]), frames, wav, sr):
            raise ValueError(f"段{g + 1} 二采成片编码失败：{media.last_error}")
        _save_png(os.path.join(root, files["thumb"]), frames[0], long_edge=256)
        _save_png(os.path.join(root, files["last"]), frames[-1])
        record = {"hash": ph, "base_hash": base_hash(mf, g), "seed": seed,
                  "done": True, "files": files, "size": [tw, th]}
        segs[g] = record
        fresh = checkpoint.load_manifest(root) or dict(mf)
        up_state = dict(fresh["upscale"]) if isinstance(fresh.get("upscale"), dict) else {}
        segs_f = list(up_state.get("segs") or [])
        while len(segs_f) <= g:
            segs_f.append(None)
        segs_f[g] = record
        up_state["segs"] = segs_f
        up_state["hash"] = ph
        up_state["params"] = {k: cfg[k] for k in _PARAM_KEYS}
        fresh["upscale"] = up_state
        fresh["updated_at"] = time.time()
        checkpoint.save_manifest(root, fresh)
        mf = fresh
        report.append(f"段{g + 1} 二采：{bw}×{bh}→{tw}×{th} · 强度 {cfg['denoise']:.2f}"
                      f" · {time.perf_counter() - t0:.0f}s"
                      + (" · 桥锚CondSync" if bridged else "")
                      + (" · 接缝平滑" if blended else ""))
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        prev_tail_wav = next_tail_wav
        prev_has_up = True

    if did_any:
        report.append(f"二采：本次完成 {len(pending)} 段 · 用时 {time.perf_counter() - t_all:.0f}s")

    # 全链记录齐且链完成 -> 流式拼接高清成片（磁盘级，复用 merge 的拼接器）
    finals = list(mf.get("finals") or [])
    has_up_final = any(str(f).startswith("final_up_") for f in finals)
    if total > 0 and done >= total and (did_any or not has_up_final) \
            and all(_record_valid(segs, root, g, ph, base_hash(mf, g)) for g in range(total)):
        size0 = segs[0].get("size") or [None, None]
        sources = [os.path.join(root, segs[g]["files"]["mp4"]) for g in range(total)]
        out_name = f"final_up_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        try:
            from . import media
        except ImportError:
            import media
        if media.concat_av_mp4(sources, os.path.join(root, out_name),
                               width=size0[0], height=size0[1]):
            fresh = checkpoint.load_manifest(root) or dict(mf)
            fresh.setdefault("finals", []).append(out_name)
            fresh["updated_at"] = time.time()
            checkpoint.save_manifest(root, fresh)
            report.append(f"二采成片：{total} 段高清拼接 → {out_name}"
                          f"（{size0[0]}×{size0[1]}）")
        else:
            report.append(f"二采成片编码失败：{media.last_error}（分段 upseg_*.mp4 不受影响）")
