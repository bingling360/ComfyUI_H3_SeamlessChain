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

    # 参数 schema v2：steps 语义从「调度总步数 N」改为「尾段精化步数 n」（denoise 语义
    # 同步明确为尾段起始 σ）。v1 旧 JSON 字段级迁移：显式存过的 steps 按旧口径
    # int(N×强度) 折算保行为（旧默认 15×0.45=6 恰为新默认）；没存过的字段直接
    # 落 v2 新默认（从未配置=用最新推荐值，而不是旧默认）。
    try:
        schema = int(up.get("schema"))
    except (TypeError, ValueError):
        schema = 1

    precision = str(up.get("precision") or "fp16")
    if precision not in PRECISIONS:
        precision = "fp16"
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
    denoise = _num("denoise", 0.35, 0.05, 1.0)
    if schema < 2:
        has_steps = up.get("steps") not in (None, "")
        steps = (min(max(int(_num("steps", 15, 1, 100) * denoise), 1), 100)
                 if has_steps else 6)
    else:
        steps = int(_num("steps", 6, 1, 100))
    return {
        "mode": mode,
        "model": str(up.get("model") or "").strip(),
        "arch": "3D" if str(up.get("arch") or "").strip().upper() == "3D" else "2D",
        "scale": _num("scale", 2.0, 1.0, 4.0),
        "denoise": denoise,
        "steps": steps,
        "cfg": _num("cfg", 1.0, 0.0, 100.0),
        "precision": precision,
        "time_bias": _num("time_bias", 0.0, 0.0, 0.2),
        "mix": _num("mix", 0.0, 0.0, 1.0),
        "include": include,
    }


def tail_refine_args(sigma_start, tail_steps):
    """(尾段起始σ, 精化步数 n) -> common_ksampler 的 (steps=N, denoise)。

    common_ksampler 对 denoise<1 的内部行为 = N 步调度序列取尾 int(N·denoise) 步，
    flow-matching 下初始加噪 (1-σ₀)x0+σ₀ε 落在截断后的 σ₀——本就是 sigma 尾段精化；
    本函数把「σ 起点 × 精化步数」两个直接语义换算回该公开 API：
    - N=round(n/σ)：simple 线性调度下截断点 σ₀≈n/N 即请求值（非 simple/带 shift
      调度为网格近似，报告行打印实际生效值兜底）；
    - N≥n+1 且 denoise=(n+0.5)/N<1：部分精化档永不撞全量重采（denoise=1 会完全
      丢弃放大后的 latent），σ≥0.95 视为明确请求全量重采才返回 denoise=1.0；
    - +0.5 抵御 int(N·denoise) 的浮点截断少 1 步。
    """
    n = min(max(int(tail_steps), 1), 100)
    s = min(max(float(sigma_start), 0.05), 1.0)
    if s >= 0.95:
        return n, 1.0
    big_n = min(max(int(round(n / s)), n + 1), 100)
    if big_n == 100:
        n = min(n, max(1, int(big_n * s)))   # σ 过小受 100 格上限，反向钳步数
    return big_n, min(1.0, (n + 0.5) / big_n)


def latent_hf_energy(video_t, max_frames=12):
    """latent 高频能量（细节代理指标，纯 torch 可单测）。

    逐帧空间 3×3 均值池化残差的 RMS（≈空域拉普拉斯能量），均匀抽样 ≤max_frames
    帧、CPU 计算（避开二采峰值期的显存）。用于「细节增益」对比：放大 latent
    （纯神经放大、无精化）vs 精化后 latent——增益为正说明尾段精化补回了高频。
    输入 [B,C,T,H,W]；非 5D 或空间 <3 返回 0.0。
    """
    if video_t is None or video_t.dim() != 5 \
            or video_t.shape[-2] < 3 or video_t.shape[-1] < 3:
        return 0.0
    t = int(video_t.shape[2])
    step = max(1, t // max_frames)
    idx = torch.arange(0, t, step)[:max_frames]
    x = video_t.detach().to("cpu", torch.float32)[0, :, idx]     # [C,k,H,W]
    smooth = torch.nn.functional.avg_pool2d(x, 3, stride=1, padding=1)
    return float((x - smooth).pow(2).mean().sqrt())


def hf_gain_ratio(hf_before, hf_after):
    """细节增益百分比（小数）：after 相对 before 的高频能量增幅；before≈0 视为 0。"""
    if hf_before < 1e-8:
        return 0.0
    return (hf_after - hf_before) / hf_before


def freq_mix_latents(base_v, refined_v, ratio, chunk=8):
    """频域细节混合（拉普拉斯分层，纯 torch、CPU 分帧块可单测）。

    out = 精化输出 + ratio·(低频(放大 latent) − 低频(精化输出))，低频 = 逐帧
    3×3 均值池化、replicate 边界填充（与细节增益度量同核；填充不用零填充
    ——零填充会把边界低频拉向 0，r=1 时角点偏差大、画面边缘出晕影，
    replicate 下常量场严格守恒）：
    - ratio=0 原样返回精化输出（关——现行为不变，且不克隆零开销）；
    - ratio=1 = 低频全走放大 latent + 高频全走精化输出：结构/段间接缝锚在
      与基础段同构的放大 latent 上（零漂移），只保留精化补回的细节增益——
      天然对冲「精化带花/内容漂移」，多段链「段间一致性优先」的取向。
    ratio 即「把低频换回放大 latent 的程度」，中间值线性过渡。
    输入 [B,C,T,H,W]（池化逐帧独立，按 T 分块无边界效应）；CPU float32
    计算（避开二采峰值期显存，同 latent_hf_energy 口径）。退化输入
    （None / 形状不一致 / 空间 <3）原样返回 refined_v——混合不可定义时
    保交付产物（精化输出），不报错阻断。
    """
    if base_v is None or refined_v is None or ratio <= 0.0:
        return refined_v
    if base_v.dim() != 5 or base_v.shape != refined_v.shape \
            or base_v.shape[-2] < 3 or base_v.shape[-1] < 3:
        return refined_v
    r = min(max(float(ratio), 0.0), 1.0)
    b = base_v.detach().to("cpu", torch.float32)
    out = refined_v.detach().to("cpu", torch.float32)
    if out.data_ptr() == refined_v.data_ptr():   # 无拷贝路径（detach/同设备 to 共享存储）→ 克隆防改入参
        out = out.clone()

    def _lp(x):
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
        return torch.nn.functional.avg_pool2d(xp, 3, stride=1)

    for t0 in range(0, out.shape[2], chunk):
        sl = slice(t0, t0 + chunk)
        out[0, :, sl] += r * (_lp(b[0, :, sl]) - _lp(out[0, :, sl]))
    return out


def time_bias_sigma(sigma_v, sigma_start, bias, start_progress=0.70,
                    end_progress=1.0):
    """尾段时间偏置核心数学（Detail-Daemon 类，T8mars 机制移植，纯函数可单测）。

    精化进度 p = 1 − σ/σ₀（0=精化开始，1=收尾）；在 [start_progress,
    end_progress] 尾窗内以 smoothstep 渐入，把模型「看到的 σ」向更干净方向
    （更小）偏置 bias。不加噪声、不改积分 σ 与前向次数——只是模型时间观感。
    bias≤0 或 σ₀≤0 时原样返回（关）。σ clamp ≥0（精化网格最小评估 σ 远大于
    bias，钳 0 仅兜底）。
    """
    if bias <= 0.0 or sigma_start <= 0.0:
        return sigma_v
    p = 1.0 - min(sigma_v / sigma_start, 1.0)
    span = max(end_progress - start_progress, 1e-6)
    w = min(max((p - start_progress) / span, 0.0), 1.0)
    w = w * w * (3.0 - 2.0 * w)             # smoothstep 渐入
    return max(sigma_v - w * bias, 0.0)


def _hash_params(cfg):
    """进指纹/记录的参数字典（params_hash 与 write_record.params 同口径）。"""
    keys = {k: cfg[k] for k in _PARAM_KEYS}
    if cfg.get("time_bias"):
        keys["time_bias"] = cfg["time_bias"]
    if cfg.get("mix"):
        keys["mix"] = cfg["mix"]
    return keys


def params_hash(cfg):
    """二采参数 -> 8 位指纹（mode/include 不进：不影响单段输出）。

    time_bias / mix 仅在 >0（启用）时进指纹——默认关闭不改变哈希，既有二采
    记录不因新增参数而失效重做。
    """
    return checkpoint.fingerprint(_hash_params(cfg))


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


class UpscaleAbortError(RuntimeError):
    """二采致命问题（显存不足 / 画布越界 / 放大模型缺失）——发现即在真正占显存前
    输出报告并终止整链。主循环单独 catch 它并 re-raise（不上报告且不降级），
    避免产出"混一段高清一段基础"的不一致结果。"""


def _diff_model(model):
    """从 ComfyUI 模型包装器取底层 diffusion model（拿不到就原样返回）。"""
    dm = getattr(model, "model", None)
    if dm is not None:
        return getattr(dm, "diffusion_model", dm)
    return model


def _vram_gb():
    """当前可回收集显存（GB，comfy 语义；失败回退 torch，再失败返回 0）。"""
    try:
        import comfy.model_management as mm
        return mm.get_free_memory() / (1024 ** 3)
    except Exception:
        try:
            import torch
            return torch.cuda.mem_get_info()[0] / (1024 ** 3)
        except Exception:
            return 0.0


_UNET_SIZE_CACHE = {}


def _unet_size_gb(model):
    """UNET 权重字节（GB，按 id 缓存——跟随生成每段预检不用重算）。"""
    key = id(model)
    v = _UNET_SIZE_CACHE.get(key)
    if v is None:
        try:
            dm = _diff_model(model)
            v = sum(p.numel() * max(p.element_size(), 2)
                    for p in dm.parameters()) / (1024 ** 3)
        except Exception:
            v = 0.0
        _UNET_SIZE_CACHE[key] = v
    return v


_CANVAS_ABORT_MP = 12.0    # 二采画布超此（MP）→ 硬停（VAE 解码 + 显存双重爆点）
_ACTIVATION_FACTOR = 4.0   # 采样峰值激活 ≈ 初始高清 latent 体积 × 系数（经验折中）
_SAFE_MARGIN_GB = 1.0      # 显存账目安全余量（避免贴着上沿静默崩）


def _calc_scale_cap(h_latent, w_latent):
    """画布不超 _CANVAS_ABORT_MP 的最大放大倍率（向下取整到 0.5）。"""
    cur = (w_latent * 16) * (h_latent * 16)
    if cur <= 0:
        return 1.0
    cap = (_CANVAS_ABORT_MP * 1e6 / cur) ** 0.5
    return max(1.0, int(cap * 2) / 2.0)


def preflight(模型, cfg, net, video_t, audio_t, report=None):
    """二采前置健康预检——在【放大 / 分配高清内存之前】跑，把问题一次暴露：

    - 放大模型就绪；
    - 二采画布（target_hw ×16）超上限即停并提示可用的最大倍率；
      >2.5MP 提示 fp16 高频溢出花屏风险（不硬停，不挡正当高清需求）；
    - 显存账目 = 当前可回收集 对比 高清重采样新增峰值（不含已在显存的 UNET），
      连最小新增需求都盖不住即停。

    每行 "✓/⚠/✗ …"，任一 ✗ -> 完整报告 append 进 report 并抛 UpscaleAbortError，
    终止整链。正常返回报告行（供调用方登记，非致命 ⚠ 已含）。
    """
    lines = []
    fail = []

    # 1) 放大模型就绪
    try:
        dev = next(net.parameters()).device
        lines.append(f"✓ 放大模型就绪：{cfg.get('arch', '2D')} @ {dev}")
    except (StopIteration, AttributeError):
        lines.append("✗ 放大模型不可用——请先在二采面板选择有效权重")
        fail.append("放大模型不可用")

    # 2) 二采画布（latent 偶数 -> 像素 ·32）
    h, w = video_t.shape[-2], video_t.shape[-1]
    scale = float(cfg.get("scale", 2.0) or 2.0)
    h2, w2 = target_hw(h, w, scale)
    tw, th = h2 * 16, w2 * 16
    mp = tw * th / 1e6
    lines.append(f"… 画布：基础 {w * 16}×{h * 16} → 二采 {tw}×{th}"
                 f"（{mp:.1f}MP，×{scale:g}）")
    if mp > _CANVAS_ABORT_MP:
        cap = _calc_scale_cap(h, w)
        lines.append(f"✗ 二采画布 {tw}×{th}（{mp:.1f}MP）超上限 {_CANVAS_ABORT_MP:.0f}MP"
                     f"——VAE 解码与显存双重爆点；请把放大倍率降到 ≤{cap:g}×")
        fail.append(f"画布超限 {mp:.1f}MP")
    elif mp > 2.5:
        lines.append(f"⚠ 高清画布 {mp:.1f}MP 超 2.5MP 安全解码区，注意 fp16 高频溢出"
                     f"（出花屏/色块就降倍率或提采样精度）")

    # 3) 显存账目：当前可回收集 vs 高清重采样新增峰值（UNET 已在显存，不计双重）
    free = _vram_gb()
    unet_gb = _unet_size_gb(模型)
    frames = grid.latent_t_to_frames(video_t.shape[2])
    latent_gb = (int(video_t.shape[1]) * frames * h2 * w2 * 4.0) / (1024 ** 3)
    need = latent_gb * _ACTIVATION_FACTOR + _SAFE_MARGIN_GB
    lines.append(f"… 显存账目：UNET≈{unet_gb:.1f}GB（已在显存）· 高清重采样需新增≈"
                 f"{need:.1f}GB（含 {_SAFE_MARGIN_GB:.1f}GB 余量）· 当前可回收集 {free:.1f}GB")
    if free > 0 and free < need:
        cap = _calc_scale_cap(h, w)
        lines.append(f"✗ 显存不足：放大后重采样至少还需 {need:.1f}GB，当前可回收集仅 {free:.1f}GB"
                     f"——请把放大倍率降到 ≤{cap:g}×、把精化步数调低/起始σ调小，或降低基础"
                     "分辨率/增大显存。本报告在分配任何高清内存前生成。")
        fail.append("显存不足")

    if fail:
        msg = "二采健康预检未通过（已停止运行）：\n" + "\n".join(lines)
        if report is not None:
            report.append(msg)
        raise UpscaleAbortError(msg)
    return lines


def time_bias_guard(dit, sigma_start, bias):
    """尾段时间偏置守卫：patch dit.forward，返回恢复函数（try/finally 调用）。

    Detail-Daemon 类技巧（T8mars/comfyui-minimax-h3-audio-T8 机制移植，GPL）：
    在精化尾窗内只把喂给共享 AV Transformer 的 timestep（=σ×1000）向更干净方向
    偏置——模型「看到稍干净的时间」从而多抠细节；不加噪声、NFE 不变。二采
    输出的音频 latent 被丢弃，偏置波及音频 token 的时间嵌入也不影响产物。
    与 cond_audio_rows_guard（挂 _cond_audio_rows 属性）不冲突；bias≤0 不安装。
    """
    orig_forward = dit.forward

    def patched_forward(x, timestep, context, transformer_options={}, **kwargs):
        sigma_v = float((timestep.flatten()[0] / 1000.0).clamp(min=1e-6))
        seen = time_bias_sigma(sigma_v, sigma_start, bias)
        if seen != sigma_v:
            timestep = torch.full_like(timestep, seen * 1000.0)
        return orig_forward(x, timestep, context, transformer_options, **kwargs)

    dit.forward = patched_forward
    return lambda: setattr(dit, "forward", orig_forward)


def render_latent(模型, clip, video_vae, audio_vae, negative, cfg, net,
                  video_t, audio_t, kind, idx, seg_prompts, seg_label_orders,
                  pool_tensors, refs, first_frame, guide, tail_kf_latent, cur_seed,
                  采样器, 调度器, report=None):
    """基础段 AV latent -> 高清视频 latent（放大 + 低强度重采样）。

    kind: "prompt"（提示词段，cond 带本段提示词/参考素材/首帧）
          / "insert"|"prologue"（外部素材段，空提示词轻精修）。
    guide: 本段生成时用的上段尾帧桥（None=首段/断链/插入段）——视频 latent
    神经放大后注入重采样 cond（CondSync：锚住段首与上段高清尾的连续性）。
    tail_kf_latent: 尾帧身份锚定的基础 latent（与主循环同语义，同步放大注入）。
    cfg.mix>0 时精化输出做频域细节混合（低频锚回纯放大 latent，见
    freq_mix_latents）；细节增益在混合后的交付 latent 上度量。
    返回 (up_out_v 高清视频 latent, tw, th, 二采种子, 是否桥锚, 细节增益小数)。
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

    # 二采真语义：先由放大网络把 latent 超分到 scale×，再在【高清 latent】上低强度重采样。
    # 放大网络的逐段输出在此被使用（重采样初始 latent），不存在「白放大」。
    # 显存账：H3 UNET fp16 常驻 ~26GB，2× 高清激活 ~5.4GB（=基础 4×），26+5.4≈31.4GB，
    # 正好贴着 32GB 卡的 31.36 限制上沿——故必须把联动的 CLIP + video/audio VAE 一起腾干净：
    # 重采样前 unload_all_models()（全卸到 CPU，含 UNET），再由 common_ksampler 原生机制
    # 只回载 UNET 到 GPU——CLIP/VAE 不再占显存，26GB UNET + 高清激活即可放下。
    # 未触发 OOM 则原样跑；一级降级=全卸只回 UNET；二级降级=LOW_VRAM 分块兜底。
    h, w = video_t.shape[-2], video_t.shape[-1]
    tw, th = target_hw(h, w, cfg["scale"])[0] * 16, target_hw(h, w, cfg["scale"])[1] * 16
    length = grid.latent_t_to_frames(video_t.shape[2])

    # 前置健康预检：在放大/分配高清内存【之前】把显存不足、画布越界、模型缺失
    # 一次暴露——任一 ✗ 抛 UpscaleAbortError，由主循环 re-raise 终止整链（不降级）。
    preflight(模型, cfg, net, video_t, audio_t, report)

    # 放大网络放大视频 latent 到 scale×（常驻 GPU，放大完卸回 CPU 腾给高清重采样）；
    # hf_up = 纯放大（无精化）的高频能量基线——细节增益度量的「前」
    up_v = upscale_video(video_t, net, cfg["scale"], cfg["arch"])
    hf_up = latent_hf_energy(up_v)
    # 频域细节混合启用时保留纯放大 latent 的 CPU 副本（避开采样期显存峰值，
    # 精化后作低频锚；关闭时零开销不复制）
    mix = float(cfg.get("mix") or 0.0)
    up_v_base = up_v.detach().to("cpu", torch.float32) if mix > 0.0 else None

    # cond 构造：按【高清目标分辨率】（重采样画布），refs/首帧由官方节点编码到高清尺寸
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

    # 桥锚 CondSync：上段尾帧桥/尾帧锚的 latent 同步神经放大后注入重采样 cond——
    # 锚住段首与上段高清尾的连续性（只依赖上段基础 latent + scale，可独立重做）
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

    # 放大网络工作完毕，卸回 CPU 释放显存给高清重采样
    net.cpu()
    torch.cuda.empty_cache()

    # 低强度重采样（高清 latent）：初始 = 放大后的视频 latent + 原音频 latent；
    # 采样输出的音频丢弃，分段/成片音轨 = 原轨（零音频回归）。
    # 参数即「sigma 尾段精化」直接语义：denoise=尾段起始 σ（默认 0.35 低噪声区间）、
    # steps=精化步数（默认 6，建议 3-8；参考工作流即「放大后 3-5 步低噪声 refine」）。
    # 提分辨率靠放大网络、二采只补细节，显存开销小；tail_refine_args 把两个
    # 直接语义换算回 common_ksampler(denoise) 的尾段截断口径。
    seed = (int(cur_seed) + 1) % 0xffffffffffffffff if cur_seed is not None else 1
    latent["samples"] = comfy.nested_tensor.NestedTensor(
        (up_v.to(dev, torch.float32), audio_t.to(dev, torch.float32)))
    del up_v

    restore_rows = plugin_nodes.cond_audio_rows_guard(模型.model.diffusion_model)

    def _is_oom(e):
        return "out of memory" in str(e).lower()

    # ⚠ 显存不足直接报错停止（用户明确要求，不做静默降级）：二采在高清 latent 上
    # 采样需要额外显存，若 32GB 卡放不下 26GB UNET + 高清激活，当场抛出明确错误，
    # 前端/报告会终止整链而不是降级基础分辨率继续（避免产出不一致的混合清晰度）。
    ks_steps, ks_denoise = tail_refine_args(cfg["denoise"], cfg["steps"])
    n_eff = int(ks_steps * ks_denoise)
    tb = float(cfg.get("time_bias") or 0.0)
    restore_tb = time_bias_guard(模型.model.diffusion_model, n_eff / ks_steps, tb) \
        if tb > 0.0 else None
    try:
        sampled = comfy_nodes.common_ksampler(
            模型, seed, ks_steps, cfg["cfg"], 采样器, 调度器,
            cond, negative, latent, denoise=ks_denoise)[0]
    except RuntimeError as e:
        if _is_oom(e):
            msg = ("二采显存不足（放大后 {0:g}× 高清 latent 上的重采样在超出当前显存 {1:g}GB）："
                   "请降低放大倍率、把精化步数调低/起始σ调小，或增大显存。"
                   "当前精化 {2} 步 @ σ≈{3:g}。本段尚未落盘二采产物，可先行释放其他模型后重试。".format(
                       cfg["scale"], _vram_gb(), n_eff, n_eff / ks_steps))
            if report is not None:
                report.append(msg)
            raise UpscaleAbortError(msg) from e
        raise
    finally:
        restore_rows()
        if restore_tb is not None:
            restore_tb()

    up_out_v, _discarded_audio = sampled["samples"].unbind()
    del cond, latent, sampled
    torch.cuda.empty_cache()
    if up_v_base is not None:
        # 频域细节混合（CPU 分块）：低频锚回纯放大 latent、高频保留精化增益，
        # 细节增益度量在【混合后的交付 latent】上取值（报告行如实反映）
        up_out_v = freq_mix_latents(up_v_base, up_out_v, mix).to(
            up_out_v.device, torch.float32)
        del up_v_base
    hf_gain = hf_gain_ratio(hf_up, latent_hf_energy(up_out_v))
    return up_out_v, tw, th, seed, bridged, hf_gain


def render_segment(模型, clip, video_vae, audio_vae, negative, cfg, net,
                   root, g, video_t, audio_t, kind, idx,
                   seg_prompts, seg_label_orders, pool_tensors, refs,
                   first_frame, guide, tail_kf_latent, cur_seed,
                   skip_f, vis_len,
                   wav, sample_rate, bh, report, 采样器, 调度器):
    """基础段 AV latent -> 高清分段直接落盘（放大→重采样→解码→裁剪）。

    主循环逐段调用（采样定稿/回放载入之后、基础段落盘之前）：分段视频与
    缩略图沿用基础段同名（单份产物——seg_NNN.mp4 即高清结果），另存尾帧锚
    uplast_NNN.png 供下游使用，并原子写 manifest.upscale 记录。
    裁剪口径与主循环 _decode_crop 完全一致：skip_f/vis_len 由调用方按基础
    分辨率帧的门控/切镜决策传入（机制零漂移，二采只接管落盘的帧）；
    音频沿用基础段原轨（零音频回归）。跨缝连续性由生成期桥锚 keyframe
    （_apply_guide）兜底，二采路径不再做任何像素级平滑。
    返回 (高清尾帧 CPU tensor, 最新 upscale 存档状态)；异常向上抛，由调用方
    降级为基础分辨率保存（基础链产物不受影响）。
    """
    import comfy.model_management

    t0 = time.perf_counter()
    purge_legacy(root, g)

    up_v, tw, th, up_seed, bridged, hf_gain = render_latent(
        模型, clip, video_vae, audio_vae, negative, cfg, net,
        video_t, audio_t, kind, idx, seg_prompts, seg_label_orders,
        pool_tensors, refs, first_frame, guide, tail_kf_latent, cur_seed,
        采样器, 调度器, report=report)
    # 解码高清 latent -> 高清帧（官方 VAE.decode 自带 OOM→tiled 降级，无需干预）
    frames = video_vae.decode(up_v)
    del up_v
    torch.cuda.empty_cache()
    if len(frames.shape) == 5:
        frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
    frames = frames[skip_f:skip_f + vis_len]
    if not checkpoint.save_segment_mp4(root, g, frames, wav, sample_rate, fresh=True):
        raise RuntimeError("高清分段编码失败（save_av_mp4 返回失败，详见 ComfyUI 控制台输出）")
    checkpoint.save_thumb(root, g, frames[0])
    files = checkpoint.upscale_files(g)
    _save_png(os.path.join(root, files["last"]), frames[-1])
    up_state = write_record(root, g, cfg, up_seed, (tw, th), bh, hf_gain=hf_gain)
    _n, _d = tail_refine_args(cfg["denoise"], cfg["steps"])
    _n_eff = int(_n * _d)
    _tb = float(cfg.get("time_bias") or 0.0)
    _mix = float(cfg.get("mix") or 0.0)
    report.append(f"段{g + 1} 二采：{tw}×{th} · {cfg['arch']} {cfg['scale']:g}× · "
                  f"精化 {_n_eff} 步 @ σ≈{_n_eff / _n:g} · 细节 {hf_gain:+.0%} · "
                  f"{time.perf_counter() - t0:.0f}s"
                  + (" · 桥锚" if bridged else "")
                  + (f" · 偏置{_tb:g}" if _tb > 0 else "")
                  + (f" · 混合{_mix:g}" if _mix > 0 else ""))
    # 收尾：释放二采残留（放大 latent / 解码帧 / 重采样缓存），给下段基础采样腾显存
    torch.cuda.empty_cache()
    return frames[-1].detach().float().cpu(), up_state


def write_record(root, g, cfg, seed, size, bh, hf_gain=None):
    """段 g 高清渲染记录原子写盘（重读 manifest 防竞态覆盖并发进度）。

    bh=该段基础身份指纹（调用方用本地 full_hashes/seeds 现算，不依赖磁盘
    manifest 的写入时机）；hf_gain=细节增益小数（仅记录供 ab_report 复盘，
    不参与重做判定）；返回最新 upscale dict（调用方回填 proj_upscale，
    后续主循环的 manifest 快照写盘才不会把记录冲掉）。
    """
    ph = params_hash(cfg)
    rec = {"hash": ph, "base_hash": bh, "seed": seed, "done": True,
           "files": checkpoint.upscale_files(g), "size": list(size)}
    if hf_gain is not None:
        rec["hf_gain"] = round(float(hf_gain), 4)
    fresh = checkpoint.load_manifest(root) or {}
    up_state = dict(fresh["upscale"]) if isinstance(fresh.get("upscale"), dict) else {}
    segs = list(up_state.get("segs") or [])
    while len(segs) <= g:
        segs.append(None)
    segs[g] = rec
    up_state["segs"] = segs
    up_state["hash"] = ph
    up_state["params"] = _hash_params(cfg)
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
