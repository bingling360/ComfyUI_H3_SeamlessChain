"""H3 Seamless Chain —— MiniMax H3 多段视频段间引导（无缝续拍）单节点。

原理：生成第 N+1 段时，把第 N 段结尾 context_frames 帧从其采样输出的 AV latent
里直接切片（不解码不重编码，零颜色漂移），作为 keyframe 钉在第 N+1 段头部
（官方 conditioning 协议 minimax_keyframes，锚 resolved_frame_index=0），
采样每步重注入，顺着上一段尾部的运动继续画；解码后裁掉头部重叠桥再拼接。

支持逐段审片（每次运行只生成一段新内容即返回，重新运行继续）与任意段重跑
（改某段提示词自动从该段重做；「重跑起始段」+ 换种子可从指定段重摇），
进度存于 latent 断点（v2 schema：逐段种子 / 提示词哈希 / 裁剪量）。

输入形态与官方 MiniMaxH3ReferenceToVideo 对齐（autogrow）：
- 提示词_1..N：每段一个输入框（或接 PrimitiveStringMultiline）
- 参考图片_0..9（<Picture i>）/ 参考视频_0..3（<Video k>）
- 参考视频音轨_0..3（与同号视频配对）/ 参考音频_0..3（<Audio j>）
- 起始视频 + 起始视频音轨：可选序章——上传视频编码为第 0 段（进断点可回放），
  成片以它开头，后续生成段从其结尾续拍（24fps 约定，经一次 VAE 重编码）

配套「长片审片」侧栏面板（web/h3chain_panel.js）：断点目录里的缩略图 /
状态文件 / manifest 经 ComfyUI 自带 /api/view 端点直接读取，零自建路由。

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
from .grid import video_latent_t, latent_t_to_frames, audio_tokens_for_frames, align_frame_count_down

try:
    import comfy.ldm.minimax.model as _minimax_model
    # PR #15439 引入的模块级函数；存在即代表支持 keyframe 音频与 refs/keyframes 合并
    KEYFRAME_AUDIO_SUPPORTED = hasattr(_minimax_model, "_ref_t_span")
except Exception:
    _minimax_model = None
    KEYFRAME_AUDIO_SUPPORTED = False

_full_bridge_cache = None


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


def _decode_audio(audio_vae, audio_latent):
    # 与官方 VAEDecodeAudio（comfy_extras/nodes_audio.vae_decode_audio）保持一致
    audio = audio_vae.decode(audio_latent).movedim(-1, 1)
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    sample_rate = getattr(audio_vae, "audio_sample_rate_output",
                           getattr(audio_vae, "audio_sample_rate", 44100))
    return audio, sample_rate


def _tail_keyframe(video_t, audio_t, ctx_frames, with_audio, back_tokens=0, back_audio=0, full_bridge=True):
    """上段尾部 ctx 帧 latent 直切为 keyframe；back_* 为桥帧门控回退偏移。

    回退量必须是 5 token（17 帧）的倍数：切片起点/终点同步平移，
    切片覆盖帧数不变，且始终落在 17k+5 网格上。
    旧版 ComfyUI keyframe 协议只收单帧 latent：full_bridge=False 时只钉
    尾部最后 1 个 token（承载上段末尾画面），且不附音频。
    """
    vt = video_latent_t(ctx_frames) if full_bridge else 1
    end = video_t.shape[2] - back_tokens
    kf = {
        "resolved_frame_index": 0,
        "latent": video_t[:, :, end - vt:end, :, :].clone(),
    }
    if with_audio and full_bridge and audio_t is not None:
        at = audio_tokens_for_frames(ctx_frames)
        aend = audio_t.shape[-1] - back_audio
        if 0 < at <= aend:
            kf["audio_latent"] = audio_t[..., aend - at:aend].clone()
    return kf


def _center_cover(frames, width, height):
    """[F,H,W,3] -> [F,height,width,3]，与官方 _resize(..., "center") 同语义的 cover-crop。"""
    x = frames[..., :3].movedim(-1, 1).float()
    x = comfy.utils.common_upscale(x, width, height, "lanczos", "center")
    return x.movedim(1, -1)


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
                io.Combo.Input("断点续拍", options=["关闭", "自动续跑"], default="关闭",
                               tooltip="自动续跑：每段采样后立即落盘 latent 断点（约5MB/段），中断后重跑自动跳过已完成段，结果与一次跑完一致"),
                io.String.Input("断点目录", default="",
                                tooltip="空=按参数指纹自动命名于 output/checkpoints/ 下；换链请填新名字或清空旧目录。断点为中间产物，跑完可删"),
                io.Combo.Input("桥帧门控", options=["关闭", "标注", "自动回退"], default="标注",
                               tooltip="对将成为重叠桥的尾帧打分（Laplacian清晰度+曝光）：标注=只写报告；自动回退=尾帧低于阈值时向前回退17/34帧取好帧续拍（该段可见帧数随之减少）"),
                io.Float.Input("清晰度阈值", default=30.0, min=0.0, max=100.0, step=0.5,
                               tooltip="桥帧总分阈值，低于判定为坏尾。建议先跑「标注」档看报告里的分数分布再定"),
                io.Int.Input("回退上限", default=34, min=0, max=68, step=17,
                             tooltip="自动回退最多向前多少帧（17 的倍数，踩 17k+5 网格）"),
                io.Combo.Input("接缝混合", options=["关闭", "smoothstep"], default="smoothstep",
                               tooltip="拼接点像素级兜底（一体化总控台同款机制）：上一段最后可见帧硬锁为本段首帧"
                                       "（逐像素一致），smoothstep 权重窗把本段前「混合帧数」帧的偏差平滑吸收，"
                                       "音频同步从上一段尾音渐入。仅作用于生成段头部；插入段/序章完整保留。"
                                       "报告中的接缝帧差仍为混合前的模型偏差（诊断用）"),
                io.Int.Input("混合帧数", default=6, min=1, max=24,
                             tooltip="smoothstep 混合窗帧数，两端权重导数为 0、中段过渡。"
                                     "运动越快窗应越短（6帧≈0.25s）；主体位移差大时长窗会拉长叠影"),
                io.Combo.Input("审片模式", options=["关闭", "逐段确认"], default="关闭",
                               tooltip="逐段确认：每次运行只生成一个新的段落即返回，预览「分段图像」后重新运行继续下一段；"
                                       "不满意可改该段提示词（自动从该段重跑）或设「重跑起始段」重摇。开启后断点自动启用"),
                io.Int.Input("重跑起始段", default=0, min=0, max=63,
                             tooltip="0=自动（沿用断点进度，改过提示词的段自动重做）；N=从第 N 段起丢弃存档重新生成"
                                     "（有序章时序章为第 1 段），配合改「种子」即可重摇该段及之后。用完记得改回 0"),
                io.Image.Input("首帧图片", optional=True,
                               tooltip="第一段的起始帧（i2v）。用了它请用 fl2va UNET，且不能同时用任何参考素材"),
                io.Image.Input("起始视频", optional=True,
                               tooltip="序章：上传视频（≥5 帧、24fps，超长只取前「每段帧数」内）编码为第 1 段存入断点，"
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
                种子, 步数, CFG, 采样器, 调度器, 首帧图片=None, 起始视频=None, 起始视频音轨=None,
                提示词组=None,
                参考图片组=None, 参考视频组=None, 参考视频音轨组=None, 参考音频组=None,
                断点续拍="关闭", 断点目录="", 桥帧门控="标注", 清晰度阈值=30.0, 回退上限=34,
                接缝混合="smoothstep", 混合帧数=6,
                审片模式="关闭", 重跑起始段=0):
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

        # 断点指纹只覆盖共享参数（不含提示词、不含种子）：改某段提示词仍指向同一条链，
        # 重跑起点由逐段提示词哈希比对定位；种子控件开着 control_after_generate 每次运行
        # 自动 +1，真正的种子序列由 manifest 权威记录（见下方续跑载入逻辑）
        resume = 断点续拍 == "自动续跑"
        review = 审片模式 == "逐段确认"
        reroll = max(0, int(重跑起始段))
        use_ckpt = resume or review          # 审片必须落盘才能跨次运行续接
        if review and not resume:
            report.append("审片模式：断点自动启用（每段落盘 latent，跨次运行续接）")
        elif reroll > 0:
            report.append("注意：「重跑起始段」仅在断点续拍/审片模式下生效，本次已忽略")
        ckpt_params = {
            "width": width, "height": height,
            "length": length, "ctx": ctx, "steps": int(步数), "cfg": float(CFG),
            "sampler": 采样器, "scheduler": 调度器, "chain": chain,
            "gate": {"mode": 桥帧门控, "threshold": float(清晰度阈值), "limit": gate_limit},
        }
        seg_hashes = [checkpoint.prompt_hash(p) for p in seg_prompts]
        prologue_hash = None
        if 起始视频 is not None:
            f0 = 起始视频[0].detach().float().cpu()
            prologue_hash = checkpoint.prompt_hash(
                f"prologue:{int(起始视频.shape[0])}:{float(f0.mean()):.4f}:{float(f0.std()):.4f}")
        root, manifest, done, seeds = None, None, 0, []
        off = 1 if 起始视频 is not None else 0
        if use_ckpt:
            root = checkpoint.ckpt_dir(ckpt_params, 断点目录.strip())
            manifest = checkpoint.load_manifest(root)
            if manifest is not None:
                if manifest.get("schema") != checkpoint.SCHEMA:
                    raise ValueError(f"断点目录格式不认识（{manifest.get('schema')}），请换一个目录名；"
                                     "旧版 v1 断点不兼容本版本，请清空旧目录或换新名字")
                checkpoint.assert_match(manifest["params"], ckpt_params)
                if 起始视频 is None and manifest.get("has_prologue"):
                    off = 1  # 输入已断开仍沿用断点序章（LoadVideo 可 bypass），哈希校验跳过
                elif 起始视频 is not None and not manifest.get("has_prologue"):
                    manifest = checkpoint.truncate(root, manifest, 0)
                    report.append("断点续跑：检测到新接入的序章视频，整链重做")
                elif 起始视频 is not None and prologue_hash is not None:
                    stored = list(manifest.get("prompt_hashes", []))
                    if stored and stored[0] != prologue_hash:
                        manifest = checkpoint.truncate(root, manifest, 0)
                        report.append("断点续跑：序章视频已更换，整链重做")
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
                    report.append(f"断点续跑：{原因}，从段 {start + 1} 起重新生成"
                                  + (f"（段 1-{start} 沿用断点）" if start else "（整链重做）"))
                seeds = [int(s) for s in manifest.get("seeds", [])]
                report.append(f"断点续跑：载入已完成 {done}/{len(seg_prompts) + off} 段，目录 {root}")
                if reroll == 0 and seeds:
                    report.append("控件种子仅在「重跑起始段」> 0 时生效，当前沿用断点种子序列")
        full_hashes = ([prologue_hash] if off else []) + seg_hashes
        if review:
            report.append("审片面板：左侧「长片审片」侧栏（旧版前端为悬浮按钮）可逐段播放分段视频、"
                          "看缩略图与接缝/桥分，一键继续下一段 / 重摇此段 / 改词重跑")

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
                prologue_origin = "断点载入"
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
                        "seams": [None], "bridge_scores": [None], "params": ckpt_params})
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
            guide = _tail_keyframe(pv, pa, ctx, False,  # ComfyUI 0.33+ H3 模型音频 token 计算变更，keyframe 带音频会形状错位；音频接缝改由 smoothstep_fade_head 兜底
                                   full_bridge=full_bridge)
            report.append(f"段1/{total}：{prologue_origin} 序章 留{pframes.shape[0]}帧 · 种子 — | guide=无（序章）")

        for i, prompt in enumerate(seg_prompts):
            g = i + off  # 全局段下标（有序章时序章占 0 号）
            replay = use_ckpt and g < done
            if replay:
                video_t, audio_t = checkpoint.load_segment(root, g)
                video_t = video_t.to(video_vae.device)
                audio_t = audio_t.to(audio_vae.device)
                dt = 0.0
                cur_seed = seeds[g] if g < len(seeds) else None
            else:
                # 种子规则：重摇（重跑起始段>0）用控件种子；否则延续断点种子序列的等差，
                # 使审片多轮运行与一次跑完逐帧一致；断点无生成段种子（仅序章/新链）才用控件种子
                if reroll > 0:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                elif len(seeds) > off:
                    cur_seed = (seeds[-1] + g - len(seeds) + 1) % 0xffffffffffffffff
                else:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                first_free = (i == 0 and off == 0)  # 无序章时首段无重叠桥
                seg_len = length + (0 if first_free else ctx)
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
                if guide is not None:
                    cond = cls._apply_guide(
                        cond, guide, latent_t_to_frames(latent["samples"].tensors[0].shape[2]))

                t0 = time.perf_counter()
                sampled = nodes.common_ksampler(
                    模型, cur_seed, 步数, CFG,
                    采样器, 调度器, cond, negative, latent, denoise=1.0)[0]
                dt = time.perf_counter() - t0
                video_t, audio_t = sampled["samples"].unbind()
                # 维持不变量 seeds[g] = 该段种子（段文件缺失导致 done 回退时，
                # 重做段的种子可能与 manifest 残留记录相同，按下标赋值避免列表重复错位）
                if g < len(seeds):
                    seeds[g] = cur_seed
                else:
                    seeds.append(cur_seed)
                if use_ckpt:
                    checkpoint.save_segment(root, g, video_t, audio_t)

            sampled_fc = latent_t_to_frames(video_t.shape[2])
            frames = video_vae.decode(video_t)
            if len(frames.shape) == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            wav, sample_rate = _decode_audio(audio_vae, audio_t)

            skip_f = 0 if (i == 0 and off == 0) else ctx
            vis_len = length
            seg_bridge_score = None
            if 桥帧门控 != "关闭" and i + 1 < len(seg_prompts):
                window = frames[max(skip_f, frames.shape[0] - (ctx + gate_limit)):]
                score = qc.frame_scores(window)
                tail_score = float(score[-1])
                seg_bridge_score = round(tail_score, 2)
                if 桥帧门控 == "标注":
                    flag = " ↓ 低于阈值" if tail_score < 清晰度阈值 else ""
                    report.append(f"段{g + 1} 桥帧总分 {tail_score:.1f}{flag}")
                else:  # 自动回退
                    back, hit = qc.pick_backtrack(score, gate_limit, float(清晰度阈值))
                    if back:
                        vis_len = length - back
                        report.append(f"段{g + 1} 尾帧低质（{tail_score:.1f} < {清晰度阈值:g}），"
                                      f"回退 {back} 帧续拍（回退点 {hit:.1f}）")
            bridge_scores.append(seg_bridge_score)
            trims.append(length - vis_len)

            frames = frames[skip_f:skip_f + vis_len]
            wav_total = wav.shape[-1]
            skip_s = round(wav_total * skip_f / sampled_fc)
            take_s = round(wav_total * vis_len / sampled_fc)
            wav = wav[..., skip_s:skip_s + take_s]

            # 接缝后验测量（测而不干预）：上一段最后可见帧 vs 本段首帧
            seam_n = int(sample_rate * 0.25)
            if (i > 0 or off) and prev_tail_frame is not None:
                d, db = qc.seam_metrics(prev_tail_frame, frames[0],
                                        prev_tail_wav, wav[..., :seam_n], rate=sample_rate)
                db_txt = f"{db:+.1f} dB" if db is not None else "—"
                flag = " ↑ 建议人工检查" if d > 0.08 or (db is not None and abs(db) > 6.0) else ""
                report.append(f"段{g + 1} 接缝：帧差 {d:.3f} · 响度跳变 {db_txt}{flag}")
                seams.append([round(d, 4), None if db is None else round(db, 2)])
            else:
                seams.append(None)

            # 接缝像素级兜底（生成后、拼接前）：锚帧硬锁 + smoothstep 窗吸收。
            # 与生成前 latent 引导互补——引导管"画得像"，混合管"拼得上"；
            # 纯像素后处理，不进断点指纹（改此参数不触发重跑，replay 段结果仍一致）
            if 接缝混合 != "关闭" and (i > 0 or off) and prev_tail_frame is not None:
                span = min(max(1, int(混合帧数)), frames.shape[0])
                frames = qc.smoothstep_blend_head(frames, prev_tail_frame, span)
                if prev_tail_wav is not None:
                    wav = qc.smoothstep_fade_head(wav, prev_tail_wav)
                report.append(f"段{g + 1} 接缝平滑：首帧硬锁锚帧 + smoothstep {span} 帧过渡（音频同步渐入）")

            all_frames.append(frames.cpu())
            seg_frames.append(frames.cpu())
            seg_wav = wav.cpu()
            seg_wavs.append({"waveform": seg_wav, "sample_rate": sample_rate})
            all_wav = seg_wav if all_wav is None else torch.cat([all_wav, seg_wav], dim=-1)
            prev_tail_frame = frames[-1].cpu()
            prev_tail_wav = seg_wav[..., -seam_n:]
            if use_ckpt:
                thumbs.append(checkpoint.save_thumb(root, g, frames[0]))
                videos.append(checkpoint.save_segment_mp4(root, g, frames, wav, sample_rate,
                                                          fresh=not replay))
            else:
                thumbs.append("")
                videos.append("")

            if guide is not None:
                note = (f"guide=上段尾{ctx}帧" if full_bridge else "guide=单帧桥(旧协议降级)") \
                    + ("+音频" if "audio_latent" in guide else "")
            else:
                note = "guide=无（首段）"
            origin = "断点载入" if replay else f"采样{sampled_fc}帧"
            seed_txt = cur_seed if cur_seed is not None else "—"
            report.append(f"段{g + 1}/{total}：{origin} 裁头{skip_f}帧 留{frames.shape[0]}帧"
                          f" · 种子 {seed_txt}" + ("" if replay else f" · 采样 {dt:.0f}s") + f" | {note}")

            if i + 1 < len(seg_prompts):
                back = length - vis_len
                guide = _tail_keyframe(video_t, audio_t, ctx, False,  # 同上：ComfyUI 0.33+ 关闭 keyframe 音频
                                       back_tokens=back // 17 * 5,
                                       back_audio=round(back * 5.0 / 3.0),
                                       full_bridge=full_bridge)

            if use_ckpt and not replay:
                done = g + 1
                checkpoint.save_manifest(root, {
                    "schema": checkpoint.SCHEMA, "done": done, "has_prologue": bool(off),
                    "seeds": list(seeds[:done]), "trims": list(trims[:done]),
                    "prompt_hashes": full_hashes[:done],
                    "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                    "prompts": prompt_list[:done],
                    "seams": seams[:done], "bridge_scores": bridge_scores[:done],
                    "params": ckpt_params,
                })

            pbar.update(1)

            if review and not replay and i + 1 < len(seg_prompts):
                report.append(f"审片：段 {g + 1} 已完成并落盘 → 预览「分段图像」或左侧「长片审片」面板；"
                              f"满意请直接重新运行继续段 {g + 2}；不满意：改「提示词_{i + 1}」后运行（自动从本段重跑），"
                              f"或设「重跑起始段」={g + 1} 并改「种子」重摇")
                break

        if use_ckpt:
            # 兜底回写：纯回放运行（无新段）也会重解码全部段，把缩略图/分段视频/指标
            # 等增量键补齐——旧版本断点的 manifest 缺这些键时由此自愈，面板无需重跑整链
            checkpoint.save_manifest(root, {
                "schema": checkpoint.SCHEMA, "done": done, "has_prologue": bool(off),
                "seeds": list(seeds[:done]), "trims": list(trims[:done]),
                "prompt_hashes": full_hashes[:done],
                "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                "prompts": prompt_list[:done],
                "seams": seams[:done], "bridge_scores": bridge_scores[:done],
                "params": ckpt_params,
            })
            if review:
                if len(seg_frames) == total:
                    report.append("审片：本链已全部完成")
                if reroll > 0:
                    report.append(f"注意：「重跑起始段」={reroll} 已生效，确认无误后请改回 0（审片面板会在成功后自动复位）")
            report.append(f"断点目录（含中间 latent，跑完可删）：{root}")
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll,
                                   "report": "\n".join(report), "updated_at": time.time()})

        images = torch.cat(all_frames, dim=0)
        return io.NodeOutput(
            images,
            {"waveform": all_wav, "sample_rate": sample_rate},
            24,
            "\n".join(report),
            seg_frames,
            seg_wavs,
        )

    @classmethod
    def IS_CHANGED(cls, 审片模式="关闭", 断点续拍="关闭", **kwargs):
        if 审片模式 == "逐段确认" or 断点续拍 == "自动续跑":
            return float("nan")   # 断点/审片激活时输入不变也强制真正执行（重读最新 manifest）
        return ""

    @staticmethod
    def _apply_guide(cond, guide, sampled_fc):
        """把上段尾帧 keyframe 注入 conditioning（官方 minimax_keyframes 协议）。"""
        return node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": [guide],
            "minimax_frame_count": sampled_fc,
        })
