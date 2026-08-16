"""H3 Seamless Chain —— MiniMax H3 多段视频段间引导（无缝续拍）单节点。

原理：生成第 N+1 段时，把第 N 段结尾 context_frames 帧从其采样输出的 AV latent
里直接切片（不解码不重编码，零颜色漂移），作为 keyframe 钉在第 N+1 段头部
（官方 conditioning 协议 minimax_keyframes，锚 resolved_frame_index=0），
采样每步重注入，顺着上一段尾部的运动继续画；解码后裁掉头部重叠桥再拼接。

输入形态与官方 MiniMaxH3ReferenceToVideo 对齐（autogrow）：
- 提示词_1..N：每段一个输入框（或接 PrimitiveStringMultiline）
- 参考图片_0..9（<Picture i>）/ 参考视频_0..3（<Video k>）
- 参考视频音轨_0..3（与同号视频配对）/ 参考音频_0..3（<Audio j>）

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
from comfy_api.latest import io

from . import checkpoint
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


def _tail_keyframe(video_t, audio_t, ctx_frames, with_audio, back_tokens=0, back_audio=0):
    """上段尾部 ctx 帧 latent 直切为 keyframe；back_* 为桥帧门控回退偏移。

    回退量必须是 5 token（17 帧）的倍数：切片起点/终点同步平移，
    切片覆盖帧数不变，且始终落在 17k+5 网格上。
    """
    vt = video_latent_t(ctx_frames)
    end = video_t.shape[2] - back_tokens
    kf = {
        "resolved_frame_index": 0,
        "latent": video_t[:, :, end - vt:end, :, :].clone(),
    }
    if with_audio and audio_t is not None:
        at = audio_tokens_for_frames(ctx_frames)
        aend = audio_t.shape[-1] - back_audio
        if 0 < at <= aend:
            kf["audio_latent"] = audio_t[..., aend - at:aend].clone()
    return kf


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
                io.Image.Input("首帧图片", optional=True,
                               tooltip="第一段的起始帧（i2v）。用了它请用 fl2va UNET，且不能同时用任何参考素材"),
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
                种子, 步数, CFG, 采样器, 调度器, 首帧图片=None, 提示词组=None,
                参考图片组=None, 参考视频组=None, 参考视频音轨组=None, 参考音频组=None,
                断点续拍="关闭", 断点目录="", 桥帧门控="标注", 清晰度阈值=30.0, 回退上限=34):
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

        length, width, height, seed = int(每段帧数), int(宽度), int(高度), int(种子)
        ctx = int(引导帧数)
        gate_limit = max(0, int(回退上限) // 17 * 17)
        if 桥帧门控 != "关闭":
            from . import qc  # torch 计算图较重，仅在启用门控时加载
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

        # 断点指纹不含种子：种子控件开着 control_after_generate 每次运行自动 +1，
        # 真正的种子以 manifest 权威记录为准（见下方续跑载入逻辑）
        resume = 断点续拍 == "自动续跑"
        ckpt_params = {
            "prompts": seg_prompts, "width": width, "height": height,
            "length": length, "ctx": ctx, "steps": int(步数), "cfg": float(CFG),
            "sampler": 采样器, "scheduler": 调度器, "chain": chain,
            "keyframe_audio": KEYFRAME_AUDIO_SUPPORTED,
            "gate": {"mode": 桥帧门控, "threshold": float(清晰度阈值), "limit": gate_limit},
        }
        root, manifest, done = None, None, 0
        if resume:
            root = checkpoint.ckpt_dir(ckpt_params, 断点目录.strip())
            manifest = checkpoint.load_manifest(root)
            if manifest is not None:
                if manifest.get("schema") != checkpoint.SCHEMA:
                    raise ValueError(f"断点目录格式不认识（{manifest.get('schema')}），请换一个目录名")
                checkpoint.assert_match(manifest["params"], ckpt_params)
                done = checkpoint.contiguous_done(root, int(manifest.get("done", 0)))
                saved_seed = int(manifest["params"].get("seed", seed))
                if saved_seed != seed:
                    report.append(f"断点续拍：控件种子 {seed} 被忽略，沿用断点种子 {saved_seed}"
                                  "（control_after_generate 自增所致，属正常）")
                    seed = saved_seed
                report.append(f"断点续跑：载入已完成 {done}/{len(seg_prompts)} 段，目录 {root}")

        pbar = comfy.utils.ProgressBar(len(seg_prompts))
        all_frames = []
        all_wav = None
        seg_frames = []
        seg_wavs = []
        trims = []
        guide = None

        for i, prompt in enumerate(seg_prompts):
            replay = resume and i < done
            if replay:
                video_t, audio_t = checkpoint.load_segment(root, i)
                video_t = video_t.to(video_vae.device)
                audio_t = audio_t.to(audio_vae.device)
            else:
                seg_len = length + (ctx if i > 0 else 0)
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

                sampled = nodes.common_ksampler(
                    模型, (seed + i) % 0xffffffffffffffff, 步数, CFG,
                    采样器, 调度器, cond, negative, latent, denoise=1.0)[0]
                video_t, audio_t = sampled["samples"].unbind()
                if resume:
                    checkpoint.save_segment(root, i, video_t, audio_t)

            sampled_fc = latent_t_to_frames(video_t.shape[2])
            frames = video_vae.decode(video_t)
            if len(frames.shape) == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            wav, sample_rate = _decode_audio(audio_vae, audio_t)

            skip_f = ctx if i > 0 else 0
            vis_len = length
            if 桥帧门控 != "关闭" and i + 1 < len(seg_prompts):
                window = frames[max(skip_f, frames.shape[0] - (ctx + gate_limit)):]
                score = qc.frame_scores(window)
                tail_score = float(score[-1])
                if 桥帧门控 == "标注":
                    flag = " ↓ 低于阈值" if tail_score < 清晰度阈值 else ""
                    report.append(f"段{i + 1} 桥帧总分 {tail_score:.1f}{flag}")
                else:  # 自动回退
                    back, hit = qc.pick_backtrack(score, gate_limit, float(清晰度阈值))
                    if back:
                        vis_len = length - back
                        report.append(f"段{i + 1} 尾帧低质（{tail_score:.1f} < {清晰度阈值:g}），"
                                      f"回退 {back} 帧续拍（回退点 {hit:.1f}）")
            trims.append(length - vis_len)

            frames = frames[skip_f:skip_f + vis_len]
            total = wav.shape[-1]
            skip_s = round(total * skip_f / sampled_fc)
            take_s = round(total * vis_len / sampled_fc)
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
            origin = "断点载入" if replay else f"采样{sampled_fc}帧"
            report.append(f"段{i + 1}/{len(seg_prompts)}：{origin} 裁头{skip_f}帧 留{frames.shape[0]}帧 | {note}")

            if i + 1 < len(seg_prompts):
                back = length - vis_len
                guide = _tail_keyframe(video_t, audio_t, ctx, KEYFRAME_AUDIO_SUPPORTED,
                                       back_tokens=back // 17 * 5,
                                       back_audio=round(back * 5.0 / 3.0))

            if resume and not replay:
                checkpoint.save_manifest(root, {
                    "schema": checkpoint.SCHEMA, "done": i + 1,
                    "params": {**ckpt_params, "seed": seed}, "trims": list(trims),
                })

            pbar.update(1)

        if resume:
            report.append(f"断点目录（含中间 latent，跑完可删）：{root}")

        images = torch.cat(all_frames, dim=0)
        return io.NodeOutput(
            images,
            {"waveform": all_wav, "sample_rate": sample_rate},
            24,
            "\n".join(report),
            seg_frames,
            seg_wavs,
        )

    @staticmethod
    def _apply_guide(cond, guide, sampled_fc):
        """把上段尾帧 keyframe 注入 conditioning（官方 minimax_keyframes 协议）。"""
        return node_helpers.conditioning_set_values(cond, {
            "minimax_keyframes": [guide],
            "minimax_frame_count": sampled_fc,
        })
