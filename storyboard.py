"""H3 Storyboard Chain —— 分镜模式：关键帧先行 + 首尾双向锚定（即梦「智能多帧」本地版）。

范式对比（与续拍 H3SeamlessChainSampler 互补）：
- 续拍 = 边生成边定锚（上段尾 latent 直切钉下段头）：接缝最顺，但误差沿链
  传播（复印机效应），长链越跑越糊
- 分镜 = 先定锚后生成（N 张关键帧图，段 i 从关键帧 i 画到关键帧 i+1）：
  误差不传播、任意段独立重做，段间是「镜头切换」语义（转场，非无缝续接）

双向锚定实现：首帧走官方 i2v（first_frame=关键帧 i，质量最好的训练路径）；
尾帧自注入官方 Add Guide keyframe 协议——关键帧 i+1 堆叠 5 帧编码为 2 token
latent，钉 resolved_frame_index=段尾-5（天然踩 17k+5 网格的 token 边界）。
r2v 链路（带参考素材）官方无 first_frame 接口，首尾两锚均自注入。

关键帧来源两种：「上传」分镜图组，或「从断点导入」——读续拍链断点目录各段
latent 解码出 N+1 张关键帧（首段首帧 + 各段尾帧），实现「续拍粗剪 → 抽帧 →
分镜精修」两遍法。

断点：复用 v2 schema。段哈希 = 提示词 + 首尾关键帧哈希的组合——改关键帧 i
自动失效段 i-1（尾锚变了）与段 i（首帧变了），改提示词只失效该段。
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
from .grid import latent_t_to_frames, align_frame_count_down
from .nodes import (_decode_audio, _center_cover, _autogrow_items,
                    _apply_anchor_noise, cond_audio_rows_guard)


def storyboard_plan(kf_count, prompts):
    """N 张关键帧 + M 条段提示词 -> 段计划 [(首kf下标, 尾kf下标, prompt)]。

    N 张图定义 N-1 段（图 i 与图 i+1 之间为段 i）；数量不符抛 ValueError。
    纯数学，stub 环境可单测。
    """
    if kf_count < 2:
        raise ValueError(f"分镜模式至少需要 2 张关键帧图（当前 {kf_count} 张）："
                         "N 张图定义 N-1 段，段 i 从关键帧 i 画到关键帧 i+1")
    segs = kf_count - 1
    if len(prompts) != segs:
        raise ValueError(f"段提示词数量不匹配：{kf_count} 张关键帧应对应 {segs} 条段提示词，"
                         f"当前 {len(prompts)} 条（每段提示词描述两关键帧之间的过渡与运镜）")
    return [(i, i + 1, p) for i, p in enumerate(prompts)]


def tail_anchor_index(sampled_fc):
    """采样帧数 -> 尾锚 resolved_frame_index（对齐 17k+5 网格，覆盖最后 5 帧=2 token）。"""
    return align_frame_count_down(int(sampled_fc)) - 5


def frame_keyframe(image, video_vae, width, height, frame_index):
    """单张关键帧图 -> Add Guide keyframe dict：cover 对齐分辨率后堆 5 帧编码（t=2 token）。"""
    img = _center_cover(image[None], width, height)[0]
    latent = video_vae.encode(torch.stack([img] * 5))
    return {"resolved_frame_index": int(frame_index), "latent": latent}


def keyframes_from_ckpt(import_dir, video_vae):
    """续拍链断点目录 -> 关键帧图列表：kf[0]=段0首帧，kf[i+1]=段i尾帧。

    按 seg_NNN.pt 连续存在读最大前缀（中断段无尾帧，不导入）；
    import_dir 支持绝对路径或相对 output/checkpoints/ 的目录名。
    """
    root = import_dir if os.path.isabs(import_dir) else os.path.join(
        checkpoint.checkpoints_root(), import_dir)
    if not os.path.isdir(root):
        raise ValueError(f"断点导入目录不存在：{root}（相对路径以 output/checkpoints/ 为基准）")
    last = -1
    while os.path.exists(checkpoint.seg_path(root, last + 1)):
        last += 1
    if last < 0:
        raise ValueError(f"断点导入目录中没有段文件（seg_NNN.pt）：{root}")
    tails = []
    head = None
    for i in range(last + 1):
        v, _ = checkpoint.load_segment(root, i)
        frames = video_vae.decode(v.to(video_vae.device))
        if frames.dim() == 5:
            frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
        frames = frames.cpu()
        if head is None:
            head = frames[0].clone()
        tails.append(frames[-1].clone())
    return [head] + tails


class H3StoryboardChain(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3StoryboardChain",
            display_name="H3 Storyboard Chain (分镜长片)",
            category="MiniMaxH3",
            description="MiniMax H3 分镜模式：N 张关键帧图把长片切成 N-1 段，每段首尾双向锚定"
                        "（首帧走官方 i2v，尾帧自注入 Add Guide 协议），误差不沿链传播，段间为镜头切换。"
                        "关键帧可上传，也可从续拍链断点导入（两遍法精修）。",
            inputs=[
                io.Model.Input("模型"),
                io.Clip.Input("文本编码器"),
                io.Vae.Input("视频VAE"),
                io.Vae.Input("音频VAE"),
                io.Int.Input("宽度", default=864, min=32, max=16384, step=32),
                io.Int.Input("高度", default=480, min=32, max=16384, step=32),
                io.Int.Input("每段帧数", default=124, min=22, max=3600,
                             tooltip="每段帧数 @24fps（建议 124≈5秒 或 141≈6秒，落在 17k+5 网格上尾锚恰好钉在段尾）。"
                                     "非网格值会被模型向上对齐（如 130→141 帧）"),
                io.Int.Input("种子", default=0, min=0, max=0xffffffffffffffff, control_after_generate=True,
                             tooltip="第 i 段实际使用 种子+i"),
                io.Int.Input("步数", default=25, min=1, max=100),
                io.Float.Input("CFG", default=1.0, min=0.0, max=100.0, step=0.1),
                io.Combo.Input("采样器", options=comfy.samplers.KSampler.SAMPLERS, default="res_multistep"),
                io.Combo.Input("调度器", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                io.Combo.Input("尾帧锚定", options=["开启", "关闭"], default="开启",
                               tooltip="每段尾帧也钉一张关键帧（首尾双向锚定，即梦智能多帧同款）。"
                                       "关闭则退化为关键帧首帧链：每段只锚首帧、尾帧自由发挥，"
                                       "误差同样不传播但段尾不受控（段间为纯转场）"),
                io.Float.Input("锚定加噪", default=0.0, min=0.0, max=0.5, step=0.05,
                               tooltip="对锚定帧注入噪声比例（SkyReels-V2 addnoise_condition 思路）："
                                       "0=关闭；0.1 微调；0.2 标准；0.3+ 干预强但细节变软。"
                                       "不进断点指纹，改参数不触发重跑"),
                io.Combo.Input("响度对齐", options=["开启", "关闭"], default="开启",
                               tooltip="段首增益匹配上段尾 RMS（±6dB 钳制，1 秒内渐出）：分镜段间画面是镜头切换，"
                                       "声音响度跳变仍会突兀，对齐只作用于段首窗口、不沿链累积"),
                io.Combo.Input("断点续拍", options=["关闭", "自动续跑"], default="关闭",
                               tooltip="自动续跑：每段采样后立即落盘 latent 断点，中断后重跑自动跳过已完成段"),
                io.String.Input("断点目录", default="",
                                tooltip="空=按参数指纹自动命名于 output/checkpoints/ 下；换链请填新名字或清空旧目录"),
                io.Combo.Input("审片模式", options=["关闭", "逐段确认"], default="关闭",
                               tooltip="逐段确认：每次运行只生成一个新段落即返回，预览后重新运行继续；"
                                       "不满意可改该段提示词（自动从该段重跑）。开启后断点自动启用"),
                io.Int.Input("重跑起始段", default=0, min=0, max=63,
                             tooltip="0=自动（沿用断点进度，改过提示词/关键帧的段自动重做）；N=从第 N 段起重新生成，"
                                     "配合改「种子」即可重摇。用完记得改回 0"),
                io.Combo.Input("关键帧来源", options=["上传", "从断点导入"], default="上传",
                               tooltip="上传=「分镜图组」逐张接图（LoadImage）；从断点导入=读续拍链断点目录各段 latent "
                                       "解码出关键帧（首段首帧+各段尾帧），实现续拍粗剪→分镜精修两遍法"),
                io.String.Input("断点导入目录", default="",
                                tooltip="关键帧来源=从断点导入时生效：续拍链断点目录名（相对 output/checkpoints/）或绝对路径"),
                io.Autogrow.Input("分镜图组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("分镜图",
                                                           tooltip="关键帧图，按序号即播放顺序。N 张图定义 N-1 段；"
                                                                   "分辨率不一致会自动 cover 对齐（建议同尺寸）"),
                                      prefix="分镜图_", min=2, max=64)),
                io.Autogrow.Input("提示词组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.String.Input("段提示词", multiline=True,
                                                            placeholder="这一段的画面描述：从关键帧 i 到关键帧 i+1 的过渡与运镜"),
                                      prefix="提示词_", min=1, max=64)),
                io.Autogrow.Input("参考图片组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("参考图片",
                                                           tooltip="参考主体图，提示词用 <Picture i> 引用。用了请接 ref2va UNET"),
                                      prefix="参考图片_", min=0, max=9)),
                io.Autogrow.Input("参考视频组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Image.Input("参考视频",
                                                           tooltip="参考视频帧（24fps，2-15 秒），提示词用 <Video k> 引用"),
                                      prefix="参考视频_", min=0, max=3)),
                io.Autogrow.Input("参考视频音轨组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.Audio.Input("参考视频音轨",
                                                           tooltip="同号参考视频的原声（参考视频音轨_1 配 参考视频_1）"),
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
                                tooltip="每段完整帧（逐段展开，接 Create Video 可单独保存）"),
                io.Audio.Output("分段音频", is_output_list=True,
                                tooltip="与分段图像逐段配对的音轨"),
            ],
        )

    @classmethod
    def execute(cls, 模型, 文本编码器, 视频VAE, 音频VAE, 宽度, 高度, 每段帧数,
                种子, 步数, CFG, 采样器, 调度器,
                尾帧锚定="开启", 锚定加噪=0.0, 响度对齐="开启",
                断点续拍="关闭", 断点目录="", 审片模式="关闭", 重跑起始段=0,
                关键帧来源="上传", 断点导入目录="",
                分镜图组=None, 提示词组=None,
                参考图片组=None, 参考视频组=None, 参考视频音轨组=None, 参考音频组=None):
        length, width, height, seed = int(每段帧数), int(宽度), int(高度), int(种子)
        clip, video_vae, audio_vae = 文本编码器, 视频VAE, 音频VAE
        tail_locked = 尾帧锚定 == "开启"
        from . import qc  # 接缝测量 + 响度对齐（延迟导入：无 ComfyUI 环境下结构单测不触达）

        if 关键帧来源 == "从断点导入":
            src = 断点导入目录.strip()
            if not src:
                raise ValueError("关键帧来源=从断点导入 时，「断点导入目录」不能为空："
                                 "填续拍链断点目录名（相对 output/checkpoints/）或绝对路径")
            kf_imgs = keyframes_from_ckpt(src, video_vae)
            base = os.path.basename(src.rstrip("/\\"))
            kf_origin = f"从断点导入（{base}，{len(kf_imgs) - 1} 段）"
        else:
            kf_imgs = [v for v in _autogrow_items(分镜图组, "kf").values() if v is not None]
            kf_imgs = [img[0] if img.dim() == 4 else img for img in kf_imgs]
            kf_origin = "上传"
        seg_prompts = [str(v).strip() for v in _autogrow_items(提示词组, "p").values() if str(v).strip()]
        plan = storyboard_plan(len(kf_imgs), seg_prompts)

        refs = {
            "ref_images": _autogrow_items(参考图片组, "ref_image_"),
            "ref_videos": _autogrow_items(参考视频组, "ref_video_"),
            "ref_video_audios": _autogrow_items(参考视频音轨组, "ref_video_audio_"),
            "ref_audios": _autogrow_items(参考音频组, "ref_audio_"),
        }
        has_refs = any(refs.values())
        chain = "分镜 r2v（ref2va UNET）" if has_refs else "分镜 i2v（fl2va UNET）"

        negative = clip.encode_from_tokens_scheduled(clip.tokenize(""))
        aug = min(max(float(锚定加噪), 0.0), 0.5)
        report = [f"H3 Storyboard Chain：{len(plan)} 段（{len(kf_imgs)} 张关键帧，{kf_origin}），"
                  f"链路 {chain}，尾帧锚定 {'开启' if tail_locked else '关闭'}"]
        if has_refs:
            counts = " ".join(f"{k}={len(v)}" for k, v in refs.items() if v)
            report.append(f"参考素材：{counts}")
        if aug > 0.0:
            report.append(f"锚定加噪 {aug:.2f}：锚定帧按参考而非逐帧复现注入（视觉 {1.0 - aug:.2f} / 音频 {1.0 - aug * 0.5:.2f} 保真）")

        # 断点：段哈希 = 提示词+首尾关键帧哈希组合（改关键帧 i -> 段 i-1 与段 i 失效）
        resume = 断点续拍 == "自动续跑"
        review = 审片模式 == "逐段确认"
        reroll = max(0, int(重跑起始段))
        use_ckpt = resume or review
        if review and not resume:
            report.append("审片模式：断点自动启用（每段落盘 latent，跨次运行续接）")
        ckpt_params = {
            "width": width, "height": height,
            "length": length, "steps": int(步数), "cfg": float(CFG),
            "sampler": 采样器, "scheduler": 调度器, "chain": chain,
            "mode": "storyboard", "tail": 尾帧锚定,
        }
        kf_hashes = [checkpoint.image_hash(img) for img in kf_imgs]
        seg_hashes = [cls._seg_hash(p, kf_hashes[hi], kf_hashes[ti], tail_locked)
                      for hi, ti, p in plan]
        root, manifest, done, seeds = None, None, 0, []
        if use_ckpt:
            root = checkpoint.ckpt_dir(ckpt_params, 断点目录.strip())
            manifest = checkpoint.load_manifest(root)
            if manifest is not None:
                if manifest.get("schema") != checkpoint.SCHEMA:
                    raise ValueError(f"断点目录格式不认识（{manifest.get('schema')}），请换一个目录名")
                checkpoint.assert_match(manifest["params"], ckpt_params)
                done = checkpoint.contiguous_done(root, int(manifest.get("done", 0)))
                start = min(max(reroll - 1, 0), done) if reroll > 0 else min(
                    checkpoint.reroll_start(list(manifest.get("prompt_hashes", [])), seg_hashes, done), done)
                if start < done:
                    manifest = checkpoint.truncate(root, manifest, start)
                    done = start
                    原因 = "手动指定「重跑起始段」" if reroll > 0 else "检测到该段提示词或关键帧已修改"
                    report.append(f"断点续跑：{原因}，从段 {start + 1} 起重新生成"
                                  + (f"（段 1-{start} 沿用断点）" if start else "（整链重做）"))
                seeds = [int(s) for s in manifest.get("seeds", [])]
                report.append(f"断点续跑：载入已完成 {done}/{len(plan)} 段，目录 {root}")
            for j, img in enumerate(kf_imgs):  # 关键帧副本（面板时间线/断点自包含）
                checkpoint.save_keyframe(root, j, img)

        pbar = comfy.utils.ProgressBar(len(plan))
        total = len(plan)
        thumbs, videos, seams, anchors = [], [], [], []
        all_frames = []
        all_wav = None
        seg_frames = []
        seg_wavs = []
        prev_tail_frame = None
        prev_tail_wav = None
        sample_rate = None

        if use_ckpt:
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll, "report": "",
                                   "updated_at": time.time()})

        for i, (hi, ti, prompt) in enumerate(plan):
            replay = use_ckpt and i < done
            if replay:
                video_t, audio_t = checkpoint.load_segment(root, i)
                video_t = video_t.to(video_vae.device)
                audio_t = audio_t.to(audio_vae.device)
                dt = 0.0
                cur_seed = seeds[i] if i < len(seeds) else None
            else:
                if reroll > 0:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                elif len(seeds) > 0:
                    cur_seed = (seeds[-1] + i - len(seeds) + 1) % 0xffffffffffffffff
                else:
                    cur_seed = (seed + i) % 0xffffffffffffffff
                if has_refs:
                    out = MiniMaxH3ReferenceToVideo.execute(
                        clip=clip, vae=video_vae, audio_vae=audio_vae,
                        prompt=prompt, width=width, height=height, length=length,
                        ref_image_size="match", **refs)
                else:
                    out = MiniMaxH3ImageToVideo.execute(
                        clip=clip, vae=video_vae, prompt=prompt,
                        width=width, height=height, length=length,
                        first_frame=kf_imgs[hi])
                cond, latent = out[0], out[1]
                sampled_fc = latent_t_to_frames(latent["samples"].tensors[0].shape[2])
                # 双向锚定：r2v 无 first_frame 接口时首锚自注入；尾锚钉段尾前 5 帧
                kfs = [k for k in (cond[0][1].get("minimax_keyframes") or [])]
                if has_refs:
                    kfs.append(frame_keyframe(kf_imgs[hi], video_vae, width, height, 0))
                if tail_locked:
                    kfs.append(frame_keyframe(kf_imgs[ti], video_vae, width, height,
                                              tail_anchor_index(sampled_fc)))
                if kfs:
                    cond = node_helpers.conditioning_set_values(cond, {
                        "minimax_keyframes": kfs,
                        "minimax_frame_count": sampled_fc,
                    })
                if aug > 0.0:
                    cond = _apply_anchor_noise(cond, aug)
                restore_audio_rows = cond_audio_rows_guard(模型.model.diffusion_model)
                try:
                    t0 = time.perf_counter()
                    sampled = nodes.common_ksampler(
                        模型, cur_seed, 步数, CFG,
                        采样器, 调度器, cond, negative, latent, denoise=1.0)[0]
                finally:
                    restore_audio_rows()
                dt = time.perf_counter() - t0
                video_t, audio_t = sampled["samples"].unbind()
                if i < len(seeds):
                    seeds[i] = cur_seed
                else:
                    seeds.append(cur_seed)
                if use_ckpt:
                    checkpoint.save_segment(root, i, video_t, audio_t)

            sampled_fc = latent_t_to_frames(video_t.shape[2])
            frames = video_vae.decode(video_t)
            if frames.dim() == 5:
                frames = frames.reshape(-1, frames.shape[-3], frames.shape[-2], frames.shape[-1])
            wav, sample_rate = _decode_audio(audio_vae, audio_t)

            # 接缝测量（分镜段间为镜头切换，帧差是转场幅度参考而非缺陷指标）
            seam_n = int(sample_rate * 0.25)
            if i > 0 and prev_tail_frame is not None:
                d, db = qc.seam_metrics(prev_tail_frame, frames[0],
                                        prev_tail_wav, wav[..., :seam_n], rate=sample_rate)
                db_txt = f"{db:+.1f} dB" if db is not None else "—"
                seams.append([round(d, 4), None if db is None else round(db, 2)])
                if 响度对齐 == "开启":
                    wav, gain_db = qc.loudness_align_head(wav, prev_tail_wav, rate=sample_rate)
                    if gain_db is not None:
                        report.append(f"段{i + 1} 响度对齐：{gain_db:+.1f} dB（段首 1s 渐出）")
            else:
                seams.append(None)

            # 尾锚达成度：生成尾帧 vs 关键帧 i+1（衡量模型被尾锚约束的程度）
            if tail_locked:
                tail_ref = _center_cover(kf_imgs[ti][None], width, height)[0]
                anchor_d = float((frames[-1].cpu() - tail_ref.cpu()).abs().mean())
                anchors.append(round(anchor_d, 4))
                report.append(f"段{i + 1} 尾锚达成：帧差 {anchor_d:.3f}（生成尾帧 vs 关键帧{ti + 1}）")
            else:
                anchors.append(None)

            frames_cpu = frames.cpu()
            seg_wav = wav.cpu()
            all_frames.append(frames_cpu)
            seg_frames.append(frames_cpu)
            seg_wavs.append({"waveform": seg_wav, "sample_rate": sample_rate})
            all_wav = seg_wav if all_wav is None else torch.cat([all_wav, seg_wav], dim=-1)
            prev_tail_frame = frames_cpu[-1].clone()
            prev_tail_wav = seg_wav[..., -seam_n:].clone()
            if use_ckpt:
                thumbs.append(checkpoint.save_thumb(root, i, frames[0]))
                videos.append(checkpoint.save_segment_mp4(root, i, frames, wav, sample_rate,
                                                          fresh=not replay))
            else:
                thumbs.append("")
                videos.append("")

            anchor_txt = "首锚+尾锚" if tail_locked else "仅首锚"
            origin = "断点载入" if replay else f"采样{sampled_fc}帧"
            seed_txt = cur_seed if cur_seed is not None else "—"
            report.append(f"段{i + 1}/{total}：{origin} 留{frames_cpu.shape[0]}帧 · 种子 {seed_txt}"
                          + ("" if replay else f" · 采样 {dt:.0f}s") + f" | {anchor_txt}（kf{hi + 1}→kf{ti + 1}）")

            if use_ckpt and not replay:
                done = i + 1
                checkpoint.save_manifest(root, {
                    "schema": checkpoint.SCHEMA, "done": done, "has_prologue": False,
                    "seeds": list(seeds[:done]), "trims": [0] * done,
                    "prompt_hashes": seg_hashes[:done],
                    "kf_hashes": kf_hashes,
                    "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                    "prompts": list(seg_prompts[:done]),
                    "seams": seams[:done], "anchors": anchors[:done],
                    "params": ckpt_params,
                })

            pbar.update(1)

            if review and not replay and i + 1 < total:
                report.append(f"审片：段 {i + 1} 已完成并落盘 → 预览「分段图像」或左侧「长片审片」面板；"
                              f"满意请直接重新运行继续段 {i + 2}；不满意：改「提示词_{i + 1}」后运行（自动从本段重跑），"
                              f"或设「重跑起始段」={i + 1} 并改「种子」重摇")
                break

        if use_ckpt:
            checkpoint.save_manifest(root, {
                "schema": checkpoint.SCHEMA, "done": done, "has_prologue": False,
                "seeds": list(seeds[:done]), "trims": [0] * done,
                "prompt_hashes": seg_hashes[:done],
                "kf_hashes": kf_hashes,
                "total": total, "thumbs": list(thumbs[:done]), "videos": list(videos[:done]),
                "prompts": list(seg_prompts[:done]),
                "seams": seams[:done], "anchors": anchors[:done],
                "params": ckpt_params,
            })
            if review:
                if len(seg_frames) == total:
                    report.append("审片：本链已全部完成")
                if reroll > 0:
                    report.append(f"注意：「重跑起始段」={reroll} 已生效，确认无误后请改回 0")
            report.append(f"断点目录（含中间 latent，跑完可删）：{root}")
            checkpoint.save_state({"dir": os.path.basename(root), "total": total, "done": done,
                                   "review": bool(review), "reroll": reroll,
                                   "report": "\n".join(report), "updated_at": time.time()})

        images = torch.cat(all_frames, dim=0)
        anchored = [(g, a) for g, a in enumerate(anchors) if a is not None]
        measured = [(g, s[0]) for g, s in enumerate(seams) if s]
        line = f"链路完成：{total} 段"
        if anchored:
            avg_a = sum(a for _, a in anchored) / len(anchored)
            worst_a = max(anchored, key=lambda x: x[1])
            line += f" · 尾锚达成平均帧差 {avg_a:.3f}（最差 段{worst_a[0] + 1} {worst_a[1]:.3f}）"
        if measured:
            avg_d = sum(d for _, d in measured) / len(measured)
            worst_d = max(measured, key=lambda x: x[1])
            line += f" · 转场平均帧差 {avg_d:.3f}（最大 段{worst_d[0] + 1} {worst_d[1]:.3f}）"
        report.append(line)
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
            return float("nan")
        return ""

    @staticmethod
    def _seg_hash(prompt, head_hash, tail_hash, tail_locked):
        """段哈希 = 提示词 + 首关键帧 + 尾关键帧（尾锚关闭时尾帧不参与）。"""
        return checkpoint.prompt_hash(f"{prompt}|{head_hash}|{tail_hash if tail_locked else '-'}")
