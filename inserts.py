"""插入段：`位置|文件名` 清单解析、统一段落计划编排与上传视频解码。

插入段是上传到 ComfyUI input 目录的现成视频，占据全局段序列的一个槽位
（位置 1 = 片头/序章），像旧序章一样经一次 VAE 重编码进断点，其后生成段
从它的尾部续拍。槽位哈希（文件名+帧数+首帧统计）复用 manifest 的
prompt_hashes 逐段比对机制：换文件 / 增删插入段 -> 其后所有段自动重做
（每段锚定上一段尾帧，锚点变了必然级联重做）。

解析与编排为纯逻辑（无 ComfyUI / PyAV 依赖，可离线单测）；视频解码延迟
导入 PyAV（ComfyUI 新视频栈自带，与分段落盘同依赖），缺失时清晰报错。
"""

import hashlib


def parse_inserts(spec: str, prompt_count: int):
    """「位置|文件名」每行一条 -> 按位置升序的 [(pos, file), ...]。

    位置为 1-based 全局段号 = 插入到第几段**之前**（1 = 片头；尾插合法）。
    允许空行与 # 注释行。位置重复 / 非正整数 / 超出可插入范围直接报错：
    第 i 个插入（按升序）之前最多已有 prompt_count 个提示词段 + i 个插入段，
    故其位置上限为 prompt_count + i + 1。
    """
    items = []
    seen = set()
    for raw in str(spec or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("|", 1)
        if len(parts) != 2 or not parts[1].strip():
            raise ValueError(f"插入视频格式错误：{raw.strip()!r}（每行应为 位置|文件名，如 2|ad.mp4）")
        try:
            pos = int(parts[0].strip())
        except ValueError:
            raise ValueError(f"插入视频位置必须是整数：{raw.strip()!r}")
        if pos < 1:
            raise ValueError(f"插入视频位置必须 ≥ 1：{raw.strip()!r}")
        if pos in seen:
            raise ValueError(f"插入视频位置重复：{pos}（每个位置只能有一个插入段）")
        seen.add(pos)
        items.append((pos, parts[1].strip()))
    items.sort()
    for i, (pos, _file) in enumerate(items):
        if pos > prompt_count + i + 1:
            raise ValueError(
                f"插入视频位置 {pos} 超出范围：当前 {prompt_count} 段提示词 + {len(items)} 个插入段，"
                f"位置最大只能到 {prompt_count + len(items)}（尾插）")
    return items


def build_plan(seg_prompts, inserts):
    """提示词 + 插入段 -> 统一段落计划（全局槽位顺序，len(plan) = 段总数）。

    元素：{"kind": "prompt", "text", "i"}（i = 该提示词在 seg_prompts 中的
    下标，种子规则「第 i 段用 种子+i」沿用它）或 {"kind": "insert", "file", "pos"}。
    inserts 需已升序（parse_inserts 的返回值）。
    """
    plan = []
    pi = 0
    for pos, file in sorted(inserts):
        while len(plan) + 1 < pos:
            plan.append({"kind": "prompt", "text": seg_prompts[pi], "i": pi})
            pi += 1
        plan.append({"kind": "insert", "file": file, "pos": pos})
    while pi < len(seg_prompts):
        plan.append({"kind": "prompt", "text": seg_prompts[pi], "i": pi})
        pi += 1
    return plan


def insert_hash(filename: str, frame_count: int, f0_mean: float, f0_std: float) -> str:
    """插入段身份哈希（与 checkpoint.prompt_hash 同法的 8 位十六进制）。

    输入为解码原始帧（对齐/裁剪前）的统计：同文件同长度同首帧 -> 同哈希；
    换文件 / 重新上传同内容 -> 哈希变化触发其后段落重做。
    """
    blob = f"insert:{filename}:{frame_count}:{f0_mean:.4f}:{f0_std:.4f}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:8]


def load_video_av(filename: str, max_frames: int = 0):
    """input 目录中的视频 -> (frames[F,H,W,3] float 0..1 CPU, audio dict | None)。

    与画布「起始视频」输入同格式：帧按解码序直接取用（24fps 约定与序章一致），
    max_frames 提前停读防长视频爆内存（调用方传「每段帧数」，超长只取前段）。
    音频重采样为 fltp 双声道 -> {"waveform": [1,2,N], "sample_rate"}；
    无音轨返回 None（调用方按静音处理）。
    """
    import av
    import numpy as np
    import torch
    from folder_paths import get_annotated_filepath

    path = get_annotated_filepath(filename)
    with av.open(path) as container:
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise ValueError(f"插入视频没有视频流：{filename}")
        astream = next((s for s in container.streams if s.type == "audio"), None)
        resampler = av.AudioResampler(format="fltp", layout="stereo") if astream else None
        frames, chunks = [], []
        for packet in container.demux(*[s for s in (vstream, astream) if s is not None]):
            for frame in packet.decode():
                if packet.stream is vstream:
                    if max_frames and len(frames) >= max_frames:
                        continue
                    frames.append(frame.to_ndarray(format="rgb24"))
                elif resampler is not None:
                    out = resampler.resample(frame)
                    if out is None:
                        continue
                    if isinstance(out, av.AudioFrame):
                        out = [out]
                    chunks.extend(f.to_ndarray() for f in out)
        if resampler is not None:
            out = resampler.resample(None)
            if out and not isinstance(out, av.AudioFrame):
                chunks.extend(f.to_ndarray() for f in out)
        if not frames:
            raise ValueError(f"插入视频没有解码出任何帧：{filename}")
        frames_t = torch.from_numpy(np.stack(frames).astype("float32") / 255.0)
        audio = None
        if chunks:
            wav = np.concatenate(chunks, axis=1)
            audio = {"waveform": torch.from_numpy(wav.copy()),
                     "sample_rate": int(astream.rate or 44100)}
    return frames_t, audio