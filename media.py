"""PyAV 编解码共享：mp4 落盘与上传视频解码。

分镜段视频（checkpoint.save_segment_mp4）、成片保存（saver.H3ChainSaver）、
采样器自动保存（nodes 自动成片）三处共用，编码逻辑只维护一份。
PyAV 是 ComfyUI 新视频栈（CreateVideo/SaveVideo）的既有依赖，无新依赖。
"""

import os

last_error = None


def save_av_mp4(path, frames, wav, sample_rate, fps=24, crf=20, threads=4):
    """可见帧 + 音轨 -> mp4（H.264 + AAC），成功返回 True。

    先写 .part 再原子改名：失败不留半成品、不破坏已有文件。编码失败
    （或 PyAV 缺失）返回 False，调用方降级（缩略图 / 空串），不影响主流程。
    失败原因存入 media.last_error 供调用方读取（报告/日志）。

    内存与 CPU 约束：分块（32 帧）搬运到 CPU，峰值内存 ~160MB 而非整链
    float32（长链数 GB，曾致内存耗尽假死）；threads 限 4（x264 默认
    线程=核数×1.5，会打满 CPU 导致整机卡顿）。
    """
    global last_error
    try:
        import av
        import numpy
    except Exception as e:
        last_error = f"PyAV/numpy 导入失败：{type(e).__name__}: {e}"
        return False
    tmp = path + ".part"
    try:
        n = int(frames.shape[0])
        h, w = int(frames.shape[1]), int(frames.shape[2])
        h, w = h - h % 2, w - h % 2  # yuv420p 要求偶数尺寸
        pcm = numpy.ascontiguousarray(wav.detach().float().cpu().clamp(-1.0, 1.0).numpy())
        if pcm.ndim == 3:                    # [batch, ch, samples] → [ch, samples]
            pcm = pcm[0]
        ch = int(pcm.shape[0])
        layout = "stereo" if ch >= 2 else "mono"

        # tmp 扩展名是 .part，PyAV 按扩展名猜不出封装格式会直接抛
        # ValueError: Could not determine output format —— 必须显式指定
        container = av.open(tmp, mode="w", format="mp4")
        try:
            vstream = container.add_stream("libx264", rate=fps)
            vstream.width, vstream.height = w, h
            vstream.pix_fmt = "yuv420p"
            vstream.options = {"crf": str(int(crf)), "preset": "veryfast",
                               "threads": str(max(1, int(threads)))}
            astream = container.add_stream("aac", rate=int(sample_rate), layout=layout)

            for start in range(0, n, 32):
                block = (frames[start:start + 32].detach().float().clamp(0.0, 1.0)
                         .cpu().numpy()[:, :h, :w, :3] * 255.0).astype("uint8")
                for i in range(block.shape[0]):
                    frame = av.VideoFrame.from_ndarray(
                        numpy.ascontiguousarray(block[i]), format="rgb24")
                    for packet in vstream.encode(frame):
                        container.mux(packet)
            if ch > 2:
                pcm = pcm[:2]
            for start in range(0, pcm.shape[1], 1024):
                aframe = av.AudioFrame.from_ndarray(
                    pcm[:, start:start + 1024], format="fltp", layout=layout)
                aframe.sample_rate = int(sample_rate)
                for packet in astream.encode(aframe):
                    container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)
        finally:
            container.close()
        os.replace(tmp, path)
        last_error = None
        return True
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        print(f"[save_av_mp4] 编码异常：{last_error}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False


def decode_av(path):
    """上传视频 -> (帧 tensor[N,H,W,3] float 0-1, 波形[C,T] float32, 采样率 int|None)。

    不做音频重采样：采样器 _encode_audio_latent 会按音频 VAE 采样率重采样。
    无音频轨时返回 (帧, zeros(1,0), None)。文件无视频轨或 PyAV 缺失抛
    RuntimeError——序章上传路径失败应明确报错而非静默降级。
    """
    try:
        import av
        import numpy
        import torch
    except Exception as e:
        raise RuntimeError("解码上传视频需要 PyAV（ComfyUI 新视频栈自带）；缺失请安装 av 后重试") from e
    with av.open(path) as container:
        vstream = next((s for s in container.streams if s.type == "video"), None)
        if vstream is None:
            raise RuntimeError("上传的文件里没有视频轨")
        frames = [f.to_ndarray(format="rgb24") for f in container.decode(vstream)]
        if not frames:
            raise RuntimeError("上传的视频里没有可解码的帧")
        astream = next((s for s in container.streams if s.type == "audio"), None)
        chunks, sample_rate = [], None
        if astream is not None:
            for f in container.decode(astream):
                arr = f.to_ndarray()  # fltp -> [C, T] float32
                if arr.ndim == 2 and arr.shape[0]:
                    chunks.append(arr)
                    sample_rate = f.sample_rate
    video = torch.from_numpy(numpy.stack(frames)).float() / 255.0
    if chunks:
        ch = chunks[0].shape[0]
        wav = torch.from_numpy(numpy.concatenate(
            [c[:ch] for c in chunks if c.shape[0] >= ch], axis=1))
    else:
        wav = torch.zeros(1, 0)
    return video, wav, sample_rate


def concat_av_mp4(sources, out_path, width=None, height=None, fps=24, crf=20, threads=4):
    """多个 mp4 按序流式拼接为一个（H.264 + AAC），成功返回 True。

    合并导出专用：逐源 demux→decode→encode，视频帧不进 Python 侧累积
    （1080p 全帧进内存会到 GB 级），内存占用 ≈ 单帧。
    - 画幅统一 width×height（缺省取第一个源）：源尺寸不同时 reformat 缩放
      （项目段 mp4 同画幅时为直通，零损）
    - 帧率统一 fps：帧按解码顺序写入（源非 24fps 时线性重定时，建议素材
      上传前转 24fps；H3 全链输出本身即 24fps）
    - 音频以第一个含音轨源的参数为基准，源间 AudioResampler 统一采样率/声道；
      无音轨源按其视频时长补静音
    先写 .part 再原子改名；失败原因存 media.last_error。
    """
    global last_error
    try:
        import av
        import numpy
    except Exception as e:
        last_error = f"PyAV/numpy 导入失败：{type(e).__name__}: {e}"
        return False
    if not sources:
        last_error = "合并清单为空"
        return False
    tmp = out_path + ".part"
    try:
        base_w = base_h = None
        base_rate, base_layout = None, None
        for p in sources:
            if not os.path.exists(p):
                raise RuntimeError(f"合并源不存在：{p}")
            with av.open(p) as probe:
                vs = next((s for s in probe.streams if s.type == "video"), None)
                if vs is None:
                    raise RuntimeError(f"{os.path.basename(p)} 没有视频轨，无法合并")
                if base_w is None:
                    base_w = int(vs.codec_context.width or 0)
                    base_h = int(vs.codec_context.height or 0)
                if base_rate is None:
                    aprobe = next((s for s in probe.streams if s.type == "audio"), None)
                    if aprobe is not None and aprobe.codec_context.sample_rate:
                        base_rate = int(aprobe.codec_context.sample_rate)
                        base_layout = "stereo" if (aprobe.codec_context.channels or 1) >= 2 else "mono"
        if base_rate is None:
            base_rate, base_layout = 44100, "stereo"
        if not base_w or not base_h:
            raise RuntimeError("无法探测源视频画幅")
        w, h = int(width or base_w), int(height or base_h)
        w, h = w - w % 2, h - h % 2   # yuv420p 要求偶数尺寸
        out_ch = 2 if base_layout == "stereo" else 1

        container = av.open(tmp, mode="w", format="mp4")
        try:
            vstream = container.add_stream("libx264", rate=fps)
            vstream.width, vstream.height = w, h
            vstream.pix_fmt = "yuv420p"
            vstream.options = {"crf": str(int(crf)), "preset": "veryfast",
                               "threads": str(max(1, int(threads)))}
            astream = container.add_stream("aac", rate=base_rate, layout=base_layout)
            for p in sources:
                with av.open(p) as src:
                    vs = next((s for s in src.streams if s.type == "video"), None)
                    as_in = next((s for s in src.streams if s.type == "audio"), None)
                    n_frames = 0
                    for vf in src.decode(vs):
                        if vf.width != w or vf.height != h or str(vf.format.name) != "yuv420p":
                            vf = vf.reformat(width=w, height=h, format="yuv420p")
                        vf.pts = None   # 丢弃源时间戳：按输出 fps 线性重定时
                        n_frames += 1
                        for packet in vstream.encode(vf):
                            container.mux(packet)
                    pcm = None
                    if as_in is not None:
                        rs = av.AudioResampler(format="fltp", layout=base_layout, rate=base_rate)
                        parts = []
                        for af in src.decode(as_in):
                            rf = rs.resample(af)
                            for f in (rf if isinstance(rf, list) else [rf]):
                                arr = f.to_ndarray()   # fltp -> [C, T] float32
                                if arr.ndim == 2 and arr.shape[0]:
                                    parts.append(arr[:out_ch])
                        if parts:
                            pcm = numpy.concatenate(parts, axis=1)
                    if pcm is None:
                        silent_n = int(round(n_frames / fps * base_rate))
                        pcm = numpy.zeros((out_ch, max(0, silent_n)), dtype="float32")
                    for start in range(0, pcm.shape[1], 1024):
                        chunk = numpy.ascontiguousarray(pcm[:, start:start + 1024])
                        af_out = av.AudioFrame.from_ndarray(chunk, format="fltp", layout=base_layout)
                        af_out.sample_rate = base_rate
                        for packet in astream.encode(af_out):
                            container.mux(packet)
            for packet in vstream.encode():
                container.mux(packet)
            for packet in astream.encode():
                container.mux(packet)
        finally:
            container.close()
        os.replace(tmp, out_path)
        last_error = None
        return True
    except Exception as e:
        last_error = f"{type(e).__name__}: {e}"
        print(f"[concat_av_mp4] 合并异常：{last_error}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return False
