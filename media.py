"""PyAV 编解码共享：mp4 落盘与上传视频解码。

分镜段视频（checkpoint.save_segment_mp4）、成片保存（saver.H3ChainSaver）、
控制台序章上传（console.H3ChainConsole）三处共用，编码逻辑只维护一份。
PyAV 是 ComfyUI 新视频栈（CreateVideo/SaveVideo）的既有依赖，无新依赖。
"""

import os


def save_av_mp4(path, frames, wav, sample_rate, fps=24, crf=20):
    """可见帧 + 音轨 -> mp4（H.264 + AAC），成功返回 True。

    先写 .part 再原子改名：失败不留半成品、不破坏已有文件。编码失败
    （或 PyAV 缺失）返回 False，调用方降级（缩略图 / 空串），不影响主流程。
    """
    try:
        import av
        import numpy
    except Exception:
        return False
    tmp = path + ".part"
    try:
        arr = (frames.detach().float().clamp(0.0, 1.0).cpu().numpy() * 255.0).astype("uint8")
        h, w = arr.shape[1], arr.shape[2]
        h, w = h - h % 2, w - w % 2  # yuv420p 要求偶数尺寸
        arr = numpy.ascontiguousarray(arr[:, :h, :w, :3])
        pcm = numpy.ascontiguousarray(wav.detach().float().cpu().clamp(-1.0, 1.0).numpy())
        ch = int(pcm.shape[0])
        layout = "stereo" if ch >= 2 else "mono"

        container = av.open(tmp, mode="w")
        try:
            vstream = container.add_stream("libx264", rate=fps)
            vstream.width, vstream.height = w, h
            vstream.pix_fmt = "yuv420p"
            vstream.options = {"crf": str(int(crf)), "preset": "veryfast"}
            astream = container.add_stream("aac", rate=int(sample_rate), layout=layout)

            for i in range(arr.shape[0]):
                for packet in vstream.encode(av.VideoFrame.from_ndarray(arr[i], format="rgb24")):
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
        return True
    except Exception:
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
    RuntimeError——控制台序章路径失败应明确报错而非静默降级。
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
