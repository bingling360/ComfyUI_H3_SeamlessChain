"""合并导出单测：concat_av_mp4 流式拼接 + merge_project 清单解析/防穿越。

python tests/test_merge.py（需要 av；torch 不需要——源 mp4 用 PyAV 直接合成）。
无 av 环境自动 SKIP（同 test_media 约定），路由层错误映射在 test_routes.py。
"""
import contextlib
import json
import os
import shutil
import sys
import tempfile
import types

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.dirname(_HERE), os.path.dirname(os.path.dirname(_HERE)), _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import av
    import numpy
    _HAS = True
except Exception:
    _HAS = False


def _make_mp4(path, n_frames=12, w=64, h=48, fps=24, audio=True, sr=44100,
              tone=440):
    """合成测试源 mp4（H.264 [+AAC 正弦波]），帧亮度随序号变化便于对账。"""
    container = av.open(path, mode="w", format="mp4")
    try:
        vs = container.add_stream("libx264", rate=fps)
        vs.width, vs.height = w, h
        vs.pix_fmt = "yuv420p"
        vs.options = {"crf": "23", "preset": "veryfast", "threads": "2"}
        as_ = container.add_stream("aac", rate=sr, layout="stereo") if audio else None
        for i in range(n_frames):
            arr = numpy.full((h, w, 3), (i * 19) % 255, dtype="uint8")
            for pkt in vs.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
                container.mux(pkt)
        for pkt in vs.encode():
            container.mux(pkt)
        if as_ is not None:
            n_samples = int(n_frames / fps * sr)
            t = numpy.arange(n_samples, dtype="float32") / sr
            pcm = numpy.stack([
                (0.2 * numpy.sin(2 * numpy.pi * tone * t)).astype("float32"),
                (0.2 * numpy.sin(2 * numpy.pi * tone * 1.5 * t)).astype("float32")])
            for s in range(0, n_samples, 1024):
                af = av.AudioFrame.from_ndarray(
                    numpy.ascontiguousarray(pcm[:, s:s + 1024]),
                    format="fltp", layout="stereo")
                af.sample_rate = sr
                for pkt in as_.encode(af):
                    container.mux(pkt)
            for pkt in as_.encode():
                container.mux(pkt)
    finally:
        container.close()


def _probe(path):
    """-> (帧数, 画幅(w,h), 音频时长秒|None)。

    视频/音频各开一次容器：concat 输出的包交错很差（逐源先全部视频再全部
    音频），同一容器先解完视频再解音频时 demuxer 已到 EOF，音频帧读不出。
    """
    with av.open(path) as c:
        n = 0
        size = None
        for f in c.decode(video=0):
            n += 1
            size = (f.width, f.height)
    with av.open(path) as c:
        has_audio = any(s.type == "audio" for s in c.streams)
        a_dur = 0.0
        if has_audio:
            for f in c.decode(audio=0):
                a_dur += f.samples / float(f.sample_rate)
    return n, size, (a_dur if has_audio else None)


@contextlib.contextmanager
def _env():
    """临时 output + input 目录（folder_paths stub，两目录都提供）。"""
    out = tempfile.mkdtemp()
    inp = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(
        get_output_directory=lambda: out, get_input_directory=lambda: inp)
    try:
        yield out, inp
    finally:
        del sys.modules["folder_paths"]
        for k in [k for k in sys.modules if k.startswith("ComfyUI_H3_SeamlessChain")]:
            del sys.modules[k]
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(inp, ignore_errors=True)


def _mk_project(out, name, manifest):
    pdir = os.path.join(out, "h3_projects", name)
    os.makedirs(pdir, exist_ok=True)
    with open(os.path.join(pdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return pdir


def test_concat_two_segments():
    if not _HAS:
        print("SKIP (无 av)")
        return
    from ComfyUI_H3_SeamlessChain import media
    with tempfile.TemporaryDirectory() as d:
        a, b, o = (os.path.join(d, x) for x in ("a.mp4", "b.mp4", "out.mp4"))
        _make_mp4(a, n_frames=12)
        _make_mp4(b, n_frames=12, tone=660)
        assert media.concat_av_mp4([a, b], o) is True
        n, size, a_dur = _probe(o)
        assert n == 24, n                                    # 帧数守恒（12+12）
        assert size == (64, 48)                              # 画幅一致直通
        assert 0.9 <= a_dur <= 1.2, a_dur                    # 音轨 ~1.0s（AAC priming 容差）


def test_concat_hq_profile_preserves_luma():
    """HQ 拼接不能把已量化的 RGB 当 0..1 浮点再次抖动，否则整片会截成白色。"""
    if not _HAS:
        print("SKIP (无 av)")
        return
    from ComfyUI_H3_SeamlessChain import media
    with tempfile.TemporaryDirectory() as d:
        a, o = (os.path.join(d, x) for x in ("a.mp4", "out.mp4"))
        _make_mp4(a, n_frames=8, audio=False)
        assert media.concat_av_mp4([a], o, crf=16, preset="medium",
                                   aq_mode=3, dither=True) is True
        with av.open(o) as c:
            means = [float(f.to_ndarray(format="rgb24").mean())
                     for f in c.decode(video=0)]
        assert len(means) == 8
        assert max(means) < 180.0                              # 非全白/高光截断
        assert means[-1] - means[0] > 80.0                    # 原亮度阶梯仍存在


def test_concat_reformat_and_silence():
    if not _HAS:
        print("SKIP (无 av)")
        return
    from ComfyUI_H3_SeamlessChain import media
    with tempfile.TemporaryDirectory() as d:
        a, b, o = (os.path.join(d, x) for x in ("a.mp4", "b.mp4", "out.mp4"))
        _make_mp4(a, n_frames=12, w=64, h=48)                # 有音轨
        _make_mp4(b, n_frames=12, w=96, h=64, audio=False)   # 异画幅 + 无音轨
        assert media.concat_av_mp4([a, b], o, width=64, height=48) is True
        n, size, a_dur = _probe(o)
        assert n == 24
        assert size == (64, 48)                              # b 被 reformat 到 64x48
        # b 的 0.5s 按视频时长补静音：总音轨 ≈ 1.0s
        assert 0.9 <= a_dur <= 1.2, a_dur


def test_concat_all_silent_still_has_audio():
    """全部源无音轨：按基准 44100/stereo 补静音，输出仍有音轨（参数统一）。"""
    if not _HAS:
        print("SKIP (无 av)")
        return
    from ComfyUI_H3_SeamlessChain import media
    with tempfile.TemporaryDirectory() as d:
        a, b, o = (os.path.join(d, x) for x in ("a.mp4", "b.mp4", "out.mp4"))
        _make_mp4(a, n_frames=12, audio=False)
        _make_mp4(b, n_frames=12, audio=False)
        assert media.concat_av_mp4([a, b], o) is True
        n, size, a_dur = _probe(o)
        assert n == 24 and size == (64, 48)
        assert a_dur is not None and 0.9 <= a_dur <= 1.2     # 静音轨仍在


def test_concat_failures():
    if not _HAS:
        print("SKIP (无 av)")
        return
    from ComfyUI_H3_SeamlessChain import media
    with tempfile.TemporaryDirectory() as d:
        o = os.path.join(d, "out.mp4")
        assert media.concat_av_mp4([], o) is False           # 空清单
        assert media.last_error
        a = os.path.join(d, "a.mp4")
        _make_mp4(a, n_frames=6)
        assert media.concat_av_mp4([a, os.path.join(d, "nope.mp4")], o) is False
        assert not os.path.exists(o)                         # 失败不留产物
        assert not os.path.exists(o + ".part")               # .part 自清理


def test_merge_project_paths_and_manifest():
    if not _HAS:
        print("SKIP (无 av)")
        return
    with _env() as (out, inp):
        from ComfyUI_H3_SeamlessChain import projects
        pdir = _mk_project(out, "甲", {
            "schema": "h3seamless/ckpt-v3", "done": 2, "total": 2,
            "videos": ["seg_000.mp4", "seg_001.mp4"],
            "params": {"width": 64, "height": 48}})
        _make_mp4(os.path.join(pdir, "seg_000.mp4"), n_frames=12, w=64, h=48)
        _make_mp4(os.path.join(pdir, "seg_001.mp4"), n_frames=12, w=64, h=48)
        _make_mp4(os.path.join(pdir, "final_a.mp4"), n_frames=12, w=96, h=64,
                  audio=False)                               # 异画幅成片并入
        os.makedirs(os.path.join(inp, "素材"))               # input 子目录素材
        _make_mp4(os.path.join(inp, "素材", "ext.mp4"), n_frames=12, w=64, h=48,
                  tone=880)

        mf = projects.merge_project("甲", [
            {"seg": 1}, {"seg": 2}, {"file": "final_a.mp4"}, {"file": "素材/ext.mp4"}])
        merged = mf["merges"][-1]["file"]
        assert merged.startswith("merged_") and merged.endswith(".mp4")
        n, size, a_dur = _probe(os.path.join(pdir, merged))
        assert n == 48                                       # 12×4
        assert size == (64, 48)                              # final_a 被压回项目画幅
        assert a_dur is not None                             # 混合有无音轨源
        assert len(mf["merges"]) == 1 and mf["merges"][0]["items"] == [
            {"seg": 1}, {"seg": 2}, {"file": "final_a.mp4"}, {"file": "素材/ext.mp4"}]

        # 项目目录优先于 input：同名文件取项目内的
        _make_mp4(os.path.join(pdir, "both.mp4"), n_frames=6)
        _make_mp4(os.path.join(inp, "both.mp4"), n_frames=18)
        mf2 = projects.merge_project("甲", [{"file": "both.mp4"}])
        n2, _, _ = _probe(os.path.join(pdir, mf2["merges"][-1]["file"]))
        assert n2 == 6                                       # 项目目录那份（6 帧）
        assert len(mf2["merges"]) == 2                       # 第二条记录追加

        # 列表摘要带上 merges
        lst = projects.list_projects()
        assert lst[0]["merges"] == [merged, mf2["merges"][-1]["file"]]


def test_merge_project_rejects():
    if not _HAS:
        print("SKIP (无 av)")
        return
    with _env() as (out, inp):
        from ComfyUI_H3_SeamlessChain import projects
        pdir = _mk_project(out, "甲", {
            "schema": "h3seamless/ckpt-v3", "done": 1, "total": 2,
            "videos": ["seg_000.mp4", ""], "params": {"width": 64, "height": 48}})
        _make_mp4(os.path.join(pdir, "seg_000.mp4"), n_frames=6)
        _make_mp4(os.path.join(inp, "evil.mp4"), n_frames=6)

        def _raise(items, name="甲"):
            try:
                projects.merge_project(name, items)
                return None
            except ValueError as e:
                return str(e)

        assert _raise([], "nope")                            # 项目不存在
        assert _raise([])                                    # 空清单
        assert _raise([{"seg": 3}])                          # 段号越界
        assert _raise([{"seg": 2}])                          # 空串占位（段未出 mp4）
        assert _raise([{"file": "../evil.mp4"}])             # 穿越到 input 根之外
        assert _raise([{"file": "a/../../evil.mp4"}])        # 多级穿越
        assert _raise([{"file": "nope.mp4"}])                # 两目录均无
        assert _raise(["seg1"])                              # 清单项非对象
        # 越界/穿越都不触磁盘读源（无 manifest 变更）
        with open(os.path.join(pdir, "manifest.json"), encoding="utf-8") as f:
            assert "merges" not in json.load(f)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
