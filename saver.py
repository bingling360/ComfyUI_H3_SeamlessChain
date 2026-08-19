"""H3ChainSaver —— 成片保存节点（一体化）：最终成片 + 分段副本 + 历史索引。

替代 VHS_VideoCombine 的收尾位：最终成片走 PyAV 编码（media.save_av_mp4，
与分段视频同款）；分段视频不重编码——从当前存档目录复制 seg_*.mp4 到
输出目录 segs/，画廊（web/h3chain_saver.js）按 h3saver_history.json 浏览
「成片历史 / 本次分段 / 当前成片」并支持删除。
"""

import json
import os
import shutil
import time

from comfy_api.latest import io

from . import checkpoint
from .media import save_av_mp4


def saver_dir(prefix):
    """输出根：output/<前缀>/（前缀空或非法时回落 h3_chain）。"""
    from folder_paths import get_output_directory

    p = str(prefix or "").strip().replace("/", "_").replace("\\", "_").replace(".", "_")
    root = os.path.join(get_output_directory(), p or "h3_chain")
    os.makedirs(root, exist_ok=True)
    return root


def append_history(dirpath, entry):
    """原子读改写 h3saver_history.json（新条目插最前，历史只增不改）。"""
    path = os.path.join(dirpath, "h3saver_history.json")
    items = []
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                items = json.load(f)
        except Exception:
            items = []
    items.insert(0, entry)
    blob = json.dumps(items, ensure_ascii=False, indent=1).encode("utf-8")
    fd = None
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=dirpath, prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def copy_segments(dirpath):
    """当前存档（state.json 指针）的 seg_*.mp4 -> 输出目录 segs/，返回文件名列表。"""
    state = checkpoint.load_state() or {}
    name = state.get("dir") or ""
    if not name or "/" in name or "\\" in name or ".." in name:
        return []
    src = os.path.join(checkpoint.checkpoints_root(), name)
    if not os.path.isdir(src):
        return []
    dst = os.path.join(dirpath, "segs")
    os.makedirs(dst, exist_ok=True)
    copied = []
    for fn in sorted(os.listdir(src)):
        if fn.startswith("seg_") and fn.endswith(".mp4"):
            shutil.copy2(os.path.join(src, fn), os.path.join(dst, fn))
            copied.append(f"segs/{fn}")
    return copied


class H3ChainSaver(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainSaver",
            display_name="H3 Chain Saver (成片保存)",
            category="MiniMaxH3",
            description="一体化成片保存：最终视频 PyAV 编码落盘，分段视频从当前存档复制（不重编码），"
                        "节点面板画廊浏览成片历史/本次分段/当前成片，可删除。接采样器的 图像/音频/帧率。",
            is_output_node=True,
            inputs=[
                io.Image.Input("图像", tooltip="完整链（或逐段审片时已确认部分）的可见帧"),
                io.Audio.Input("音频", tooltip="与图像配对的音轨（采样器「音频」输出自带采样率）"),
                io.Int.Input("帧率", default=24, min=1, max=120),
                io.String.Input("输出前缀", default="h3_chain",
                                tooltip="输出目录名：output/<前缀>/ 下放成片、segs/ 分段与 h3saver_history.json"),
                io.Int.Input("CRF", default=20, min=0, max=51,
                             tooltip="H.264 质量因子：小=画质高体积大（18-22 常用）"),
            ],
            outputs=[
                io.String.Output("成片文件名", tooltip="本次保存的成片文件名（相对输出目录），编码失败为空串"),
            ],
        )

    @classmethod
    def execute(cls, 图像, 音频, 帧率, 输出前缀="h3_chain", CRF=20):
        dirpath = saver_dir(输出前缀)
        wav = 音频.get("waveform") if isinstance(音频, dict) else 音频
        sr = int(音频.get("sample_rate", 44100)) if isinstance(音频, dict) else 44100
        name = f"final_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
        ok = save_av_mp4(os.path.join(dirpath, name), 图像, wav, sr, int(帧率), int(CRF))
        if not ok:
            from . import media
            err = media.last_error or "未知原因"
            raise RuntimeError(f"成片编码失败：{err}（分段视频不受影响，仍会复制）")
        segs = copy_segments(dirpath)
        append_history(dirpath, {
            "file": name,
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "frames": int(图像.shape[0]),
            "archive": (checkpoint.load_state() or {}).get("dir") or "",
            "segments": len(segs),
        })
        return io.NodeOutput(name)
