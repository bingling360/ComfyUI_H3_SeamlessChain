"""断点续跑：参数指纹 + manifest 原子读写 + 段 AV latent 落盘。

断点只存采样输出的 latent（约 5 MB/段），不存解码像素（约 300 MB/段）：
续跑时重解码秒级回放，结果与一次跑完逐帧一致（每段种子 = 种子+i，无隐藏状态）。

torch / folder_paths 延迟导入：指纹与 manifest 逻辑在无 ComfyUI 环境下可单测。
"""

import hashlib
import json
import os
import tempfile

SCHEMA = "h3seamless/ckpt-v1"


def fingerprint(params: dict) -> str:
    """严格参数 -> 8 位十六进制指纹（sha256，跨进程稳定）。

    params 不含种子：种子控件开着 control_after_generate，每次运行自动 +1，
    若入指纹则崩溃后续跑永远找不到原目录；种子由 manifest 权威记录。
    """
    blob = json.dumps(params, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:8]


def ckpt_dir(params: dict, custom: str = "") -> str:
    """断点根目录：output/checkpoints/<自定义名> 或按参数指纹自动命名。"""
    from folder_paths import get_output_directory

    if custom:
        root = os.path.join(get_output_directory(), "checkpoints", custom)
    else:
        name = (f"h3chain_{params['width']}x{params['height']}"
                f"_{params['length']}f_ctx{params['ctx']}_{fingerprint(params)}")
        root = os.path.join(get_output_directory(), "checkpoints", name)
    os.makedirs(root, exist_ok=True)
    return root


def _atomic_write(path: str, data: bytes):
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), prefix=".tmp_", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def save_manifest(root: str, manifest: dict):
    _atomic_write(os.path.join(root, "manifest.json"),
                  json.dumps(manifest, ensure_ascii=False, indent=1).encode("utf-8"))


def load_manifest(root: str):
    path = os.path.join(root, "manifest.json")
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def assert_match(old: dict, new: dict):
    """严格校验断点参数；不一致直接报错（种子例外，由调用方以断点为准）。"""
    diffs = [k for k in new if old.get(k) != new[k]]
    if diffs:
        detail = "; ".join(f"{k}: 断点={old.get(k)!r} 当前={new[k]!r}" for k in diffs)
        raise ValueError(
            f"断点目录参数与当前不一致（{detail}）。续拍必须沿用原参数原提示词；"
            "要开新链请清空断点目录，或在「断点目录」里填一个新名字")


def seg_path(root: str, idx: int) -> str:
    return os.path.join(root, f"seg_{idx:03d}.pt")


def contiguous_done(root: str, done: int) -> int:
    """manifest 记录的完成段数，遇到缺失的段文件向前截断（该段起重新采样）。"""
    while done > 0 and not os.path.exists(seg_path(root, done - 1)):
        done -= 1
    return done


def save_segment(root: str, idx: int, video_t, audio_t):
    """段 AV latent 原子落盘（CPU 副本，不动显存里的原张量）。"""
    import torch

    payload = {
        "video": video_t.detach().cpu().clone(),
        "audio": audio_t.detach().cpu().clone(),
    }
    buf = tempfile.SpooledTemporaryFile(max_size=64 << 20)
    torch.save(payload, buf)
    buf.seek(0)
    _atomic_write(seg_path(root, idx), buf.read())
    buf.close()


def load_segment(root: str, idx: int):
    """读回段 AV latent（CPU 张量，调用方按需 .to(device)）。"""
    import torch

    with open(seg_path(root, idx), "rb") as f:
        payload = torch.load(f, map_location="cpu", weights_only=True)
    return payload["video"], payload["audio"]
