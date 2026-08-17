"""控制台/成片保存节点的三个轻量路由（官方 ComfyExtension.add_routes 钩子挂载）。

前端（web/h3chain_console.js / h3chain_saver.js）需要目录列举与上传能力，
ComfyUI 自带 /api/view 只能读已知路径的文件——本模块补三个只读/受控端点，
无新依赖；钩子不可用时前端自动降级（手输存档名 + state.json 单链浏览）。
"""

import json
import os
import shutil
import time

ALLOWED_EXT = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".mov", ".mkv", ".webm", ".wav"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm"}


def archives_index(root):
    """存档根目录 -> 存档卡片列表（按更新时间倒序）。

    纯函数（只吃路径），路由与单测共用。manifest 损坏的目录跳过不炸接口。
    """
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        mpath = os.path.join(root, name, "manifest.json")
        if not os.path.isfile(mpath):
            continue
        try:
            with open(mpath, "r", encoding="utf-8") as f:
                mf = json.load(f)
        except Exception:
            continue
        seams = [s[0] for s in (mf.get("seams") or [])
                 if isinstance(s, (list, tuple)) and s and s[0] is not None]
        params = mf.get("params") or {}
        out.append({
            "name": name,
            "done": int(mf.get("done", 0)),
            "total": int(mf.get("total", 0)),
            "updated_at": os.path.getmtime(mpath),
            "has_prologue": bool(mf.get("has_prologue")),
            "prompts": mf.get("prompts") or [],
            "worst_seam": max(seams) if seams else None,
            "width": params.get("width"), "height": params.get("height"),
            "length": params.get("length"), "ctx": params.get("ctx"),
        })
    out.sort(key=lambda a: a["updated_at"], reverse=True)
    return out


def safe_name(name):
    """存档名安全校验：拒空、路径分隔、盘符、与点开头（防目录穿越）。"""
    s = str(name or "").strip()
    if not s or s.startswith(".") or "/" in s or "\\" in s or ":" in s or ".." in s:
        return None
    return s


def safe_relfile(rel):
    """saver 输出目录内单文件：<前缀>/<文件名>，两级、无穿越。"""
    s = str(rel or "").strip()
    parts = s.split("/")
    if (len(parts) != 2 or not all(safe_name(p) for p in parts)
            or not os.path.splitext(parts[1])[1]):
        return None
    return s


def add_routes(routes):
    """挂到 PromptServer routes（aiohttp）；由 __init__.py 的扩展钩子调用。"""
    from aiohttp import web
    from folder_paths import get_input_directory, get_output_directory

    from . import checkpoint

    @routes.get("/h3chain/archives")
    async def list_archives(request):
        return web.json_response({"archives": archives_index(checkpoint.checkpoints_root())})

    @routes.post("/h3chain/upload")
    async def upload(request):
        reader = await request.multipart()
        part = await reader.next()
        if part is None:
            return web.json_response({"error": "缺少文件"}, status=400)
        raw = part.filename or ""
        ext = os.path.splitext(raw)[1].lower()
        base = safe_name(os.path.basename(raw))
        if not base or ext not in ALLOWED_EXT:
            return web.json_response({"error": f"不支持的文件类型：{raw}"}, status=400)
        dest_dir = os.path.join(get_input_directory(), "h3chain")
        os.makedirs(dest_dir, exist_ok=True)
        final = f"{time.strftime('%Y%m%d_%H%M%S')}_{base}"
        path = os.path.join(dest_dir, final)
        with open(path, "wb") as f:
            while True:
                chunk = await part.read_chunk(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        kind = "video" if ext in VIDEO_EXT else ("audio" if ext == ".wav" else "image")
        return web.json_response({"filename": f"h3chain/{final}", "kind": kind})

    @routes.post("/h3chain/delete")
    async def delete(request):
        data = await request.json()
        if data.get("archive"):
            name = safe_name(data["archive"])
            root = os.path.realpath(checkpoint.checkpoints_root())
            target = os.path.realpath(os.path.join(root, name)) if name else root
            if not name or not target.startswith(root + os.sep) or not os.path.isdir(target):
                return web.json_response({"error": "存档不存在"}, status=404)
            shutil.rmtree(target)
            return web.json_response({"ok": True})
        if data.get("file"):
            rel = safe_relfile(data["file"])
            root = os.path.realpath(get_output_directory())
            target = os.path.realpath(os.path.join(root, rel)) if rel else root
            if not rel or not target.startswith(root + os.sep) or not os.path.isfile(target):
                return web.json_response({"error": "文件不存在"}, status=404)
            os.remove(target)
            return web.json_response({"ok": True})
        return web.json_response({"error": "缺少 archive 或 file 参数"}, status=400)
