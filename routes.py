"""成片保存节点画廊的删除路由（官方 ComfyExtension.add_routes 钩子挂载）。

前端（web/h3chain_saver.js）删除历史成片走 POST /h3chain/delete，
受限于本节点输出目录内、两级相对路径、防目录穿越；无新依赖。
"""

import os

from aiohttp import web
from folder_paths import get_output_directory


def safe_name(name):
    """文件名安全校验：拒空、路径分隔、盘符、与点开头（防目录穿越）。"""
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

    @routes.post("/h3chain/delete")
    async def delete(request):
        data = await request.json()
        if "file" not in data:
            return web.json_response({"error": "缺少 file 参数"}, status=400)
        rel = safe_relfile(data.get("file"))
        root = os.path.realpath(get_output_directory())
        target = os.path.realpath(os.path.join(root, rel)) if rel else root
        if not rel or not target.startswith(root + os.sep) or not os.path.isfile(target):
            return web.json_response({"error": "文件不存在"}, status=404)
        os.remove(target)
        return web.json_response({"ok": True})
