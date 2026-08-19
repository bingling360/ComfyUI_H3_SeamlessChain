"""成片保存节点画廊与导演台存档的删除路由（官方 ComfyExtension.add_routes 钩子挂载）。

前端删除历史成片走 POST /h3chain/delete（saver 画廊）；
删除项目存档走 POST /h3chain/delete_archive（导演台左栏，删 checkpoints/<名> 与
h3_auto/<名> 两处，若被删的是当前链则一并清掉 h3chain_state.json 指针）。
均受限于输出目录内、防目录穿越；无新依赖。
"""

import json
import os
import shutil
import traceback

from aiohttp import web
from folder_paths import get_output_directory

_registered = False


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

    @routes.post("/h3chain/delete_archive")
    async def delete_archive(request):
        data = await request.json()
        name = safe_name(data.get("dir"))
        if not name or name == "h3chain_state.json":
            return web.json_response({"error": "无效的存档目录名"}, status=400)
        root = os.path.realpath(get_output_directory())
        deleted = []
        for sub in ("checkpoints", "h3_auto"):
            target = os.path.realpath(os.path.join(root, sub, name))
            if not target.startswith(root + os.sep) or not os.path.isdir(target):
                continue
            shutil.rmtree(target)
            deleted.append(os.path.relpath(target, root).replace("\\", "/"))
        # 被删的是当前链时清掉状态指针，防面板继续指向已删目录
        state_path = os.path.join(root, "checkpoints", "h3chain_state.json")
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            if isinstance(state, dict) and state.get("dir") == name:
                os.remove(state_path)
        except Exception:
            pass
        if not deleted:
            return web.json_response({"error": "存档目录不存在"}, status=404)
        return web.json_response({"ok": True, "deleted": deleted})


def register():
    """把路由挂到 PromptServer.instance.routes（自定义节点标准方式）。

    此前用 ComfyExtension.add_routes 扩展钩子注册，但现行 ComfyUI 的基类
    没有 add_routes 钩子、server.py 也不调用它——路由从未注册，
    前端 POST 命中前端托管层返回 405。custom nodes 加载晚于 PromptServer
    构造，导入期 instance 已就绪（ComfyUI-Manager 等同款做法）。
    """
    global _registered
    if _registered:
        return True
    try:
        from server import PromptServer
    except Exception:
        print("[ComfyUI_H3_SeamlessChain] 未找到 server.PromptServer，跳过路由注册（测试环境正常）")
        return False
    try:
        add_routes(PromptServer.instance.routes)
        _registered = True
        print("[ComfyUI_H3_SeamlessChain] 路由已注册：POST /h3chain/delete, /h3chain/delete_archive")
        return True
    except Exception:
        print("[ComfyUI_H3_SeamlessChain] 路由注册失败（删除功能不可用）。详细错误：")
        traceback.print_exc()
        return False
