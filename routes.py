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


def register(routes=None):
    """把路由挂到 PromptServer.instance.routes（自定义节点标准方式）。

    兼容多种运行环境：
    - 标准 ComfyUI：custom nodes 加载晚于 PromptServer 构造，instance 已就绪。
    - Comfy Desktop / 新版 ComfyUI：可能通过 ComfyExtension.add_routes(routes)
      直接把 routes 对象传进来，此时用 routes 参数注册。
    - PromptServer.instance 尚未构造：给 PromptServer.__init__ 安装钩子，
      在实例化瞬间自动注册。
    - 测试环境：无 server 模块，返回 False。
    """
    global _registered
    if _registered:
        return True

    # 扩展钩子直接传入了 routes 对象（Comfy Desktop / 新版 ComfyUI 官方路径）
    if routes is not None:
        try:
            add_routes(routes)
            _registered = True
            print("[ComfyUI_H3_SeamlessChain] 路由已注册：POST /h3chain/delete, /h3chain/delete_archive")
            return True
        except Exception:
            print("[ComfyUI_H3_SeamlessChain] 路由注册失败（删除功能不可用）。详细错误：")
            traceback.print_exc()
            return False

    PromptServer = _import_promptserver()
    if PromptServer is None:
        print("[ComfyUI_H3_SeamlessChain] 未找到 PromptServer，跳过路由注册（测试环境正常）")
        return False

    inst = getattr(PromptServer, "instance", None)
    if inst is None:
        # PromptServer 尚未构造：装钩子，实例化后自动注册
        _install_promptserver_hook(PromptServer)
        return False

    # 优先直接挂到运行中的 app.router：绕过 RouteTableDef 的挂载时序问题。
    # ComfyUI 部分版本（如 0.33.x）在 custom node 导入前已执行
    # app.add_routes(PromptServer.instance.routes)，导入期加进 RouteTableDef
    # 的路由不会被装载，导致代码里"注册成功"但浏览器访问 404。
    app = getattr(inst, "app", None)
    router = getattr(app, "router", None) if app is not None else None
    target = router if router is not None else getattr(inst, "routes", None)
    if target is not None:
        try:
            add_routes(target)
            _registered = True
            where = "app.router" if router is not None else "PromptServer.instance.routes"
            print(f"[ComfyUI_H3_SeamlessChain] 路由已注册（{where}）：POST /h3chain/delete, /h3chain/delete_archive")
            return True
        except Exception:
            # app.router 失败则回退 RouteTableDef（保留旧版兼容）
            routes2 = getattr(inst, "routes", None)
            if routes2 is not None and routes2 is not target:
                try:
                    add_routes(routes2)
                    _registered = True
                    print("[ComfyUI_H3_SeamlessChain] 路由已注册：POST /h3chain/delete, /h3chain/delete_archive")
                    return True
                except Exception:
                    traceback.print_exc()
                    return False
            print("[ComfyUI_H3_SeamlessChain] 路由注册失败（删除功能不可用）。详细错误：")
            traceback.print_exc()
            return False

    # app 还没建好：装钩子，等 __init__ 完成后重试（此时 app.router 应已就绪）
    _install_promptserver_hook(PromptServer)
    return False


def _import_promptserver():
    """尝试多种 PromptServer 导入路径（标准 ComfyUI / Comfy Desktop / 打包版）。"""
    import importlib
    for path in ("server", "comfy.server", "comfy_api.server"):
        try:
            return importlib.import_module(path).PromptServer
        except Exception:
            continue
    return None


def _install_promptserver_hook(PromptServer):
    """当 PromptServer 实例化时自动注册路由（处理加载顺序不一致的 Desktop 环境）。"""
    if getattr(PromptServer, "_h3_route_hook_installed", False):
        return
    orig_init = PromptServer.__init__

    def _h3_init(self, *args, **kwargs):
        result = orig_init(self, *args, **kwargs)
        try:
            register()
        except Exception:
            pass
        return result

    PromptServer.__init__ = _h3_init
    PromptServer._h3_route_hook_installed = True
    print("[ComfyUI_H3_SeamlessChain] PromptServer 尚未实例化，已安装实例化后自动注册路由的钩子")
