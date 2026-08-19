"""成片画廊删除路由单测（stub 环境挂真实 handler）：python tests/test_routes.py

覆盖：相对文件名安全校验（防目录穿越）、delete 路由真实挂载与端点行为
（限 output 内两级路径；穿越在语法层即被拒，不触文件系统）。
"""
import asyncio
import contextlib
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插件父目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests 目录

from test_node_structure import _install_stubs


def test_safe_name():
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes
        for ok in ("链A", "h3chain_864x480", "20260818_120000_x", "a"):
            assert routes.safe_name(ok) == ok
        for bad in ("", "  ", ".", "..", "a/b", "a\\b", "c:d", ".hidden", "../up", None):
            assert routes.safe_name(bad) is None


def test_safe_relfile():
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes
        assert routes.safe_relfile("h3_chain/final_1.mp4") == "h3_chain/final_1.mp4"
        for bad in ("h3_chain", "a/b/c.mp4", "../h3_chain/x.mp4", "h3_chain/noext",
                    "", ".hidden/x.mp4", "/abs/path.mp4"):
            assert routes.safe_relfile(bad) is None                # 恒两级、可读名、带扩展名


class _Req:
    def __init__(self, payload):
        self._p = payload

    async def json(self):
        return self._p


class _Router:
    def __init__(self):
        self.table = {}

    def _deco(self, path):
        def deco(fn):
            self.table[path] = fn
            return fn
        return deco

    def get(self, path):
        return self._deco(path)

    def post(self, path):
        return self._deco(path)


@contextlib.contextmanager
def _server_env():
    """临时 output 目录 + aiohttp.web stub，返回捕获到 handler 的路由表。"""
    out = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(
        get_output_directory=lambda: out)

    def json_response(data, status=200):
        return {"json": data, "status": status}

    aiohttp = types.ModuleType("aiohttp")
    aiohttp.web = types.SimpleNamespace(json_response=json_response)
    sys.modules["aiohttp"] = aiohttp
    router = _Router()
    try:
        from ComfyUI_H3_SeamlessChain import routes
        routes.add_routes(router)
        yield out, router
    finally:
        del sys.modules["folder_paths"], sys.modules["aiohttp"]
        # 插件模块也一并出缓存：routes 顶层绑定了 folder_paths.get_output_directory，
        # 不清的话第二个用例会拿到已 rmtree 的旧临时目录
        for k in [k for k in sys.modules if k.startswith("ComfyUI_H3_SeamlessChain")]:
            del sys.modules[k]
        shutil.rmtree(out)


def test_routes_endpoints():
    with _server_env() as (out, router):
        assert set(router.table) == {"/h3chain/delete", "/h3chain/delete_archive"}

        # delete 文件：限 output 内两级路径
        fdir = os.path.join(out, "h3_chain")
        os.makedirs(fdir)
        open(os.path.join(fdir, "final_1.mp4"), "wb").close()
        assert asyncio.run(router.table["/h3chain/delete"](
            _Req({"file": "h3_chain/final_1.mp4"})))["json"] == {"ok": True}
        assert not os.path.exists(os.path.join(fdir, "final_1.mp4"))
        for bad in ("h3_chain/nope.mp4", "../h3_chain/x.mp4", "h3_chain", "/abs/x.mp4", None):
            resp = asyncio.run(router.table["/h3chain/delete"](_Req({"file": bad})))
            assert resp["status"] == 404
        resp = asyncio.run(router.table["/h3chain/delete"](_Req({})))
        assert resp["status"] == 400                              # 缺参数


def test_delete_archive():
    with _server_env() as (out, router):
        handler = router.table["/h3chain/delete_archive"]
        # 同时存在 checkpoints 与 h3_auto 下的同名存档：一起删除
        for sub in ("checkpoints", "h3_auto"):
            d = os.path.join(out, sub, "链A")
            os.makedirs(d)
            open(os.path.join(d, "x.bin"), "wb").close()
        # checkpoints/链A 还带状态指针 -> 删除后指针一并清掉
        state = os.path.join(out, "checkpoints", "h3chain_state.json")
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"dir": "链A"}')
        resp = asyncio.run(handler(_Req({"dir": "链A"})))
        assert resp["json"]["ok"] is True
        assert set(resp["json"]["deleted"]) == {"checkpoints/链A", "h3_auto/链A"}
        assert not os.path.exists(os.path.join(out, "checkpoints", "链A"))
        assert not os.path.exists(os.path.join(out, "h3_auto", "链A"))
        assert not os.path.exists(state)                          # 指向被删目录的状态指针已清

        # 无效名 / 缺参数 -> 400；不存在的存档 -> 404
        for bad in ({"dir": "../up"}, {"dir": "a/b"}, {}):
            assert asyncio.run(handler(_Req(bad)))["status"] == 400
        assert asyncio.run(handler(_Req({"dir": "nope"})))["status"] == 404

        # 删除别的存档不清别人的指针
        os.makedirs(os.path.join(out, "checkpoints", "链B"))
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"dir": "链B"}')
        asyncio.run(handler(_Req({"dir": "链A"})))                # 链A 已不存在 -> 404，指针不动
        assert asyncio.run(handler(_Req({"dir": "链A"})))["status"] == 404
        assert os.path.exists(state)


def test_register_promptserver():
    """register()：server stub 注入 -> 挂到 PromptServer.instance.routes；幂等不重复挂。"""
    import types as _types
    with _server_env():
        router2 = _Router()
        srv = _types.ModuleType("server")

        class _PS:
            instance = _types.SimpleNamespace(routes=router2)

        srv.PromptServer = _PS
        sys.modules["server"] = srv
        try:
            from ComfyUI_H3_SeamlessChain import routes as r1
            assert r1.register() is True
            assert set(router2.table) == {"/h3chain/delete", "/h3chain/delete_archive"}
            assert r1.register() is True                     # 幂等：防重复注册
            assert len(router2.table) == 2
            resp = asyncio.run(router2.table["/h3chain/delete_archive"](_Req({"dir": "nope"})))
            assert resp["status"] == 404                     # 挂载的 handler 真实可用
        finally:
            del sys.modules["server"]


def test_register_no_server_module():
    """测试环境（无 server 模块）：register() 返回 False 且不抛异常。"""
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes as r2
        sys.modules.pop("server", None)
        assert r2.register() is False


def test_register_prefers_app_router():
    """register()：server 同时有 app.router 与 routes 时，优先挂到 app.router。

    对应 ComfyUI 0.33.x 在 custom node 导入前已 app.add_routes(routes) 的情况：
    挂进 RouteTableDef 的路由不会被装载，必须直接挂到运行中的 app.router。
    """
    with _server_env():
        router3 = _Router()
        router4 = _Router()  # 不应被使用
        ps_instance = types.SimpleNamespace(
            app=types.SimpleNamespace(router=router3),
            routes=router4,
        )

        class _PS:
            instance = ps_instance
        srv = types.SimpleNamespace(PromptServer=_PS)
        sys.modules["server"] = srv
        try:
            from ComfyUI_H3_SeamlessChain import routes as r3
            assert r3.register() is True
            # 路由只挂到了 app.router（router3），未挂到 RouteTableDef（router4）
            assert set(router3.table) == {"/h3chain/delete", "/h3chain/delete_archive"}
            assert router4.table == {}
            resp = asyncio.run(router3.table["/h3chain/delete_archive"](_Req({"dir": "nope"})))
            assert resp["status"] == 404
        finally:
            del sys.modules["server"]


class _RouterAddPost:
    """模拟 aiohttp 的 UrlDispatcher（app.router）：只有 add_post，没有 .post 装饰器。"""
    def __init__(self):
        self.table = {}

    def add_post(self, path, handler):
        self.table[path] = handler


def test_add_routes_on_app_router():
    """add_routes 应支持 UrlDispatcher（app.router）的 add_post，而非仅 RouteTableDef.post。

    此前用 @routes.post 装饰器，对 app.router 会抛 AttributeError 并回退到未装载的
    RouteTableDef，导致 404。此测试保证 app.router 路径真正挂上 handler。
    """
    with _server_env():
        router = _RouterAddPost()
        from ComfyUI_H3_SeamlessChain import routes
        routes.add_routes(router)
        assert set(router.table) == {"/h3chain/delete", "/h3chain/delete_archive"}
        # handler 真实可用：缺参 400
        assert asyncio.run(router.table["/h3chain/delete"](_Req({})))["status"] == 400
        # 不存在的存档 404
        assert asyncio.run(router.table["/h3chain/delete_archive"](_Req({"dir": "nope"})))["status"] == 404


def test_add_routes_rejects_unknown_type():
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes
        try:
            routes.add_routes(object())
            assert False, "expected TypeError"
        except TypeError:
            pass


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
