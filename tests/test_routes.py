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
    from ComfyUI_H3_SeamlessChain import routes
    for ok in ("链A", "h3chain_864x480", "20260818_120000_x", "a"):
        assert routes.safe_name(ok) == ok
    for bad in ("", "  ", ".", "..", "a/b", "a\\b", "c:d", ".hidden", "../up", None):
        assert routes.safe_name(bad) is None


def test_safe_relfile():
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
        shutil.rmtree(out)


def test_routes_endpoints():
    with _server_env() as (out, router):
        assert set(router.table) == {"/h3chain/delete"}

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


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
