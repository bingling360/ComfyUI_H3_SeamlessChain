"""控制台路由单测（stub 环境挂真实 handler）：python tests/test_routes.py

覆盖：存档名 / 相对文件名安全校验（防目录穿越）、存档索引
（manifest 解析 / 排序 / 坏数据跳过）、三路由真实挂载与 archives/delete 端点行为。
upload 需要真实 multipart，不在单测范围。
"""
import asyncio
import contextlib
import json
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


def test_archives_index():
    from ComfyUI_H3_SeamlessChain import routes
    root = tempfile.mkdtemp()
    try:
        assert routes.archives_index(root) == []               # 空目录
        assert routes.archives_index(os.path.join(root, "nope")) == []
        os.makedirs(os.path.join(root, "older"))
        with open(os.path.join(root, "older", "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"done": 2, "total": 4, "has_prologue": True,
                       "prompts": ["p1", "p2"],
                       "seams": [None, [0.05, 1.0], [0.2, 3.0]],
                       "params": {"width": 864, "height": 480, "length": 124, "ctx": 22}}, f)
        os.utime(os.path.join(root, "older", "manifest.json"), (1, 1))       # 人为变旧
        os.makedirs(os.path.join(root, "newer"))
        with open(os.path.join(root, "newer", "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({}, f)                                   # 空 manifest -> 全默认值
        os.makedirs(os.path.join(root, "broken"))
        with open(os.path.join(root, "broken", "manifest.json"), "w", encoding="utf-8") as f:
            f.write("{broken")                                 # 坏 JSON -> 跳过不炸
        os.makedirs(os.path.join(root, "no_manifest"))         # 无 manifest -> 非存档
        out = routes.archives_index(root)
        assert [a["name"] for a in out] == ["newer", "older"]  # 按更新时间倒序
        old = out[1]
        assert old["done"] == 2 and old["total"] == 4 and old["has_prologue"] is True
        assert old["prompts"] == ["p1", "p2"]
        assert old["worst_seam"] == 0.2                        # 接缝分取最大（诊断用）
        assert old["width"] == 864 and old["ctx"] == 22
        assert out[0]["worst_seam"] is None and out[0]["prompts"] == []
    finally:
        shutil.rmtree(root)


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
        get_output_directory=lambda: out, get_input_directory=lambda: os.path.join(out, "input"))

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
        assert set(router.table) == {"/h3chain/archives", "/h3chain/upload", "/h3chain/delete"}

        # archives：读 checkpoints 下的存档
        arch = os.path.join(out, "checkpoints", "chainA")
        os.makedirs(arch)
        with open(os.path.join(arch, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump({"done": 1, "total": 3, "prompts": ["p1"]}, f)
        resp = asyncio.run(router.table["/h3chain/archives"](None))
        assert resp["status"] == 200
        assert resp["json"]["archives"][0]["name"] == "chainA"

        # delete 存档：真实删除 + 拒绝穿越
        assert asyncio.run(router.table["/h3chain/delete"](
            _Req({"archive": "chainA"})))["json"] == {"ok": True}
        assert not os.path.exists(arch)
        for bad in ("../checkpoints", "..", "no_such"):
            resp = asyncio.run(router.table["/h3chain/delete"](_Req({"archive": bad})))
            assert resp["status"] == 404                          # 穿越/不存在一律 404

        # delete 文件：限 output 内两级路径（穿越在语法层即被拒，不触文件系统）
        fdir = os.path.join(out, "h3_chain")
        os.makedirs(fdir)
        open(os.path.join(fdir, "final_1.mp4"), "wb").close()
        assert asyncio.run(router.table["/h3chain/delete"](
            _Req({"file": "h3_chain/final_1.mp4"})))["json"] == {"ok": True}
        assert not os.path.exists(os.path.join(fdir, "final_1.mp4"))
        for bad in ("h3_chain/nope.mp4", "../h3_chain/x.mp4", "h3_chain", "/abs/x.mp4"):
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
