"""项目存档路由单测（stub 环境挂真实 handler）：python tests/test_routes.py

覆盖：相对文件名安全校验（防目录穿越）、ping/projects/project/delete_project/
delete_file 端点真实挂载与行为、register/ensure_registered 注册回退链。
"""
import asyncio
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
        sys.path.insert(0, _p)   # 插件目录 / 插件父目录 / tests 目录

from test_node_structure import _install_stubs

ROUTE_SET = {"/h3chain/ping", "/h3chain/projects", "/h3chain/project",
             "/h3chain/upscale_models", "/h3chain/create_project", "/h3chain/save_prompts",
             "/h3chain/delete", "/h3chain/delete_project", "/h3chain/delete_file",
             "/h3chain/merge", "/h3chain/upscale_reset"}
# 新版前端 api.fetchApi() 强制给非 /api 路径加 /api 前缀（ComfyApi.apiURL），
# 自定义路由必须同时挂 /api 副本，否则新前端全部 404
API_ROUTE_SET = {"/api" + p for p in ROUTE_SET}
FULL_ROUTE_SET = ROUTE_SET | API_ROUTE_SET


def test_safe_relfile():
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes
        assert routes.safe_relfile("h3_chain/final_1.mp4") == "h3_chain/final_1.mp4"
        for bad in ("h3_chain", "a/b/c.mp4", "../h3_chain/x.mp4", "h3_chain/noext",
                    "", ".hidden/x.mp4", "/abs/path.mp4"):
            assert routes.safe_relfile(bad) is None                # 恒两级、可读名、带扩展名


class _Req:
    def __init__(self, payload=None, query=None):
        self._p = payload
        self.query = query or {}

    async def json(self):
        return self._p


class _Router:
    """RouteTableDef 风格：get/post 装饰器。"""

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
        # 插件模块也一并出缓存：routes/projects 延迟导入 folder_paths，
        # 不清的话第二个用例会拿到已 rmtree 的旧临时目录
        for k in [k for k in sys.modules if k.startswith("ComfyUI_H3_SeamlessChain")]:
            del sys.modules[k]
        shutil.rmtree(out)


def _mk_project(out, name, manifest=None):
    pdir = os.path.join(out, "h3_projects", name)
    os.makedirs(pdir, exist_ok=True)
    if manifest is not None:
        with open(os.path.join(pdir, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False)
    return pdir


def test_ping():
    with _server_env() as (out, router):
        resp = asyncio.run(router.table["/h3chain/ping"](_Req()))
        assert resp["json"]["ok"] is True
        assert {r["path"] for r in resp["json"]["routes"]} == ROUTE_SET


def test_projects_endpoint():
    with _server_env() as (out, router):
        _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "title": "甲项目",
                                "done": 2, "total": 5, "updated_at": 10.0})
        os.makedirs(os.path.join(out, "h3_projects", "空目录"))     # 无 manifest -> 不列出
        resp = asyncio.run(router.table["/h3chain/projects"](_Req()))
        assert resp["json"]["ok"] is True
        assert [p["dir"] for p in resp["json"]["projects"]] == ["甲"]
        assert resp["json"]["projects"][0]["title"] == "甲项目"


def test_project_detail():
    with _server_env() as (out, router):
        handler = router.table["/h3chain/project"]
        _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "done": 1, "prompts": ["p1"]})
        resp = asyncio.run(handler(_Req(query={"dir": "甲"})))
        assert resp["json"]["manifest"]["prompts"] == ["p1"]
        assert asyncio.run(handler(_Req(query={"dir": "nope"})))["status"] == 404
        assert asyncio.run(handler(_Req(query={"dir": "../up"})))["status"] == 404


def test_create_project_endpoint():
    with _server_env() as (out, router):
        handler = router.table["/h3chain/create_project"]
        resp = asyncio.run(handler(_Req({"dir": "乙"})))
        assert resp["json"]["ok"] is True
        pdir = os.path.join(out, "h3_projects", "乙")
        assert os.path.isdir(pdir)                                # 当场建文件夹
        assert resp["json"]["manifest"]["done"] == 0
        # 再创建（幂等）不报错
        assert asyncio.run(handler(_Req({"dir": "乙"})))["json"]["ok"] is True
        for bad in ({"dir": "../up"}, {"dir": "a/b"}, {}):
            assert asyncio.run(handler(_Req(bad)))["status"] == 400


def test_save_prompts_endpoint():
    """切换/编辑项目时提示词回写（持久源=项目文件夹）。"""
    with _server_env() as (out, router):
        handler = router.table["/h3chain/save_prompts"]
        _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "done": 1, "total": 1,
                                "prompts": ["旧词"], "prompt_hashes": ["h1"]})
        resp = asyncio.run(handler(_Req({"dir": "甲", "prompts": ["新词一", "新词二"]})))
        assert resp["json"]["ok"] is True
        assert resp["json"]["manifest"]["prompts"] == ["新词一", "新词二"]
        assert resp["json"]["manifest"]["total"] == 2
        with open(os.path.join(out, "h3_projects", "甲", "manifest.json"),
                  encoding="utf-8") as f:                       # 真落盘
            assert json.load(f)["prompts"] == ["新词一", "新词二"]
        assert asyncio.run(handler(_Req({"dir": "nope", "prompts": []})))["status"] == 404
        assert asyncio.run(handler(_Req({"dir": "../up", "prompts": []})))["status"] == 404
        assert asyncio.run(handler(_Req({"dir": "甲", "prompts": "str"})))["status"] == 404


def test_delete_project_endpoint():
    with _server_env() as (out, router):
        handler = router.table["/h3chain/delete_project"]
        _mk_project(out, "甲", {"done": 1})
        open(os.path.join(out, "h3_projects", "甲", "seg_000.mp4"), "wb").close()
        state = os.path.join(out, "h3_projects", "h3chain_state.json")
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"dir": "甲"}')
        resp = asyncio.run(handler(_Req({"dir": "甲"})))
        assert resp["json"] == {"ok": True, "deleted": "h3_projects/甲"}
        assert not os.path.exists(os.path.join(out, "h3_projects", "甲"))
        assert not os.path.exists(state)                            # 当前链指针一并清掉

        for bad in ({"dir": "../up"}, {"dir": "a/b"}, {}):
            assert asyncio.run(handler(_Req(bad)))["status"] == 400
        assert asyncio.run(handler(_Req({"dir": "nope"})))["status"] == 404


def test_delete_file():
    with _server_env() as (out, router):
        handler = router.table["/h3chain/delete_file"]
        _mk_project(out, "甲")
        fpath = os.path.join(out, "h3_projects", "甲", "final_1.mp4")
        open(fpath, "wb").close()
        resp = asyncio.run(handler(_Req({"path": "h3_projects/甲/final_1.mp4"})))
        assert resp["json"] == {"ok": True}
        assert not os.path.exists(fpath)

        # 越权/非法路径 -> 400（语法层拒绝，不触文件系统）
        for bad in ("h3_projects/甲", "h3_projects/甲/a/b.mp4", "checkpoints/甲/x.mp4",
                    "../h3_projects/甲/x.mp4", "h3_projects/../甲/x.mp4", "",
                    "h3_projects/甲/noext", "/abs/x.mp4"):
            assert asyncio.run(handler(_Req({"path": bad})))["status"] == 400
        # 合法格式但文件不存在 -> 404
        assert asyncio.run(handler(_Req({"path": "h3_projects/甲/nope.mp4"})))["status"] == 404


def test_delete_endpoint():
    with _server_env() as (out, router):
        # delete 文件：限 output 内两级路径（saver 画廊）
        fdir = os.path.join(out, "h3_chain")
        os.makedirs(fdir)
        open(os.path.join(fdir, "final_1.mp4"), "wb").close()
        assert asyncio.run(router.table["/h3chain/delete"](
            _Req({"file": "h3_chain/final_1.mp4"})))["json"] == {"ok": True}
        assert not os.path.exists(os.path.join(out, "h3_chain", "final_1.mp4"))
        for bad in ("h3_chain/nope.mp4", "../h3_chain/x.mp4", "h3_chain", "/abs/x.mp4", None):
            resp = asyncio.run(router.table["/h3chain/delete"](_Req({"file": bad})))
            assert resp["status"] == 404
        resp = asyncio.run(router.table["/h3chain/delete"](_Req({})))
        assert resp["status"] == 400                              # 缺参数


def test_merge_endpoint_errors():
    """merge 端点错误路径（不触真实编码）：清单/项目/段号校验 -> 400。"""
    with _server_env() as (out, router):
        handler = router.table["/h3chain/merge"]
        _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "done": 2, "total": 3,
                                "videos": ["seg_000.mp4", "", "seg_002.mp4"],
                                "params": {"width": 864, "height": 480}})
        open(os.path.join(out, "h3_projects", "甲", "seg_000.mp4"), "wb").close()
        open(os.path.join(out, "h3_projects", "甲", "seg_002.mp4"), "wb").close()

        for bad in ({"dir": "nope", "items": [{"seg": 1}]},          # 项目不存在
                    {"dir": "../up", "items": [{"seg": 1}]},         # 穿越名
                    {"dir": "甲", "items": []},                      # 空清单
                    {"dir": "甲", "items": "not-a-list"},            # 清单类型
                    {"dir": "甲", "items": [{"seg": 0}]},            # 段号下界（1-based）
                    {"dir": "甲", "items": [{"seg": 4}]},            # 段号越界（>len(videos)）
                    {"dir": "甲", "items": [{"seg": 2}]},            # 该段无 mp4（空串占位）
                    {"dir": "甲", "items": [{"seg": "x"}]},          # 段号非整数
                    {"dir": "甲", "items": [{"file": "../up.mp4"}]},  # 穿越文件名
                    {"dir": "甲", "items": [{"file": "noext"}]},     # 缺扩展名
                    {"dir": "甲", "items": [{"file": "nope.mp4"}]},  # 两目录均无此文件
                    {"dir": "甲", "items": ["seg1"]}):               # 清单项非对象
            resp = asyncio.run(handler(_Req(bad)))
            assert resp["status"] == 400, bad
            assert "error" in resp["json"]

        # 混合清单里一项缺失 -> 整单 400（乙有 seg_000.mp4 但 missing.mp4 不存在）
        _mk_project(out, "乙", {"schema": "h3seamless/ckpt-v3", "done": 1,
                                "videos": ["seg_000.mp4"]})
        open(os.path.join(out, "h3_projects", "乙", "seg_000.mp4"), "wb").close()
        resp = asyncio.run(handler(_Req({"dir": "乙",
                                         "items": [{"seg": 1}, {"file": "missing.mp4"}]})))
        assert resp["status"] == 400
        # 失败不落 manifest.merges
        with open(os.path.join(out, "h3_projects", "乙", "manifest.json"),
                  encoding="utf-8") as f:
            assert "merges" not in json.load(f)


def test_merge_endpoint_encode_failure_maps_500():
    """merge 编码失败（0 字节 mp4）-> RuntimeError -> 500，无 .part 残留。

    executor 内异常能传播回 handler 且映射 500；真实成功路径的合成视频
    断言在 test_merge.py（需要 av）。
    """
    with _server_env() as (out, router):
        handler = router.table["/h3chain/merge"]
        pdir = _mk_project(out, "丙", {"schema": "h3seamless/ckpt-v3", "done": 2, "total": 2,
                                       "videos": ["seg_000.mp4", "seg_001.mp4"],
                                       "params": {"width": 864, "height": 480}})
        open(os.path.join(pdir, "seg_000.mp4"), "wb").close()
        open(os.path.join(pdir, "seg_001.mp4"), "wb").close()
        resp = asyncio.run(handler(_Req({"dir": "丙", "items": [{"seg": 1}, {"seg": 2}]})))
        assert resp["status"] == 500
        assert "合并编码失败" in resp["json"]["error"]
        assert [f for f in os.listdir(pdir) if f.endswith(".part")] == []  # 无半成品
        with open(os.path.join(pdir, "manifest.json"), encoding="utf-8") as f:
            assert "merges" not in json.load(f)                   # 失败不记档


def test_upscale_models_endpoint():
    """二采权重目录列表：无 torch/无目录环境恒 ok（前端下拉数据源兜底空列表）。"""
    with _server_env() as (out, router):
        handler = router.table["/h3chain/upscale_models"]
        resp = asyncio.run(handler(_Req()))
        assert resp["json"] == {"ok": True, "models": []}       # stub 无模型目录 -> 空


def test_upscale_reset_endpoint():
    """二采记录重置端点：清记录+删高清产物（同名合并存储 seg mp4 一并删）；非法 -> 400。"""
    with _server_env() as (out, router):
        handler = router.table["/h3chain/upscale_reset"]
        pdir = _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "done": 2, "total": 2,
                                       "prompt_hashes": ["a", "b"], "seeds": [1, 2],
                                       "upscale": {"segs": [
                                           {"hash": "h", "base_hash": "a|1", "done": True},
                                           {"hash": "h", "base_hash": "b|2", "done": True}]}})
        made = ["seg_001.mp4", "thumb_001.png", "uplast_001.png", "upseg_001.mp4"]
        for f in made:
            open(os.path.join(pdir, f), "wb").close()

        resp = asyncio.run(handler(_Req({"dir": "甲", "seg": 2})))
        assert resp["json"]["ok"] is True
        assert resp["json"]["manifest"]["upscale"]["segs"][1] is None    # 记录已清
        assert resp["json"]["manifest"]["upscale"]["segs"][0] is not None
        for f in made:
            assert not os.path.exists(os.path.join(pdir, f)), f         # 产物已删（含旧版名）

        # 段号/项目名非法 -> 400（ValueError 映射）
        for bad in ({"dir": "甲", "seg": "x"}, {"dir": "甲", "seg": 0},
                    {"dir": "甲", "seg": 3}, {"dir": "nope", "seg": 1},
                    {"dir": "../up", "seg": 1}, {"dir": "", "seg": 1}, {}):
            resp = asyncio.run(handler(_Req(bad)))
            assert resp["status"] == 400, bad
            assert "error" in resp["json"]


def test_register_promptserver():
    """register()：server stub 注入 -> 挂到 PromptServer.instance.routes；幂等不重复挂。"""
    with _server_env():
        router2 = _Router()
        srv = types.ModuleType("server")

        class _PS:
            instance = types.SimpleNamespace(routes=router2)

        srv.PromptServer = _PS
        sys.modules["server"] = srv
        try:
            from ComfyUI_H3_SeamlessChain import routes as r1
            assert r1.register() is True
            assert set(router2.table) == FULL_ROUTE_SET
            assert r1.register() is True                     # 幂等：防重复注册
            assert len(router2.table) == len(FULL_ROUTE_SET)
            resp = asyncio.run(router2.table["/h3chain/delete_project"](_Req({"dir": "nope"})))
            assert resp["status"] == 404                     # 挂载的 handler 真实可用
        finally:
            del sys.modules["server"]


def test_register_no_server_module():
    """测试环境（无 server 模块）：register() 返回 False 且不抛异常。"""
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes as r2
        sys.modules.pop("server", None)
        assert r2.register() is False


def test_ensure_registered():
    """ensure_registered()：未注册时重试，成功后短路（幂等）。"""
    with _server_env():
        from ComfyUI_H3_SeamlessChain import routes as r3
        sys.modules.pop("server", None)
        assert r3.ensure_registered() is False               # 无 server -> 失败但不抛
        router5 = _Router()

        class _PS:
            instance = types.SimpleNamespace(routes=router5)

        sys.modules["server"] = types.SimpleNamespace(PromptServer=_PS)
        try:
            assert r3.ensure_registered() is True            # 运行期兜底注册成功
            assert set(router5.table) == FULL_ROUTE_SET
            assert r3.ensure_registered() is True            # 已注册 -> 短路
            assert len(router5.table) == len(FULL_ROUTE_SET)
        finally:
            del sys.modules["server"]


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
            from ComfyUI_H3_SeamlessChain import routes as r4
            assert r4.register() is True
            # 路由只挂到了 app.router（router3），未挂到 RouteTableDef（router4）
            assert set(router3.table) == FULL_ROUTE_SET
            assert router4.table == {}
            resp = asyncio.run(router3.table["/h3chain/delete_project"](_Req({"dir": "nope"})))
            assert resp["status"] == 404
        finally:
            del sys.modules["server"]


class _RouterAddPost:
    """模拟 aiohttp 的 UrlDispatcher（app.router）：add_get/add_post 方法。"""
    def __init__(self):
        self.table = {}

    def add_get(self, path, handler):
        self.table[path] = handler

    def add_post(self, path, handler):
        self.table[path] = handler


def test_add_routes_on_app_router():
    """add_routes 应支持 UrlDispatcher（app.router）的 add_get/add_post，而非仅装饰器。

    此前用 @routes.post 装饰器，对 app.router 会抛 AttributeError 并回退到未装载的
    RouteTableDef，导致 404。此测试保证 app.router 路径真正挂上 handler。
    """
    with _server_env():
        router = _RouterAddPost()
        from ComfyUI_H3_SeamlessChain import routes
        routes.add_routes(router)
        assert set(router.table) == FULL_ROUTE_SET
        # handler 真实可用：缺参 400
        assert asyncio.run(router.table["/h3chain/delete"](_Req({})))["status"] == 400
        # 不存在的项目 404
        assert asyncio.run(router.table["/h3chain/delete_project"](_Req({"dir": "nope"})))["status"] == 404
        # 同一目标重复调用不重复挂载（aiohttp 会因重复路由抛错）
        routes.add_routes(router)
        assert len(router.table) == len(FULL_ROUTE_SET)


def test_api_prefix_copies():
    """新版前端 fetchApi 强制 /api 前缀：每条根路径必须有等价的 /api 副本且行为一致。"""
    with _server_env():
        router = _RouterAddPost()
        from ComfyUI_H3_SeamlessChain import routes
        routes.add_routes(router)
        assert set(router.table) == FULL_ROUTE_SET
        # /api 副本挂的是同一套 handler：ping 双前缀行为一致
        root = asyncio.run(router.table["/h3chain/ping"](_Req()))
        via_api = asyncio.run(router.table["/api/h3chain/ping"](_Req()))
        assert root["json"]["ok"] is True and via_api["json"]["ok"] is True
        assert root["json"]["routes"] == via_api["json"]["routes"]
        # POST 路由的 /api 副本同样可用（缺参 400）
        assert asyncio.run(router.table["/api/h3chain/delete"](_Req({})))["status"] == 400


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
