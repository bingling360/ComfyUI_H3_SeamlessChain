"""游戏式项目存档单测：python tests/test_projects.py（无需 torch / ComfyUI）。

覆盖：list_projects 磁盘扫描（正常/空根/无 manifest/坏 manifest/排序/封面）、
read_project、delete_project（删文件夹+当前链指针清理+穿越防护）。
"""
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


@contextlib.contextmanager
def _env():
    """临时 output 目录（folder_paths stub 动态生效，checkpoint 为延迟导入）。"""
    out = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(get_output_directory=lambda: out)
    try:
        yield out
    finally:
        del sys.modules["folder_paths"]
        shutil.rmtree(out)


def _mk_project(out, name, manifest):
    pdir = os.path.join(out, "h3_projects", name)
    os.makedirs(pdir)
    with open(os.path.join(pdir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False)
    return pdir


def test_list_projects():
    with _env() as out:
        from ComfyUI_H3_SeamlessChain import projects
        assert projects.list_projects() == []                      # 空根（目录不存在）

        _mk_project(out, "甲", {
            "schema": "h3seamless/ckpt-v3", "title": "甲项目", "done": 2, "total": 5,
            "updated_at": 200.0, "thumbs": ["thumb_000.png"], "finals": ["final_a.mp4"],
            "params": {"width": 864}})
        b = _mk_project(out, "乙", {
            "schema": "h3seamless/ckpt-v3", "done": 1, "total": 3, "updated_at": 100.0,
            "thumbs": ["thumb_000.png"]})
        open(os.path.join(b, "thumb_000.png"), "wb").close()       # 乙的封面文件存在
        os.makedirs(os.path.join(out, "h3_projects", "no_manifest"))   # 无 manifest -> 跳过
        c = _mk_project(out, "坏档", {"title": "x"})               # 有 manifest 但缺进度键
        with open(os.path.join(c, "manifest.json"), "w", encoding="utf-8") as f:
            f.write("{broken")                                     # 坏 JSON -> 跳过
        open(os.path.join(out, "h3_projects", "h3chain_state.json"), "w").close()  # 文件非目录 -> 跳过

        lst = projects.list_projects()
        assert [p["dir"] for p in lst] == ["甲", "乙"]              # 按 updated_at 倒序
        a0 = lst[0]
        assert a0["title"] == "甲项目"
        assert a0["done"] == 2 and a0["total"] == 5
        assert a0["finals"] == ["final_a.mp4"] and a0["params"] == {"width": 864}
        assert a0["cover"] == ""                                   # thumb 声明了但文件不存在
        assert lst[1]["title"] == "乙"                             # title 缺省回退目录名
        assert lst[1]["cover"] == "thumb_000.png"
        assert lst[1]["finals"] == [] and lst[1]["params"] == {}


def test_read_project():
    with _env() as out:
        from ComfyUI_H3_SeamlessChain import projects
        _mk_project(out, "甲", {"schema": "h3seamless/ckpt-v3", "done": 1})
        mf = projects.read_project("甲")
        assert isinstance(mf, dict) and mf["done"] == 1
        assert projects.read_project("nope") is None               # 不存在
        for bad in ("", "../up", "a/b", None, ".hidden"):
            assert projects.read_project(bad) is None              # 非法名不触文件系统


def test_create_project():
    """新建项目当场落盘（游戏存档槽）：文件夹 + 0 段初始 manifest，列表立即可见。"""
    with _env() as out:
        from ComfyUI_H3_SeamlessChain import checkpoint, projects
        assert projects.create_project("../up") is None            # 非法名 -> 拒绝
        assert projects.create_project("") is None

        mf = projects.create_project("新链")
        assert isinstance(mf, dict)
        pdir = os.path.join(out, "h3_projects", "新链")
        assert os.path.isdir(pdir)                                  # 文件夹立即出现
        assert mf["schema"] == checkpoint.SCHEMA                    # 首跑 schema 校验能过
        assert mf["done"] == 0 and mf["total"] == 0
        assert mf["params"] == {}                                   # 空旧档：assert_match 视为一致
        assert mf["title"] == "新链" and mf["created_at"] > 0

        # 列表立即包含它（此前要等首段采样完 manifest 落盘后才可见）
        lst = projects.list_projects()
        assert [p["dir"] for p in lst] == ["新链"]
        assert lst[0]["done"] == 0 and lst[0]["total"] == 0

        # 幂等：再次创建不报错不重写（已有 manifest 原样返回）
        mf2 = projects.create_project("新链")
        assert mf2["created_at"] == mf["created_at"]

        # 对正式项目（跑过一段）同样幂等：进度不被清零
        _mk_project(out, "老档", {"schema": "h3seamless/ckpt-v3", "done": 3, "total": 8})
        mf3 = projects.create_project("老档")
        assert mf3["done"] == 3 and mf3["total"] == 8


def test_save_prompts():
    """提示词回写：项目的提示词持久源=项目文件夹（切换/新建/编辑时落盘）。"""
    with _env() as out:
        from ComfyUI_H3_SeamlessChain import projects

        # 不存在/非法输入 -> None，不触磁盘
        assert projects.save_prompts("nope", ["a"]) is None
        assert projects.save_prompts("../up", ["a"]) is None
        assert projects.save_prompts("甲", "not-a-list") is None

        created = projects.create_project("甲")
        mf = projects.save_prompts("甲", ["镜头一", "镜头二"])
        assert mf["prompts"] == ["镜头一", "镜头二"]
        assert mf["total"] == 2 and mf["done"] == 0
        assert mf["title"] == created["title"]              # title/created_at 继承
        assert mf["created_at"] == created["created_at"]
        assert mf["updated_at"] >= created["updated_at"]

        # 只动 prompts/total/updated_at：done/params/哈希/成片清单原样保留
        _mk_project(out, "乙", {
            "schema": "h3seamless/ckpt-v3", "done": 3, "total": 3,
            "params": {"width": 864}, "prompt_hashes": ["h1", "h2", "h3"],
            "finals": ["final_a.mp4"], "title": "乙", "created_at": 1.0, "updated_at": 1.0,
            "prompts": ["旧1", "旧2", "旧3"]})
        mf2 = projects.save_prompts("乙", ["旧1", "新2", "旧3", "加一段"])
        assert mf2["prompts"] == ["旧1", "新2", "旧3", "加一段"]
        assert mf2["total"] == 4 and mf2["done"] == 3       # 改词不截 done（运行时按哈希判定）
        assert mf2["prompt_hashes"] == ["h1", "h2", "h3"]   # 哈希不动 -> 重做判定语义不变
        assert mf2["params"] == {"width": 864} and mf2["finals"] == ["final_a.mp4"]

        # 段数缩水 -> done 钳制到新 total（与 reroll_start 的 min 语义一致）
        mf3 = projects.save_prompts("乙", ["旧1"])
        assert mf3["total"] == 1 and mf3["done"] == 1

        # 序章项目：自动补回占位头，与运行时写盘格式一致
        _mk_project(out, "丙", {
            "schema": "h3seamless/ckpt-v3", "done": 2, "total": 3, "has_prologue": True,
            "prompts": ["「序章（上传视频）」", "正片1", "正片2"]})
        mf4 = projects.save_prompts("丙", ["正片1改", "正片2"])
        assert mf4["prompts"] == ["「序章（上传视频）」", "正片1改", "正片2"]
        assert mf4["total"] == 3

        # 回写后 list_projects 立即反映新进度
        lst = projects.list_projects()
        by_dir = {p["dir"]: p for p in lst}
        assert by_dir["乙"]["total"] == 1 and by_dir["乙"]["done"] == 1


def test_delete_project():
    with _env() as out:
        from ComfyUI_H3_SeamlessChain import projects
        _mk_project(out, "甲", {"done": 1})
        open(os.path.join(out, "h3_projects", "甲", "seg_000.mp4"), "wb").close()
        state = os.path.join(out, "h3_projects", "h3chain_state.json")
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"dir": "甲"}')

        assert projects.delete_project("甲") == "h3_projects/甲"
        assert not os.path.exists(os.path.join(out, "h3_projects", "甲"))
        assert not os.path.exists(state)                            # 当前链指针一并清理

        # 删除别的项目不动别人指针；不存在的项目返回 None
        _mk_project(out, "乙", {"done": 0})
        with open(state, "w", encoding="utf-8") as f:
            f.write('{"dir": "乙"}')
        assert projects.delete_project("nope") is None
        assert os.path.exists(state)
        assert projects.delete_project("../up") is None            # 穿越防护
        assert projects.delete_project("h3chain_state.json") is None

        assert projects.delete_project("乙") == "h3_projects/乙"
        assert not os.path.exists(state)                            # 乙是当前链，指针也清了


def test_safe_name():
    from ComfyUI_H3_SeamlessChain import projects
    for ok in ("链A", "h3_projects_x", "20260818_120000", "a"):
        assert projects.safe_name(ok) == ok
    for bad in ("", "  ", ".", "..", "a/b", "a\\b", "c:d", ".hidden", "../up", None):
        assert projects.safe_name(bad) == ""


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
