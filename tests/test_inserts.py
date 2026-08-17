"""插入清单解析与统一段落计划纯逻辑单测：python tests/test_inserts.py（无需 torch / ComfyUI）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import inserts


def test_parse_inserts():
    assert inserts.parse_inserts("", 3) == []
    assert inserts.parse_inserts("\n  # 注释行\n\n", 3) == []
    assert inserts.parse_inserts("2|ad.mp4", 3) == [(2, "ad.mp4")]
    assert inserts.parse_inserts("3|b.mp4\n1|a.mp4", 3) == [(1, "a.mp4"), (3, "b.mp4")]  # 自动升序
    assert inserts.parse_inserts("2|sub/dir/v.mp4", 3) == [(2, "sub/dir/v.mp4")]         # 子目录
    assert inserts.parse_inserts("4|tail.mp4", 3) == [(4, "tail.mp4")]                   # 尾插合法


def test_parse_inserts_errors():
    for spec, word in [
        ("xx", "格式"),            # 缺 |
        ("2|", "格式"),            # 空文件名
        ("|a.mp4", "整数"),        # 空位置
        ("a|b.mp4", "整数"),       # 非整数位置
        ("0|a.mp4", "≥ 1"),
        ("-1|a.mp4", "≥ 1"),
        ("2|a.mp4\n2|b.mp4", "重复"),
        ("9|a.mp4", "超出范围"),   # 3 段提示词，首插位置上限 4
    ]:
        try:
            inserts.parse_inserts(spec, 3)
            raise AssertionError(f"should reject {spec!r}")
        except ValueError as e:
            assert word in str(e), (spec, str(e))


def test_build_plan():
    plan = inserts.build_plan(["p1", "p2", "p3"], [])
    assert [it["kind"] for it in plan] == ["prompt"] * 3
    assert [it["i"] for it in plan] == [0, 1, 2]           # 提示词原始下标（种子规则沿用）

    plan = inserts.build_plan(["p1", "p2"], [(1, "head.mp4")])   # 片头
    assert [(it["kind"], it.get("file") or it.get("text")) for it in plan] == \
        [("insert", "head.mp4"), ("prompt", "p1"), ("prompt", "p2")]

    plan = inserts.build_plan(["p1", "p2", "p3"], [(2, "ad.mp4")])  # 中插
    assert [it["kind"] for it in plan] == ["prompt", "insert", "prompt", "prompt"]

    plan = inserts.build_plan(["p1", "p2"], [(1, "a.mp4"), (4, "z.mp4")])  # 片头 + 尾插
    assert [it["kind"] for it in plan] == ["insert", "prompt", "prompt", "insert"]

    # 画布「起始视频」= 位置 1 的 (1, None) 插入项（nodes.py 约定）
    plan = inserts.build_plan(["p1"], [(1, None)])
    assert plan[0]["kind"] == "insert" and plan[0]["file"] is None and plan[0]["pos"] == 1


def test_insert_hash():
    a = inserts.insert_hash("a.mp4", 124, 0.5, 0.1)
    assert a == inserts.insert_hash("a.mp4", 124, 0.5, 0.1)
    assert a != inserts.insert_hash("a.mp4", 124, 0.6, 0.1)   # 换内容 -> 其后段重做
    assert a != inserts.insert_hash("b.mp4", 124, 0.5, 0.1)   # 换文件名 -> 重做
    assert len(a) == 8


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
