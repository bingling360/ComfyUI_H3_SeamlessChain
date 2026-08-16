"""断点模块纯逻辑单测：python tests/test_checkpoint.py（无需 torch / ComfyUI）。"""
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checkpoint


def test_fingerprint_stable():
    p = {"width": 864, "height": 480, "length": 124, "ctx": 22, "steps": 25}
    assert checkpoint.fingerprint(p) == checkpoint.fingerprint(dict(reversed(list(p.items()))))
    assert len(checkpoint.fingerprint(p)) == 8
    assert checkpoint.fingerprint(p) != checkpoint.fingerprint({**p, "width": 960})


def test_prompt_hash():
    assert checkpoint.prompt_hash("abc") == checkpoint.prompt_hash("abc")
    assert checkpoint.prompt_hash("abc") != checkpoint.prompt_hash("abd")
    assert checkpoint.prompt_hash("雪花") == checkpoint.prompt_hash("雪花")   # 中文跨进程稳定
    assert len(checkpoint.prompt_hash("abc")) == 8


def test_reroll_start():
    hs = ["a", "b", "c", "d"]
    # 返回 >= done 表示无需重做（调用方仅在返回值 < done 时截断；返回 0 = 整链重做）
    assert checkpoint.reroll_start(hs, list(hs), 4) == 4              # 一致 -> 无需重做
    assert checkpoint.reroll_start(hs, ["a", "b", "X", "d"], 4) == 2  # 改第 3 段 -> 从下标 2
    assert checkpoint.reroll_start(["X"] + hs[1:], hs, 4) == 0        # 改第 1 段 -> 整链重做
    assert checkpoint.reroll_start(hs, ["a", "b"], 4) == 2           # 提示词变少 -> 截到新数量
    assert checkpoint.reroll_start(hs, hs + ["e"], 4) == 4           # 末尾追加 -> 前缀不动
    assert checkpoint.reroll_start([], ["a"], 0) == 0


def test_truncate():
    root = tempfile.mkdtemp()
    try:
        for idx in range(4):
            open(checkpoint.seg_path(root, idx), "wb").close()
        m = {"schema": checkpoint.SCHEMA, "done": 4,
             "seeds": [1, 2, 3, 4], "trims": [0, 0, 17, 0],
             "prompt_hashes": ["a", "b", "c", "d"], "params": {"width": 864},
             "total": 8, "thumbs": ["t0", "t1", "t2", "t3"],
             "prompts": ["p0", "p1", "p2", "p3"],
             "seams": [None, [0.01, -0.5], [0.09, 7.0], None],
             "bridge_scores": [None, 31.5, 20.1, 44.0]}
        out = checkpoint.truncate(root, m, 2)
        assert out["done"] == 2
        assert out["seeds"] == [1, 2] and out["trims"] == [0, 0] and out["prompt_hashes"] == ["a", "b"]
        assert out["params"] == {"width": 864}                       # 原参数不动
        assert out["total"] == 8                                      # 段总数是链属性，不截断
        assert out["thumbs"] == ["t0", "t1"] and out["prompts"] == ["p0", "p1"]
        assert out["seams"] == [None, [0.01, -0.5]] and out["bridge_scores"] == [None, 31.5]
        assert checkpoint.load_manifest(root) == out                 # 截断即时落盘
        for idx in range(2):
            assert os.path.exists(checkpoint.seg_path(root, idx))    # 保留段
        for idx in (2, 3):
            assert not os.path.exists(checkpoint.seg_path(root, idx))  # 被弃段已删
    finally:
        shutil.rmtree(root)


def test_manifest_roundtrip():
    root = tempfile.mkdtemp()
    try:
        assert checkpoint.load_manifest(root) is None    # 空目录
        m = {"schema": checkpoint.SCHEMA, "done": 3,
             "seeds": [101, 102, 103], "trims": [0, 17, 0],
             "prompt_hashes": ["a1b2c3d4", "e5f6a7b8", "c9d0e1f2"],
             "params": {"width": 864, "seed": 7}}
        checkpoint.save_manifest(root, m)
        assert checkpoint.load_manifest(root) == m
        assert not [f for f in os.listdir(root) if f.startswith(".tmp_")]  # 无残留临时文件
    finally:
        shutil.rmtree(root)


def test_assert_match():
    old = {"prompts": ["a"], "width": 864, "gate": {"mode": "标注", "threshold": 30.0, "limit": 34}}
    new = {"prompts": ["a"], "width": 864, "gate": {"mode": "标注", "threshold": 30.0, "limit": 34}}
    checkpoint.assert_match(old, new)                    # 一致 -> 通过
    checkpoint.assert_match({**old, "seed": 7}, new)     # 断点多出的 seed 键不参与比较
    try:
        checkpoint.assert_match(old, {**new, "width": 960})
        raise AssertionError("should reject changed params")
    except ValueError as e:
        assert "width" in str(e)


def test_contiguous_done():
    root = tempfile.mkdtemp()
    try:
        for idx in (0, 1):
            open(checkpoint.seg_path(root, idx), "wb").close()
        assert checkpoint.contiguous_done(root, 2) == 2
        assert checkpoint.contiguous_done(root, 3) == 2   # seg_002 缺失 -> 截断到 2
        assert checkpoint.contiguous_done(root, 0) == 0
    finally:
        shutil.rmtree(root)


def test_state_roundtrip():
    out_dir = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(get_output_directory=lambda: out_dir)
    try:
        assert checkpoint.load_state() is None                 # 尚无状态文件
        state = {"dir": "h3chain_x", "total": 5, "done": 2, "review": True,
                 "reroll": 0, "report": "段1…", "updated_at": 1.5}
        checkpoint.save_state(state)
        assert checkpoint.load_state() == state                 # 原子写后可原样读回
        # 坏 JSON（写入中途损坏）-> None，面板据此显示空态而不是抛错
        with open(os.path.join(checkpoint.checkpoints_root(), "h3chain_state.json"), "w") as f:
            f.write("{broken")
        assert checkpoint.load_state() is None
    finally:
        del sys.modules["folder_paths"]
        shutil.rmtree(out_dir)


def test_save_thumb_graceful():
    # 无 PIL / 非 torch 帧对象 -> 返回空串（面板降级为占位），绝不抛错
    out_dir = tempfile.mkdtemp()
    try:
        assert checkpoint.save_thumb(out_dir, 0, object()) == ""
        assert not os.path.exists(os.path.join(out_dir, "thumb_000.png"))
    finally:
        shutil.rmtree(out_dir)


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
