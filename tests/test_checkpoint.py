"""断点模块纯逻辑单测：python tests/test_checkpoint.py（无需 torch / ComfyUI）。"""
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import checkpoint


def test_fingerprint_stable():
    p = {"prompts": ["a", "b"], "width": 864, "height": 480,
         "length": 124, "ctx": 22, "steps": 25}
    assert checkpoint.fingerprint(p) == checkpoint.fingerprint(dict(reversed(list(p.items()))))
    assert len(checkpoint.fingerprint(p)) == 8
    p2 = dict(p, prompts=["a", "c"])                     # 提示词变了 -> 指纹必须变
    assert checkpoint.fingerprint(p) != checkpoint.fingerprint(p2)


def test_manifest_roundtrip():
    root = tempfile.mkdtemp()
    try:
        assert checkpoint.load_manifest(root) is None    # 空目录
        m = {"schema": checkpoint.SCHEMA, "done": 3,
             "params": {"prompts": ["甲", "乙"], "seed": 7}, "trims": [0, 17, 0]}
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


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
