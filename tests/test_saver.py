"""成片保存节点单测（stub 纯逻辑）：python tests/test_saver.py

覆盖：输出目录清洗、历史索引原子读改写（坏 JSON 容错 / 新条目插最前）、
分段复制（state 指针 -> segs/，穿越名拒绝）、编码失败显式报错、schema 结构。
真实编码走 PyAV（ComfyUI 自带），单测只验证失败路径与目录逻辑。
"""
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


@contextlib.contextmanager
def _output_env():
    """stub folder_paths -> 临时 output 目录（saver_dir / checkpoints_root 都吃它）。"""
    out = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(get_output_directory=lambda: out)
    try:
        yield out
    finally:
        del sys.modules["folder_paths"]
        shutil.rmtree(out)


def test_saver_dir():
    from ComfyUI_H3_SeamlessChain import saver
    with _output_env() as out:
        assert saver.saver_dir("") == os.path.join(out, "h3_chain")     # 空前缀回落默认名
        assert saver.saver_dir("  ") == os.path.join(out, "h3_chain")
        assert saver.saver_dir("a/b\\c.d") == os.path.join(out, "a_b_c_d")  # 分隔符/点拒进路径
        for name in ("", "a/b\\c.d", "我的链"):
            assert os.path.isdir(saver.saver_dir(name))                 # 目录惰性建好


def test_append_history():
    from ComfyUI_H3_SeamlessChain import saver
    with _output_env() as out:
        saver.append_history(out, {"file": "a.mp4"})
        saver.append_history(out, {"file": "b.mp4"})
        path = os.path.join(out, "h3saver_history.json")
        with open(path, "r", encoding="utf-8") as f:
            assert [e["file"] for e in json.load(f)] == ["b.mp4", "a.mp4"]  # 新条目插最前
        with open(path, "w", encoding="utf-8") as f:
            f.write("{broken")                       # 写入中途损坏 -> 从空重建，不抛错
        saver.append_history(out, {"file": "c.mp4"})
        with open(path, "r", encoding="utf-8") as f:
            assert [e["file"] for e in json.load(f)] == ["c.mp4"]
        assert not [f for f in os.listdir(out) if f.startswith(".tmp_")]  # 无残留临时文件


def test_copy_segments():
    from ComfyUI_H3_SeamlessChain import saver, checkpoint
    with _output_env() as out:
        assert saver.copy_segments(out) == []                 # 无 state 指针 -> 空列表不炸
        arch = os.path.join(out, "checkpoints", "chainA")
        os.makedirs(arch)
        for fn in ("seg_000.mp4", "seg_001.mp4", "manifest.json", "thumb_000.png"):
            open(os.path.join(arch, fn), "wb").close()
        checkpoint.save_state({"dir": "chainA", "total": 2, "done": 2})
        assert saver.copy_segments(out) == ["segs/seg_000.mp4", "segs/seg_001.mp4"]  # 相对输出根
        segs = os.path.join(out, "segs")
        assert os.path.isfile(os.path.join(segs, "seg_000.mp4"))          # 只拷 seg_*.mp4
        assert not os.path.exists(os.path.join(segs, "manifest.json"))
        for evil in ("../evil", "a/b", ".."):
            checkpoint.save_state({"dir": evil, "total": 1, "done": 1})
            assert saver.copy_segments(out) == []             # 穿越 / 多级名一律拒绝
        checkpoint.save_state({"dir": "no_such_archive", "total": 1, "done": 1})
        assert saver.copy_segments(out) == []                 # 目录不存在 -> 空


def test_execute_encoding_failure():
    from ComfyUI_H3_SeamlessChain import saver
    with _output_env() as out:
        try:
            saver.H3ChainSaver.execute(图像=object(),                    # 无 detach -> 编码必失败
                                       音频={"waveform": None, "sample_rate": 44100},
                                       帧率=24, 输出前缀="t", CRF=20)
            raise AssertionError("should raise on encoding failure")
        except RuntimeError as e:
            assert "成片编码失败" in str(e)                    # 显式报错而非静默空输出
        assert not [f for f in os.listdir(os.path.join(out, "t"))
                    if f.startswith("final_")]                # 失败不留半成品


def test_schema():
    from ComfyUI_H3_SeamlessChain import saver
    schema = saver.H3ChainSaver.define_schema()
    assert [inp.id for inp in schema.inputs] == ["图像", "音频", "帧率", "输出前缀", "CRF"]
    assert [o.id for o in schema.outputs] == ["成片文件名"]
    assert schema.kwargs.get("is_output_node") is True        # 输出节点（无下游也执行）
    by_id = {inp.id: inp for inp in schema.inputs}
    assert by_id["帧率"].kwargs.get("default") == 24
    assert by_id["CRF"].kwargs.get("default") == 20 and by_id["CRF"].kwargs.get("min") == 0
    assert by_id["输出前缀"].kwargs.get("default") == "h3_chain"


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
