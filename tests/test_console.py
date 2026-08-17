"""控制台节点单测（stub 纯逻辑）：python tests/test_console.py

覆盖：提示词组 join/split 规则（段序 / 空值剔除）、load_upload 三态
（空 / 图片 / 文件缺失报错）、execute 输出字典、schema 结构。
视频序章分支依赖 PyAV，环境缺失时验证报错语义后跳过。
"""
import contextlib
import importlib.util
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插件父目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests 目录

from test_node_structure import _install_stubs


def test_join_prompts():
    from ComfyUI_H3_SeamlessChain import console
    assert console.join_prompts(None) == ""                       # 无输入组
    assert console.join_prompts({}) == ""
    assert console.join_prompts({"提示词_1": " 雪原 ", "提示词_2": "  ",
                                 "提示词_3": "城市"}) == "雪原\n城市"  # 空值剔除 + 去首尾空白
    assert console.join_prompts({"提示词_10": "十", "提示词_2": "二"}) == "二\n十"  # 按数字排序
    assert console.join_prompts({"提示词_1": None, "提示词_2": "b"}) == "b"  # None 剔除


def test_split_prompts():
    from ComfyUI_H3_SeamlessChain import console
    assert console.split_prompts("a\n\n b \nc") == ["a", "b", "c"]  # 空行剔除 + strip
    assert console.split_prompts("") == [] and console.split_prompts(None) == []
    # join -> split 往返稳定（与采样器侧拆分同规则）
    group = {"提示词_1": "a", "提示词_2": "", "提示词_3": "c"}
    assert console.split_prompts(console.join_prompts(group)) == ["a", "c"]


def test_load_upload_empty():
    from ComfyUI_H3_SeamlessChain import console
    assert console.load_upload("") == (None, None, None)
    assert console.load_upload("   ") == (None, None, None)


@contextlib.contextmanager
def _folder_paths_env():
    """stub folder_paths.get_annotated_filepath -> 注记名映射到临时目录。"""
    root = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(
        get_annotated_filepath=lambda name, default_dir=None: os.path.join(root, name))
    try:
        yield root
    finally:
        del sys.modules["folder_paths"]
        shutil.rmtree(root)


def test_load_upload_missing():
    from ComfyUI_H3_SeamlessChain import console
    with _folder_paths_env():
        try:
            console.load_upload("h3chain/nope.png")
            raise AssertionError("should reject missing upload")
        except ValueError as e:
            assert "上传文件不存在" in str(e)                 # 被手动删除属异常态，报错不静默


def test_load_upload_image():
    import torch
    if not hasattr(torch, "from_numpy"):
        return  # stub 环境（无真实 torch）跳过像素级校验
    from PIL import Image
    from ComfyUI_H3_SeamlessChain import console
    with _folder_paths_env() as root:
        path = os.path.join(root, "h3chain", "first.png")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        Image.new("RGB", (8, 6), (255, 0, 0)).save(path)
        first, prologue, audio = console.load_upload("h3chain/first.png")
        assert first is not None and prologue is None and audio is None  # 图片只出首帧
        assert tuple(first.shape) == (1, 6, 8, 3)                        # 与官方 LoadImage 同形
        assert abs(float(first[0, 0, 0, 0]) - 1.0) < 1e-4                # 红通道 ≈ 1.0


def test_load_upload_video_needs_pyav():
    if importlib.util.find_spec("av") is not None:
        return  # 有 PyAV 的环境走真实解码，跳过缺依赖语义测试
    from ComfyUI_H3_SeamlessChain import console
    with _folder_paths_env() as root:
        path = os.path.join(root, "h3chain", "prologue.mp4")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "wb").close()
        try:
            console.load_upload("h3chain/prologue.mp4")
            raise AssertionError("should require PyAV")
        except RuntimeError as e:
            assert "PyAV" in str(e)                         # 明确报缺依赖，不静默降级


def test_execute():
    from ComfyUI_H3_SeamlessChain import console
    out = console.H3ChainConsole.execute(
        存档名="  链A  ", 首帧文件="",
        提示词组={"提示词_1": "a", "提示词_2": " ", "提示词_3": "b"})
    assert out["存档名"] == "链A"                            # passthrough 且去空白
    assert out["提示词清单"] == "a\nb"
    assert out["首帧"] is None and out["序章视频"] is None and out["序章音轨"] is None


def test_schema():
    from ComfyUI_H3_SeamlessChain import console
    schema = console.H3ChainConsole.define_schema()
    ids = [inp.id for inp in schema.inputs]
    assert ids == ["存档名", "首帧文件", "提示词组"]
    assert [o.id for o in schema.outputs] == ["首帧", "序章视频", "序章音轨", "提示词清单", "存档名"]
    by_id = {inp.id: inp for inp in schema.inputs}
    assert by_id["存档名"].kwargs.get("default") == ""
    assert by_id["首帧文件"].kwargs.get("default") == ""


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    del sys.modules["torch"]  # 换回真实 torch（_load_image 需要 from_numpy；nodes 已随 stub 加载）
    try:
        import torch  # noqa: F401
    except ImportError:
        pass
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
