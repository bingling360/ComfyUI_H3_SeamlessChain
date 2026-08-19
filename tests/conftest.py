"""pytest 合集运行环境：stub 统一安装与逐测试恢复。

背景：各测试文件按「单文件脚本」设计（python tests/test_xxx.py 时自行
_install_stubs）。合集运行（pytest tests/）时模块收集顺序按字母序，
stub 依赖收集期副作用——且 test_checkpoint/test_saver/test_routes 的
临时环境上下文退出时会 del sys.modules["folder_paths"] 甚至整个包，
导致后续测试拿到残缺的 stub 环境。

本 conftest 在收集任何测试模块前安装一次 stub，并在每个测试结束后
幂等重装（_install_stubs 可重入），使合集与单跑行为一致。
真 torch 环境下 _install_stubs 自动跳过 torch stub（见 test_node_structure），
真 torch 测试（metrics/qc/refine/smart_cut/seam_doctor）正常跑真逻辑。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)


@pytest.fixture(autouse=True)
def _restore_stubs_after_each_test():
    """每个测试结束后恢复 stub 全集（防 del sys.modules 泄漏到后续测试）。"""
    yield
    _install_stubs(with_audio_support=False)
