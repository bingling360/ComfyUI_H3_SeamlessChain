"""选择性重做（重摇标记）纯函数：解析校验/种子 bump/列表合并/尾锚切片。
python tests/test_redo.py
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402

try:
    import torch as _real_torch
    if not hasattr(_real_torch, "zeros"):   # stub 环境视为无 torch
        _real_torch = None
except ImportError:
    _real_torch = None


def test_parse_valid_and_order():
    ds = {"redo_segs": [{"slot": 3, "mode": "双锚"}, {"slot": 1, "mode": "无锚"}]}
    kinds = ["prompt"] * 6
    out = plugin_nodes._parse_redo_segs(ds, 6, kinds, 0)
    assert out == [(3, "双锚"), (1, "无锚")]   # 提交序保留（顺序处理由链序决定）


def test_parse_dedupe_and_bad_entries():
    ds = {"redo_segs": [
        {"slot": 2, "mode": "双锚"},
        {"slot": 2, "mode": "无锚"},          # 重复槽位：只留首个
        {"slot": "x", "mode": "双锚"},        # 非法槽位
        {"mode": "双锚"},                     # 缺槽位
        {"slot": 2},                          # 缺 mode → 默认双锚（但槽位已占，剔除）
        [1, 2],                               # 非对象
    ]}
    out = plugin_nodes._parse_redo_segs(ds, 6, ["prompt"] * 6, 0)
    assert out == [(2, "双锚")]


def test_parse_default_mode():
    out = plugin_nodes._parse_redo_segs({"redo_segs": [{"slot": 0}]}, 3,
                                        ["prompt"] * 3, 0)
    assert out == [(0, "双锚")]


def test_parse_invalid_mode():
    ds = {"redo_segs": [{"slot": 0, "mode": "全锚"}, {"slot": 1, "mode": "双锚"}]}
    out = plugin_nodes._parse_redo_segs(ds, 3, ["prompt"] * 3, 0)
    assert out == [(1, "双锚")]


def test_parse_bounds():
    kinds = ["prompt"] * 4
    # 越界：slot >= done（未完成段本来就要生成）
    assert plugin_nodes._parse_redo_segs({"redo_segs": [{"slot": 4, "mode": "双锚"}]},
                                         4, kinds, 0) == []
    # 序章槽位（slot < off）不可重摇
    assert plugin_nodes._parse_redo_segs({"redo_segs": [{"slot": 0, "mode": "双锚"}]},
                                         4, kinds, 1) == []
    # done=0（无已完成段）：全部剔除
    assert plugin_nodes._parse_redo_segs({"redo_segs": [{"slot": 0, "mode": "双锚"}]},
                                         0, kinds, 0) == []


def test_parse_insert_slot_rejected():
    kinds = ["prompt", "insert", "prompt"]
    out = plugin_nodes._parse_redo_segs({"redo_segs": [{"slot": 1, "mode": "双锚"}]},
                                        3, kinds, 0)
    assert out == []   # 插入段不可重摇（专用操作）


def test_parse_disabled_rejected():
    kinds = ["prompt", "prompt", "prompt"]
    ds = {"redo_segs": [{"slot": 0, "mode": "双锚"}, {"slot": 2, "mode": "无锚"}]}
    # 禁用段 0：重摇标记剔除；段 2 启用正常保留
    out = plugin_nodes._parse_redo_segs(ds, 3, kinds, 0, [True, False, False])
    assert out == [(2, "无锚")]
    # disabled=None（旧调用口径）：不校验禁用，全部保留
    assert plugin_nodes._parse_redo_segs(ds, 3, kinds, 0, None) == \
        [(0, "双锚"), (2, "无锚")]
    # disabled 长度不足：越界位置不视为禁用（防御性，正常不发生）
    assert plugin_nodes._parse_redo_segs(ds, 3, kinds, 0, [False]) == \
        [(0, "双锚"), (2, "无锚")]


def test_parse_empty_and_malformed_ds():
    assert plugin_nodes._parse_redo_segs({}, 5, ["prompt"] * 5, 0) == []
    assert plugin_nodes._parse_redo_segs({"redo_segs": []}, 5, ["prompt"] * 5, 0) == []
    assert plugin_nodes._parse_redo_segs({"redo_segs": "x"}, 5, ["prompt"] * 5, 0) == []
    assert plugin_nodes._parse_redo_segs(None, 5, ["prompt"] * 5, 0) == []
    assert plugin_nodes._parse_redo_segs({"redo_segs": [None, 3, "a"]}, 5,
                                         ["prompt"] * 5, 0) == []


def test_seed_bump_when_equal():
    s = plugin_nodes._redo_seed(100, 2, 102)   # 100+2 == 存档 102 → bump
    assert s == 103
    s = plugin_nodes._redo_seed(100, 2, 999)   # 不相等 → 原样
    assert s == 102


def test_seed_bump_wraparound():
    M = 0xffffffffffffffff
    # s = (M+1) % M = 1（回绕），存档 0 → 不等 → 1
    assert plugin_nodes._redo_seed(M, 1, 0) == 1
    # s = M % M = 0 == 存档 M % M = 0 → bump → 1（回绕后仍保证必变）
    assert plugin_nodes._redo_seed(M, 0, M) == 1


def test_seed_bump_alternates():
    """连续多次重摇同段（前端未随机种子）：bump 交替不出死循环。"""
    seed_ctrl, i, archived = 42, 0, 42
    s1 = plugin_nodes._redo_seed(seed_ctrl, i, archived)   # ==42 → 43
    s2 = plugin_nodes._redo_seed(seed_ctrl, i, s1)         # 43≠42 → 42
    s3 = plugin_nodes._redo_seed(seed_ctrl, i, s2)         # ==42 → 43
    assert [s1, s2, s3] == [43, 42, 43]


def test_seed_bump_archived_none():
    assert plugin_nodes._redo_seed(7, 0, None) == 7
    assert plugin_nodes._redo_seed(7, 3, "bad") == 10      # 非法存档值不 bump


def test_merged_list_fills_tail():
    old = ["a", "b", "c", "d", "e"]
    new = ["A", "B"]
    out = plugin_nodes._merged_list(new, old, 5)
    assert out == ["A", "B", "c", "d", "e"]   # 新前缀 + 旧补尾


def test_merged_list_truncate_when_full():
    old = ["a", "b", "c"]
    new = ["A", "B", "C", "D"]
    assert plugin_nodes._merged_list(new, old, 4) == ["A", "B", "C", "D"]
    assert plugin_nodes._merged_list(new, old, 3) == ["A", "B", "C"]   # upto 截断


def test_merged_list_short_old_safe():
    out = plugin_nodes._merged_list(["A"], ["a", "b"], 5)   # 旧不足补不满（安全）
    assert out == ["A", "b"]


def test_merged_list_empty_inputs():
    assert plugin_nodes._merged_list([], [], 3) == []
    assert plugin_nodes._merged_list(None, ["a", "b"], 2) == ["a", "b"]
    assert plugin_nodes._merged_list(["A"], None, 1) == ["A"]

if _real_torch is not None:   # 尾锚切片需真 torch（stub 环境整体跳过，不注册）

    def test_seam_tail_kf_slice():
        """裁头后首个可见 token 直切（帧归属：帧0→token0，帧f→token (f+3)//4）。"""
        vt = _real_torch.arange(2 * 4 * 20 * 2 * 2,
                                dtype=_real_torch.float32).reshape(2, 4, 20, 2, 2)
        kf = plugin_nodes._seam_tail_kf(vt, 56)   # (56+3)//4 = 14 → token 14
        assert kf.shape == (2, 4, 1, 2, 2)
        assert _real_torch.equal(kf, vt[:, :, 14:15])
        # 裁头 0 → token 0（首帧）
        assert _real_torch.equal(plugin_nodes._seam_tail_kf(vt, 0), vt[:, :, 0:1])
        # 越界（裁头超出 latent 时长）→ None
        assert plugin_nodes._seam_tail_kf(vt, 200) is None

    def test_seam_tail_kf_no_modification():
        vt = _real_torch.zeros(1, 4, 5, 2, 2)
        kf = plugin_nodes._seam_tail_kf(vt, 8)
        kf.fill_(1.0)   # clone 隔离：改 kf 不影响源
        assert float(vt.abs().sum()) == 0.0


def test_truncate_clears_redo_queue():
    """链结构变化（truncate）清空重摇队列。"""
    import tempfile

    from ComfyUI_H3_SeamlessChain import checkpoint

    manifest = {"schema": checkpoint.SCHEMA, "done": 5,
                "seeds": [1, 2, 3, 4, 5], "trims": [0] * 5,
                "prompt_hashes": ["h"] * 5, "thumbs": [], "videos": [],
                "prompts": ["p"] * 5, "seams": [None] * 5,
                "bridge_scores": [None] * 5, "seam_metrics": [None] * 5,
                "params": {}, "redo_queue": [[2, "双锚"], [4, "无锚"]]}
    with tempfile.TemporaryDirectory() as td:
        checkpoint.save_manifest(td, manifest)
        out = checkpoint.truncate(td, manifest, 2)
    assert out.get("redo_queue") == []
    assert out["done"] == 2


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK {len(fns)} tests")
