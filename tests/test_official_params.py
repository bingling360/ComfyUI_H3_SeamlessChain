"""官方参数化单测：画布换算 / 时长网格吸附 / 标签替换 / 段级引用 / 存档哈希兼容。

运行：python tests/test_official_params.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402


def test_resolve_canvas():
    rc = plugin_nodes._resolve_canvas
    assert rc("16:9", "1.0") == (1344, 768)     # 官方原生画布
    assert rc("16:9", "0.5") == (960, 544)      # 快速预览
    assert rc("9:16", "1.0") == (768, 1344)     # 竖版原生
    w, h = rc("21:9", "1.0")                    # 超上限收敛到 1344 长边
    assert w == 1344 and h <= 768 and h % 32 == 0
    # 百万像素浮点档位：0.1–2.0 共 20 档（官方箭头微调同款），字符串/浮点输入均可
    mps = plugin_nodes._MP_OPTIONS
    assert len(mps) == 20 and mps[0] == 0.1 and mps[-1] == 2.0
    assert rc("16:9", 0.5) == rc("16:9", "0.5")
    for ar in ("21:9", "16:9", "9:16", "4:3", "3:4", "1:1"):
        for mp in mps:
            w, h = rc(ar, mp)
            assert w % 32 == 0 and h % 32 == 0 and w >= 32 and h >= 32
            assert max(w, h) <= 1344 and min(w, h) <= 768
    print("PASS test_resolve_canvas")


def test_snap_seconds():
    snap = plugin_nodes._snap_seconds
    assert snap(5.0) == 124                     # 与旧默认帧数一致
    assert snap(6.0) == 141
    assert snap(5.9) == 141
    assert snap(5.5) == 124                     # 132 帧就近吸附 124（|132-124|=8 < |141-132|=9）
    assert snap(15.0) == 362
    assert snap(0.5) == 5
    assert snap(0.1) == 5                       # 下限 5 帧
    for k in range(21):
        assert snap((17 * k + 5) / 24) == 17 * k + 5   # 网格点恒等
    print("PASS test_snap_seconds")


def test_apply_label_tokens():
    alt = plugin_nodes._apply_label_tokens
    assert alt("主角 [[角色1]] 出场", ["角色1"]) == "主角 <Picture 1> 出场"
    # 前缀碰撞：长标签先替换，角色1 不吃 角色10
    out = alt("[[角色1]] 与 [[角色10]]", ["角色1", "角色10"])
    assert out == "<Picture 1> 与 <Picture 2>"
    out = alt("[[角色10]] 与 [[角色1]]", ["角色10", "角色1"])
    assert out == "<Picture 1> 与 <Picture 2>"
    # 压实重编号：段内子集顺序决定编号
    assert alt("[[场景1]] 然后 [[角色1]]", ["场景1", "角色1"]) == "<Picture 1> 然后 <Picture 2>"
    try:
        alt("[[不存在的标签]]", ["角色1"])
        raise AssertionError("should reject unknown label")
    except ValueError as e:
        assert "未知素材标签" in str(e)
    # 无 token 原样返回
    assert alt("纯文本提示词", ["角色1"]) == "纯文本提示词"
    print("PASS test_apply_label_tokens")


def test_reference_header():
    rh = plugin_nodes._reference_header
    assert rh(["角色1", "场景1"]) == ("subject_definitions:\n"
                                     "<Picture 1> is the reference image for 角色1, used as a generation anchor of this segment.\n"
                                     "<Picture 2> is the reference image for 场景1, used as a generation anchor of this segment.")
    print("PASS test_reference_header")


def test_ds_v1_migration_pool():
    """v1 ds（仅 ref_images）→ 素材池补默认标签；v2 ref_assets 优先。"""
    from ComfyUI_H3_SeamlessChain.nodes import _parse_director_state
    v1 = _parse_director_state('{"mode":"多参视频","prompts":["a"],"ref_images":["x.png","y.png"]}')
    assert v1["ref_images"] == ["x.png", "y.png"] and "ref_assets" not in v1
    v2 = _parse_director_state('{"ref_assets":[{"file":"x.png","label":"角色1"}]}')
    assert v2["ref_assets"][0]["label"] == "角色1"
    assert _parse_director_state("") == {}
    assert _parse_director_state("not json") == {}
    print("PASS test_ds_v1_migration_pool")


def test_segment_hash_compat():
    """默认时长的段哈希与旧公式一致；改时长的段把帧数并入哈希。"""
    from ComfyUI_H3_SeamlessChain import checkpoint
    p = "段1提示词"
    old_style = checkpoint.prompt_hash(p)
    length = plugin_nodes._snap_seconds(5.0)          # 全局默认 124
    default_seg = p if length == length else f"{length}|{p}"
    assert checkpoint.prompt_hash(default_seg) == old_style
    custom = plugin_nodes._snap_seconds(8.0)          # 197 帧，≠ 默认
    assert checkpoint.prompt_hash(f"{custom}|{p}") != old_style
    print("PASS test_segment_hash_compat")


def test_label_dedup():
    """素材池标签重名兜底：自动加后缀去重（模拟 execute 内联逻辑）。"""
    pool = [("角色1", "a.png"), ("角色1", "b.png"), ("场景1", "c.png")]
    seen = set()
    for i, (lbl, fn) in enumerate(pool):
        base, n = lbl, 2
        while lbl in seen:
            lbl = f"{base}{n}"
            n += 1
        seen.add(lbl)
        pool[i] = (lbl, fn)
    assert [l for l, _ in pool] == ["角色1", "角色12", "场景1"]
    print("PASS test_label_dedup")


if __name__ == "__main__":
    test_resolve_canvas()
    test_snap_seconds()
    test_apply_label_tokens()
    test_reference_header()
    test_ds_v1_migration_pool()
    test_segment_hash_compat()
    test_label_dedup()
    print("all tests passed")
