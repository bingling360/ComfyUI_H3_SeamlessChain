"""多媒体素材池单测：三类素材 token 压实 / 参考头 / 段级子集编号 / 官方上限。

运行：python tests/test_assets_kinds.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402


def test_ref_caps_official():
    assert plugin_nodes.REF_CAPS == {"image": 9, "video": 3, "audio": 3}
    assert plugin_nodes._KIND_NAME == {"image": "图片", "video": "视频", "audio": "音频"}
    print("PASS test_ref_caps_official")


def test_normalize_order():
    no = plugin_nodes._normalize_order
    # 旧格式纯字符串 -> image（兼容旧调用/旧测试）
    assert no(["角色1", "场景1"]) == [("image", "角色1"), ("image", "场景1")]
    # (kind, 标签) 原样保留
    assert no([("video", "视频1"), ("audio", "音频1")]) == [("video", "视频1"), ("audio", "音频1")]
    print("PASS test_normalize_order")


def test_kind_tokens_independent_numbering():
    kt = plugin_nodes._kind_tokens
    order = [("image", "角色1"), ("video", "片头"), ("image", "场景1"),
             ("audio", "BGM"), ("video", "转场")]
    m = kt(order)
    assert m["角色1"] == "<Picture 1>" and m["场景1"] == "<Picture 2>"
    assert m["片头"] == "<Video 1>" and m["转场"] == "<Video 2>"
    assert m["BGM"] == "<Audio 1>"
    try:
        kt([("gif", "未知类")])
        raise AssertionError("should reject unknown kind")
    except ValueError as e:
        assert "未知素材类别" in str(e)
    print("PASS test_kind_tokens_independent_numbering")


def test_apply_label_tokens_mixed_kinds():
    alt = plugin_nodes._apply_label_tokens
    order = [("image", "角色1"), ("video", "视频1"), ("audio", "音频1")]
    out = alt("[[角色1]] 走过 [[视频1]] 的场景，配乐 [[音频1]]", order)
    assert out == "<Picture 1> 走过 <Video 1> 的场景，配乐 <Audio 1>"
    # 旧格式（纯标签列表）仍按 image 处理
    assert alt("主角 [[角色1]] 出场", ["角色1"]) == "主角 <Picture 1> 出场"
    # 长标签先替换：图片1 不吃 图片10
    assert alt("[[图片1]] 与 [[图片10]]", ["图片1", "图片10"]) == "<Picture 1> 与 <Picture 2>"
    # 未知标签报错
    try:
        alt("[[不存在的]]", order)
        raise AssertionError("should reject unknown label")
    except ValueError as e:
        assert "未知素材标签" in str(e)
    print("PASS test_apply_label_tokens_mixed_kinds")


def test_reference_header_mixed():
    rh = plugin_nodes._reference_header
    order = [("image", "角色1"), ("video", "视频1"), ("audio", "音频1")]
    assert rh(order) == ("subject_definitions:\n"
                         "<Picture 1> is the reference image for 角色1, used as a generation anchor of this segment.\n"
                         "<Video 1> is the reference video for 视频1, used as a generation anchor of this segment.\n"
                         "<Audio 1> is the reference audio for 音频1, used as a generation anchor of this segment.")
    assert rh(["场景1"]) == ("subject_definitions:\n"
                             "<Picture 1> is the reference image for 场景1, used as a generation anchor of this segment.")
    print("PASS test_reference_header_mixed")


def test_per_segment_subset_renumber():
    """不同段勾选不同子集：编号按各自段的顺序独立压实。"""
    alt = plugin_nodes._apply_label_tokens
    rh = plugin_nodes._reference_header
    pool = [("image", "角色1"), ("image", "场景1"), ("video", "视频1"), ("audio", "BGM")]
    # 段1 只用 角色1 + BGM；段2 只用 场景1 + 视频1
    seg1 = [("image", "角色1"), ("audio", "BGM")]
    seg2 = [("image", "场景1"), ("video", "视频1")]
    assert alt("[[角色1]] 配 [[BGM]]", seg1) == "<Picture 1> 配 <Audio 1>"
    assert alt("[[场景1]] 引用 [[视频1]]", seg2) == "<Picture 1> 引用 <Video 1>"
    # 提示词提到但未勾选的素材由 execute 并入 order（此处直接验证并入后的头行）
    assert rh(seg1) == ("subject_definitions:\n"
                        "<Picture 1> is the reference image for 角色1, used as a generation anchor of this segment.\n"
                        "<Audio 1> is the reference audio for BGM, used as a generation anchor of this segment.")
    assert len(pool) == 4
    print("PASS test_per_segment_subset_renumber")


def test_ds_ref_assets_kinds_roundtrip():
    """v2 状态：ref_assets 三类混排解析；bogus kind 由 execute 归一为 image（前端同规则）。"""
    from ComfyUI_H3_SeamlessChain.nodes import _parse_director_state
    raw = ('{"mode":"多参视频","prompts":["p1"],'
           '"ref_assets":[{"file":"a.png","label":"角色1","kind":"image"},'
           '{"file":"v.mp4","kind":"video"},{"file":"s.mp3","kind":"audio"},'
           '{"file":"b.png","kind":"bogus"}],'
           '"segments":[{"scene_prompt":"夜景","character_prompt":"主角","seconds":6.5,'
           '"refs":["角色1","BGM"]}]}')
    ds = _parse_director_state(raw)
    a = ds["ref_assets"]
    assert [x["kind"] for x in a] == ["image", "video", "audio", "bogus"]
    assert ds["segments"][0]["refs"] == ["角色1", "BGM"]
    assert ds["segments"][0]["seconds"] == 6.5
    # execute 内联归一：bogus -> image（与前端 getDs 一致）
    kinds = []
    for item in a:
        k = str(item.get("kind") or "image")
        kinds.append(k if k in plugin_nodes._KIND_NAME else "image")
    assert kinds == ["image", "video", "audio", "image"]
    # 缺省标签按类别编号，只对未命名素材计数（execute 内联规则，与前端 getDs 一致）
    n = {"image": 0, "video": 0, "audio": 0}
    labels = []
    for item in a:
        k = str(item.get("kind") or "image")
        k = k if k in plugin_nodes._KIND_NAME else "image"
        label = str(item.get("label") or "").strip()
        if not label:
            n[k] += 1
            label = f"{plugin_nodes._KIND_NAME[k]}{n[k]}"
        labels.append(label)
    assert labels == ["角色1", "视频1", "音频1", "图片1"]
    print("PASS test_ds_ref_assets_kinds_roundtrip")


def test_segment_order_dedup_and_unknown():
    """段引用顺序去重 + 未知标签报错（execute 内联规则的行为样本）。"""
    pool_labels = ["角色1", "场景1", "视频1"]
    pool_kind = {"角色1": "image", "场景1": "image", "视频1": "video"}
    refs_sel = ["角色1", "角色1", "视频1"]
    order = []
    for r in refs_sel:
        lbl = str(r).strip()
        assert lbl in pool_labels, f"未知标签 {lbl}"
        item = (pool_kind[lbl], lbl)
        if item not in order:
            order.append(item)
    assert order == [("image", "角色1"), ("video", "视频1")]
    print("PASS test_segment_order_dedup_and_unknown")


if __name__ == "__main__":
    test_ref_caps_official()
    test_normalize_order()
    test_kind_tokens_independent_numbering()
    test_apply_label_tokens_mixed_kinds()
    test_reference_header_mixed()
    test_per_segment_subset_renumber()
    test_ds_ref_assets_kinds_roundtrip()
    test_segment_order_dedup_and_unknown()
    print("all tests passed")
