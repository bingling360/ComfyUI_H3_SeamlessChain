"""段级首尾帧图引用（frame_refs）解析/默认值/哈希规则与头锚注入：python tests/test_frame_refs.py"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402


def test_default_first_last():
    d = plugin_nodes._default_frame_refs
    assert d(0, 3, True, True) == ["首帧图"]
    assert d(1, 3, True, True) == []
    assert d(2, 3, True, True) == ["尾帧图"]
    assert d(0, 1, True, True) == ["首帧图", "尾帧图"]   # 单段链两者都默认（=旧行为）
    assert d(0, 3, False, True) == []                    # 没设首帧图就不引用
    assert d(2, 3, True, False) == []                    # 没设尾帧图就不引用


def test_parse_missing_key_uses_default():
    picked, explicit = plugin_nodes._parse_frame_refs([{}, {}, {}], 3, True, True)
    assert picked == [["首帧图"], [], ["尾帧图"]]
    assert explicit == [False, False, False]


def test_parse_segments_shorter_than_n():
    picked, explicit = plugin_nodes._parse_frame_refs([{}], 3, True, True)
    assert picked == [["首帧图"], [], ["尾帧图"]]
    assert explicit == [False, False, False]


def test_parse_explicit_filters_dedupes_order():
    segs = [{"frame_refs": ["尾帧图", "未知", "首帧图", "尾帧图"]}, {}, {"frame_refs": []}]
    picked, explicit = plugin_nodes._parse_frame_refs(segs, 3, True, True)
    assert picked[0] == ["尾帧图", "首帧图"]   # 未知忽略、去重、保序
    assert picked[2] == []                     # 显式空数组=两图都不参考
    assert explicit == [True, False, True]


def test_parse_non_list_treated_as_unset():
    segs = [{"frame_refs": "首帧图"}, {"frame_refs": None}, {"frame_refs": ["尾帧图"]}]
    picked, explicit = plugin_nodes._parse_frame_refs(segs, 3, True, True)
    assert explicit == [False, False, True]
    assert picked[0] == ["首帧图"]              # 非数组视为未设置 → 默认
    assert picked[2] == ["尾帧图"]


def _hash_flags(picked, explicit, n, has_first, has_end):
    return [explicit[i] and picked[i] != plugin_nodes._default_frame_refs(i, n, has_first, has_end)
            for i in range(n)]


def test_hash_rule_explicit_equal_default_no_rerender():
    # 与 nodes.py 逐段哈希同一条件：显式但等于默认 → 不进哈希（不触发重做）
    segs = [{"frame_refs": ["首帧图"]}, {}, {"frame_refs": ["尾帧图"]}]
    picked, explicit = plugin_nodes._parse_frame_refs(segs, 3, True, True)
    assert _hash_flags(picked, explicit, 3, True, True) == [False, False, False]


def test_hash_rule_explicit_differs_triggers_rerender():
    segs = [{}, {"frame_refs": ["首帧图"]}, {"frame_refs": []}]
    picked, explicit = plugin_nodes._parse_frame_refs(segs, 3, True, True)
    # 中段加首帧头锚、末段关尾帧锚 → 各自触发该段起重做；未设置段不受影响
    assert _hash_flags(picked, explicit, 3, True, True) == [False, True, True]


def test_hash_rule_no_images_explicit_still_differs():
    # 没设图（文生/多参模式）：显式非空勾选仍与空默认不同 → 进哈希。
    # 实际无害：UI 只在有图时显示 chips（无法勾到不存在的图），且换图/撤图
    # 本身改 chain 指纹触发整链重做，此处仅文档化行为
    segs = [{}, {"frame_refs": ["首帧图", "尾帧图"]}, {}]
    picked, explicit = plugin_nodes._parse_frame_refs(segs, 3, False, False)
    assert picked[1] == ["首帧图", "尾帧图"]   # 解析不依赖图存在（注入时才判）
    assert _hash_flags(picked, explicit, 3, False, False) == [False, True, False]


def _patch_csv():
    import node_helpers
    real = node_helpers.conditioning_set_values

    def csv(conditioning, values):
        return [[conditioning[0][0], {**conditioning[0][1], **values}]]

    node_helpers.conditioning_set_values = csv
    return node_helpers, real


def test_apply_guide_head_keyframe_position():
    nh, real = _patch_csv()
    try:
        guide = {"resolved_frame_index": 0, "latent": "G"}
        head = {"latent": "HEAD"}
        tail = {"latent": "T"}
        existing = [{"resolved_frame_index": 5, "latent": "E"}]
        cond = [["s", {"minimax_keyframes": list(existing)}]]
        out = plugin_nodes.H3SeamlessChainSampler._apply_guide(
            cond, guide, 48, tail_kf_latent=tail, head_kf_latent=head)
        kfs = out[0][1]["minimax_keyframes"]
        # 已有→桥→头锚（包 keyframe 壳钉 index 0）→尾锚（钉段尾）
        assert [k["latent"] for k in kfs] == ["E", "G", head, tail]
        assert kfs[2]["resolved_frame_index"] == 0       # 头锚钉段头（与桥同位叠加）
        assert kfs[3]["resolved_frame_index"] == 47      # 尾锚钉段尾
        assert out[0][1]["minimax_frame_count"] == 48
    finally:
        nh.conditioning_set_values = real


def test_apply_guide_head_only_still_injects():
    nh, real = _patch_csv()
    head = {"latent": "HEAD"}
    try:
        cond = [["s", {"minimax_keyframes": []}]]
        out = plugin_nodes.H3SeamlessChainSampler._apply_guide(
            cond, None, 48, tail_kf_latent=None, head_kf_latent=head)
        kfs = out[0][1]["minimax_keyframes"]
        assert len(kfs) == 1 and kfs[0]["latent"] == head
        assert kfs[0]["resolved_frame_index"] == 0
    finally:
        nh.conditioning_set_values = real


def test_apply_guide_none_head_unchanged_behavior():
    nh, real = _patch_csv()
    try:
        cond = [["s", {"minimax_keyframes": []}]]
        out = plugin_nodes.H3SeamlessChainSampler._apply_guide(
            cond, None, 48, tail_kf_latent=None, head_kf_latent=None)
        assert out is cond   # 无任何锚 → 原样返回（零影响）
    finally:
        nh.conditioning_set_values = real


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"OK {len(fns)} tests")
