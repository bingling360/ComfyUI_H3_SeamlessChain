"""分镜模式单测（stub 纯逻辑）：python tests/test_storyboard.py

覆盖：段计划纯数学 / 尾锚网格对齐 / 段哈希失效规则（改关键帧/提示词 -> 正确起始段）/
节点 schema 结构 / IS_CHANGED / 断点导入目录校验。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # 插件父目录
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests 目录

from test_node_structure import _install_stubs


def test_storyboard_plan():
    from ComfyUI_H3_SeamlessChain import storyboard
    plan = storyboard.storyboard_plan(3, ["a", "b"])
    assert plan == [(0, 1, "a"), (1, 2, "b")]                 # N 图定义 N-1 段
    for bad in (1, 0):
        try:
            storyboard.storyboard_plan(bad, [])
            raise AssertionError("should require >=2 keyframes")
        except ValueError:
            pass
    try:
        storyboard.storyboard_plan(4, ["a", "b"])
        raise AssertionError("should reject prompt count mismatch")
    except ValueError as e:
        assert "3" in str(e) and "2" in str(e)                # 4 图应 3 段，当前 2 条


def test_tail_anchor_index():
    from ComfyUI_H3_SeamlessChain import storyboard
    assert storyboard.tail_anchor_index(124) == 119           # 124-5 = 17*7
    assert storyboard.tail_anchor_index(141) == 136           # 141-5 = 17*8
    assert storyboard.tail_anchor_index(130) == 119           # 非网格先向下对齐 124
    assert storyboard.tail_anchor_index(22) == 17
    for fc in (22, 39, 56, 73, 124, 141, 175):
        assert storyboard.tail_anchor_index(fc) % 17 == 0     # 恒落在 token 边界


def test_seg_hash_invalidation():
    from ComfyUI_H3_SeamlessChain import storyboard, checkpoint
    locked = lambda p, a, b: storyboard.H3StoryboardChain._seg_hash(p, a, b, True)
    kfh = ["h0", "h1", "h2", "h3", "h4"]
    prompts = ["p1", "p2", "p3", "p4"]
    segs = [locked(prompts[i], kfh[i], kfh[i + 1]) for i in range(4)]

    def reroll_with(kf_mod, prompt_mod=None):
        k = list(kfh)
        if kf_mod is not None:
            k[kf_mod[0]] = kf_mod[1]
        p = list(prompts)
        if prompt_mod is not None:
            p[prompt_mod[0]] = prompt_mod[1]
        new = [locked(p[i], k[i], k[i + 1]) for i in range(4)]
        return checkpoint.reroll_start(segs, new, 4)

    assert reroll_with(None, (2, "p3'")) == 2                 # 改段3提示词 -> 段3起重做
    assert reroll_with((1, "h1'")) == 0                       # 改kf2：段1尾锚+段2首帧 -> 段1起
    assert reroll_with((2, "h2'")) == 1                       # 改kf3：段2尾锚+段3首帧 -> 段2起
    assert reroll_with((4, "h4'")) == 3                       # 改末帧：仅段4尾锚 -> 段4起
    assert reroll_with((0, "h0'")) == 0                       # 改首帧：仅段1首帧 -> 段1起

    open_hash = lambda p, a, b: storyboard.H3StoryboardChain._seg_hash(p, a, b, False)
    segs_open = [open_hash(prompts[i], kfh[i], kfh[i + 1]) for i in range(4)]
    k3 = list(kfh)
    k3[2] = "h2'"
    segs_open2 = [open_hash(prompts[i], k3[i], k3[i + 1]) for i in range(4)]
    # 尾锚关闭：改关键帧 i 只影响段 i 首帧 -> 段 i 起重做（不含段 i-1）
    assert checkpoint.reroll_start(segs_open, segs_open2, 4) == 2
    # 提示词相同+关键帧相同 -> 无需重做
    assert checkpoint.reroll_start(segs, list(segs), 4) == 4


def test_schema():
    from ComfyUI_H3_SeamlessChain import storyboard
    schema = storyboard.H3StoryboardChain.define_schema()
    ids = [inp.id for inp in schema.inputs]
    for key in ["模型", "文本编码器", "视频VAE", "音频VAE", "宽度", "高度", "每段帧数",
                "种子", "步数", "CFG", "采样器", "调度器",
                "尾帧锚定", "锚定加噪", "响度对齐",
                "断点续拍", "断点目录", "审片模式", "重跑起始段",
                "关键帧来源", "断点导入目录",
                "分镜图组", "提示词组",
                "参考图片组", "参考视频组", "参考视频音轨组", "参考音频组"]:
        assert key in ids, f"missing input: {key}"
    by_id = {inp.id: inp for inp in schema.inputs}
    assert by_id["尾帧锚定"].kwargs.get("options") == ["开启", "关闭"]
    assert by_id["尾帧锚定"].kwargs.get("default") == "开启"
    assert by_id["响度对齐"].kwargs.get("default") == "开启"
    assert by_id["关键帧来源"].kwargs.get("options") == ["上传", "从断点导入"]
    assert by_id["每段帧数"].kwargs.get("min") == 22
    assert by_id["锚定加噪"].kwargs.get("default") == 0.0
    outs = [(o.id, o.is_output_list) for o in schema.outputs]
    assert outs == [("图像", False), ("音频", False), ("帧率", False), ("报告", False),
                    ("分段图像", True), ("分段音频", True)]


def test_is_changed():
    import math
    from ComfyUI_H3_SeamlessChain import storyboard
    cls = storyboard.H3StoryboardChain
    assert math.isnan(cls.IS_CHANGED(审片模式="逐段确认"))
    assert math.isnan(cls.IS_CHANGED(断点续拍="自动续跑"))
    assert cls.IS_CHANGED() == ""


def test_keyframes_from_ckpt_missing():
    from ComfyUI_H3_SeamlessChain import storyboard
    try:
        storyboard.keyframes_from_ckpt(os.path.abspath(os.path.join(
            os.sep, "__no_such_dir__", "storyboard_test")), None)
        raise AssertionError("should reject missing dir")
    except ValueError as e:
        assert "不存在" in str(e)


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
