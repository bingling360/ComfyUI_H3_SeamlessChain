"""官方 h3-prompt-writing 提示词组装单测：python tests/test_prompt_compose.py（无需 torch）。

覆盖：官方格式直通、三字段组装与句号连接、留空省略（不写 N/A）、
「」对白自动转 <d>[中文]>（含已手写 <d> 跳过）、[Shot 前缀去重、
参考头 subject_definitions 格式、i2v 首段指令行注入条件。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402


def test_official_passthrough():
    """主提示词已含官方字段标签 -> 原样直通，不二次包装。"""
    co = plugin_nodes._compose_official
    official = ("integrated_multimodal_description: [Shot 1] Live-action.\n"
                "overall_soundscape: Rain.\n"
                "non_diegetic_music: N/A")
    out, dlg = co("", "", official, "雨声", "钢琴")
    assert out == official and dlg == 0                      # 场景/角色/声音全被直通丢弃
    for label in ("integrated_multimodal_description:", "overall_soundscape:", "detailed_description:"):
        raw = f"{label} some body"
        out2, _ = co("场景", "角色", raw, "", "")
        assert out2 == raw                                   # 单标签也算官方格式


def test_compose_three_fields():
    """场景/角色/主提示词句号连接进描述行；环境音/配乐成独立字段行。"""
    co = plugin_nodes._compose_official
    out, dlg = co("雨夜的东京小巷", "米色风衣短发女主",
                  "她抬头看向雨幕", "雨点敲打柏油路", "慢速钢琴单音")
    lines = out.split("\n")
    assert lines[0] == ("integrated_multimodal_description: [Shot 1] "
                        "雨夜的东京小巷。米色风衣短发女主。她抬头看向雨幕")
    assert lines[1] == "overall_soundscape: 雨点敲打柏油路"
    assert lines[2] == "non_diegetic_music: 慢速钢琴单音"
    assert dlg == 0
    # 尾部标点剥离：各部分以句号结尾时不产生句号堆积
    out2, _ = co("小巷。", "女主。", "抬头。", "雨声。", "钢琴。")
    assert out2.split("\n")[0].endswith("抬头")
    assert "。。" not in out2


def test_compose_empty_fields_omitted():
    """环境音/配乐留空 -> 整行省略（不写 N/A）；全部留空只剩描述行。"""
    co = plugin_nodes._compose_official
    out, _ = co("场景", "", "动作", "", "")
    assert out == "integrated_multimodal_description: [Shot 1] 场景。动作"
    assert "overall_soundscape" not in out and "non_diegetic_music" not in out
    out2, _ = co("", "", "只有主提示词", "", "")
    assert out2 == "integrated_multimodal_description: [Shot 1] 只有主提示词"
    out3, _ = co("", "", "", "只有环境音", "只有配乐")
    lines = out3.split("\n")
    assert len(lines) == 2                                   # 无描述体时不产生空描述行
    assert lines[0] == "overall_soundscape: 只有环境音"
    assert lines[1] == "non_diegetic_music: 只有配乐"


def test_dialogue_wrap():
    """「……」→ <d>[中文] ……</d>；已手写 <d> 时跳过；英文双引号不动。"""
    co = plugin_nodes._compose_official
    out, dlg = co("", "", '她说：「今天也不顺利。」随后沉默', "", "")
    assert dlg == 1
    assert "<d>[中文] 今天也不顺利。</d>" in out
    out2, dlg2 = co("", "", '她说「甲」又说「乙」', "", "")
    assert dlg2 == 2 and out2.count("<d>") == 2              # 多处对白全转
    # 已手写 <d>：整体跳过，「」原样保留（用户自管格式）
    out3, dlg3 = co("", "", '她说：<d>[中文] 手写</d>，又「没转」', "", "")
    assert dlg3 == 0 and "「没转」" in out3 and "<d>[中文] 手写</d>" in out3
    # 英文双引号是官方可见文字记法，不动
    out4, _ = co("", "", 'A sign reading "营业中" glows', "", "")
    assert '"营业中"' in out4 and "<d>" not in out4


def test_shot_prefix_dedup():
    """主提示词已带 [Shot N] 开头标签：剥掉防重复，场景并入其后，保留用户镜号。"""
    co = plugin_nodes._compose_official
    out, _ = co("场景", "", "[Shot 1] 开场镜头描述", "", "")
    assert out.startswith("integrated_multimodal_description: [Shot 1] 场景。开场镜头描述")
    assert out.count("[Shot") == 1
    out2, _ = co("", "", "[Shot 3] 续镜描述", "", "")
    assert out2.startswith("integrated_multimodal_description: [Shot 3] 续镜描述")


def test_i2v_instruction_line():
    """i2v 首段指令行注入逻辑（复现 execute 内联条件）。"""
    seg_prompts = ["普通首段", "普通续段"]
    first_frame = object()                                   # 非 None 即 i2v
    if first_frame is not None and not plugin_nodes._OFFICIAL_FIELD_RE.search(seg_prompts[0]):
        seg_prompts[0] = ("For the target video, at 0.00 seconds into the target video, "
                           "<Picture 1> (from [Shot 1]) is fully referenced.\n" + seg_prompts[0])
    assert seg_prompts[0].startswith("For the target video, at 0.00 seconds")   # 普通段被注入
    assert "For the target video" not in seg_prompts[1]                         # 续段不注入

    seg2 = ["integrated_multimodal_description: 已是官方格式"]
    if first_frame is not None and not plugin_nodes._OFFICIAL_FIELD_RE.search(seg2[0]):
        seg2[0] = "PREFIX\n" + seg2[0]
    assert seg2[0].startswith("integrated")                 # 官方格式段不注入


def test_segments_json_backward_compat():
    """旧 segments JSON 无 soundscape/music 键 -> 空值参与组装不报错（execute 内联规则）。"""
    co = plugin_nodes._compose_official
    seg = {"scene_prompt": "夜景", "character_prompt": "主角", "seconds": 6.5}   # 旧 v3 状态
    out, _ = co(str(seg.get("scene_prompt", "")).strip(),
                str(seg.get("character_prompt", "")).strip(),
                "主提示词",
                str(seg.get("soundscape", "")).strip(),
                str(seg.get("music", "")).strip())
    assert out == "integrated_multimodal_description: [Shot 1] 夜景。主角。主提示词"


if __name__ == "__main__":
    test_official_passthrough()
    test_compose_three_fields()
    test_compose_empty_fields_omitted()
    test_dialogue_wrap()
    test_shot_prefix_dedup()
    test_i2v_instruction_line()
    test_segments_json_backward_compat()
    print("ALL PASS")
