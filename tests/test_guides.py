"""官方 AddGuide 锚点语义单测：python tests/test_guides.py（无 torch 依赖）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from ComfyUI_H3_SeamlessChain import guides


class _FakeAudio:
    """最小音频 latent 替身：只需 shape[-1] 与末维切片（stub 环境可跑）。"""

    def __init__(self, t):
        self.shape = (1, 32, 2, int(t))

    def __getitem__(self, key):
        sl = key[1] if isinstance(key, tuple) else key
        return _FakeAudio(sl.stop - (sl.start or 0))

    def clone(self):
        return _FakeAudio(self.shape[-1])


class _FakeVideo:
    """最小视频 latent 替身 [1,24,T,H,W]：guides 只读 shape[2]。"""

    def __init__(self, t):
        self.shape = (1, 24, int(t), 8, 8)


def test_resolve_frame_index():
    assert guides.resolve_frame_index(0, 124) == 0
    assert guides.resolve_frame_index(30, 124) == 30
    assert guides.resolve_frame_index(-1, 124) == 123      # 负索引自尾部
    assert guides.resolve_frame_index(-124, 124) == 0
    assert guides.resolve_frame_index(-200, 124) == -76    # 越界（由 validate 拦截）


def test_clip_guide_frames():
    for n in (0, 1, 2, 3, 4):
        assert guides.clip_guide_frames(n) == 1            # 不足 5 帧 -> 单帧锚
    assert guides.clip_guide_frames(5) == 5
    assert guides.clip_guide_frames(6) == 5
    assert guides.clip_guide_frames(21) == 5
    assert guides.clip_guide_frames(22) == 22
    assert guides.clip_guide_frames(30) == 22              # 向下对齐 17k+5
    assert guides.clip_guide_frames(39) == 39
    for k in range(0, 10):
        assert guides.clip_guide_frames(17 * k + 5) == 17 * k + 5


def test_validate_anchor():
    assert guides.validate_anchor(0, 1, 124) == (True, "")
    assert guides.validate_anchor(0, 22, 124) == (True, "")
    assert guides.validate_anchor(102, 22, 124) == (True, "")   # 102+22=124 恰好
    ok, why = guides.validate_anchor(103, 22, 124)
    assert not ok and "超出" in why
    ok, why = guides.validate_anchor(-1, 1, 124)
    assert ok                                              # 123+1=124 合法
    ok, why = guides.validate_anchor(124, 1, 124)
    assert not ok and "超出" in why
    ok, why = guides.validate_anchor(-125, 1, 124)
    assert not ok and "负索引" in why


def test_crop_audio_for_anchor():
    audio_t = 207                                          # 124 帧对应的音频 token
    a = _FakeAudio(audio_t)
    assert guides.crop_audio_for_anchor(a, 0, audio_t) is a      # 帧 0：剩余全长
    assert guides.crop_audio_for_anchor(None, 0, audio_t) is None
    # 帧 100：floor(207 - 5/3*100) = 40 -> 裁剪
    cropped = guides.crop_audio_for_anchor(a, 100, audio_t)
    assert cropped is not a and cropped.shape[-1] == 40
    # 帧 124：floor(207 - 206.67) = 0 -> 不足 1，丢弃音频分支
    assert guides.crop_audio_for_anchor(a, 124, audio_t) is None
    assert guides.crop_audio_for_anchor(a, 123, audio_t).shape[-1] == 2


def test_build_keyframe_omits_empty_branches():
    a = _FakeAudio(10)
    kf = guides.build_keyframe(7, video_latent=None, audio_latent=a)
    assert kf == {"resolved_frame_index": 7, "audio_latent": a}
    kf = guides.build_keyframe(0, video_latent=a)
    assert set(kf) == {"resolved_frame_index", "latent"}
    assert guides.build_keyframe(3) == {"resolved_frame_index": 3}


def test_audit_keyframes_report_only():
    """体检只出告警、不改数据：越界的才报，合法的静默。"""
    good = [{"resolved_frame_index": 0, "latent": _FakeVideo(7)},
            {"resolved_frame_index": 123, "latent": _FakeVideo(1)}]
    assert guides.audit_keyframes(good, 124) == []
    bad = good + [{"resolved_frame_index": 110, "latent": _FakeVideo(7)}]
    warns = guides.audit_keyframes(bad, 124)
    assert len(warns) == 1 and "越界告警" in warns[0]
    assert len(bad) == 3                       # 不删元素
    assert guides.audit_keyframes([], 124) == []
    assert guides.audit_keyframes(None, 124) == []


def test_mid_anchor_index():
    assert guides.mid_anchor_index(124) == 62
    assert guides.mid_anchor_index(124, 0.25) == 31
    assert guides.mid_anchor_index(124, 1.0) == 123        # 夹到末帧
    assert guides.mid_anchor_index(0) == -1


def test_latent_frames_of():
    assert guides.latent_frames_of(_FakeVideo(1)) == 1
    # 7 token = 22 帧（FRAME_PER_TOKEN 循环：1,4,4,4,4,1,4）——17k+5 ↔ 5k+2 的逆映射
    assert guides.latent_frames_of(_FakeVideo(7)) == 22
    assert guides.latent_frames_of(None) == 0


def test_prepare_anchor_soft_degrades():
    a = _FakeAudio(207)
    # 越界：7 token（22 帧）片段钉在帧 110（110+22 > 124）
    kf, note = guides.prepare_anchor(110, 124, video_latent=_FakeVideo(7), label="段中")
    assert kf is None and "段中锚跳过" in note
    # 合法 + 音频裁剪
    kf, note = guides.prepare_anchor(100, 124, audio_latent=a, audio_t=207, label="段尾")
    assert kf is not None
    assert kf["resolved_frame_index"] == 100
    assert kf["audio_latent"].shape[-1] == 40
    assert "裁剪" in note
    # 无素材
    kf, note = guides.prepare_anchor(10, 124)
    assert kf is None and "无可用素材" in note


def test_prepare_anchor_tail_audio_trim():
    """段末锚（帧 frame_count-1）的音频按剩余时长裁剪；不足时只丢音频、保留画面锚。"""
    v = _FakeVideo(1)          # 单帧画面锚：123 + 1 = 124，恰好合法
    a = _FakeAudio(207)
    kf, note = guides.prepare_anchor(123, 124, video_latent=v, audio_latent=a, audio_t=207)
    assert kf is not None and "latent" in kf
    # 剩余 = floor(207 - 5/3*123) = 2 token
    assert kf["audio_latent"].shape[-1] == 2
    assert "裁剪" in note
    # 音频流比锚点剩余还短（素材形状不匹配时的防御）：丢弃音频分支，画面锚照常
    kf2, note2 = guides.prepare_anchor(123, 124, video_latent=v,
                                       audio_latent=_FakeAudio(1), audio_t=1)
    assert kf2 is not None and "latent" in kf2 and "audio_latent" not in kf2
    assert "音频剩余时长不足" in note2


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"  ok  {name}")
    print("test_guides: 全部通过")
