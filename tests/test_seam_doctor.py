"""接缝体检单测：python tests/test_seam_doctor.py（真实 torch，无需 ComfyUI）。

合成图案：缓慢漂移 + 半径缓慢增长的圆盘（grow>0 -> 非刚体，整体平移补不掉，
用于区分「时间断层」与「位移瞬移」）。时间线显式管理 t_start：
- 正常段：时间紧接前段
- 断层段：时间跳过 skip 帧（内容演化差 = 尺寸+位移，平移补不掉）
- 瞬移段：时间连续，圆心一次性挪 shift 像素
- 重复段：时间倒退 back 帧重播（stutter）
- 染色段：时间连续，整段加蓝色偏移
"""
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import seam_doctor as sd

W, H = 96, 64
_grid = torch.meshgrid(torch.arange(H).float(), torch.arange(W).float(), indexing="ij")
_GY, _GX = _grid


def circle_frame(t, cx0=24.0, drift=0.5, r0=10.0, grow=0.6, tint=(1.0, 1.0, 1.0)):
    cx = cx0 + drift * t
    r = r0 + grow * t
    dist = ((_GX - cx) ** 2 + (_GY - 32.0) ** 2).sqrt()
    # 纹理挂在圆盘坐标系（随内容整体移动）：背景平坦，平移搜索不被静态噪声骗
    v = (0.1 + 0.8 * (dist < r).float() + 0.05 * torch.sin(dist * 0.5))
    frame = torch.stack([v * tint[0], v * tint[1], v * tint[2]], dim=-1)
    return frame.clamp(0, 1)


def frames(fn, n, t0=0.0):
    return torch.stack([fn(t0 + i) for i in range(n)])


def seg_normal(n=12, t_start=0):
    return frames(lambda t: circle_frame(t), n, t0=t_start)


def seg_jump(n=12, t_start=0, skip=5):
    """断层：从 t_start 应起却跳到 t_start+skip（跳过 skip 帧内容）。"""
    return frames(lambda t: circle_frame(t), n, t0=t_start + skip)


def seg_warp(n=12, t_start=0, shift=10.0):
    """瞬移：时间连续，圆心一次性挪 shift 像素。"""
    return frames(lambda t: circle_frame(t, cx0=24.0 + shift), n, t0=t_start)


def seg_tint(n=12, t_start=0, blue=0.3):
    return frames(lambda t: circle_frame(t, tint=(1.0, 1.0, 1.0 + blue)), n, t0=t_start)


def seg_stutter(n=12, t_start=0, back=2):
    """重复：时间倒退 back 帧重播（缝后第 back 帧回到缝前内容）。"""
    return frames(lambda t: circle_frame(t), n, t0=t_start - back)


def test_frame_diffs_and_median():
    v = seg_normal(10)
    d = sd.frame_diffs(v)
    assert len(d) == 9
    assert all(x > 0 for x in d)
    med = float(torch.tensor(d).median())
    assert 0 < med < 0.06                       # 慢漂移：帧差很小


def test_locate_seams_from_segments():
    segs = [seg_normal(5), seg_normal(5), seg_normal(5)]
    assert sd.locate_seams_from_segments(segs) == [5, 10]
    assert sd.locate_seams_from_segments([seg_normal(7)]) == []


def test_auto_detect_seams():
    video = torch.cat([seg_normal(10), seg_jump(6, t_start=10, skip=5), seg_normal(6, t_start=21)])
    d = sd.frame_diffs(video)
    med = float(torch.tensor(d).median())
    assert d[9] > 3.0 * med                     # 断层缝差确实超阈值
    hits = sd.auto_detect_seams(d, ratio=3.0)
    assert 10 in hits                           # 断层缝被自动定位


def test_best_shift_finds_displacement():
    a = circle_frame(0)
    b = circle_frame(0, cx0=24.0 + 10)          # 挪 10px
    dx, dy, e0, eb, _ = sd.best_shift(a, b, radius=16)
    assert dx == 10 and dy == 0                 # 原图 96px 宽，降采样不缩，位移保真
    assert eb < 0.25 * e0                       # 平移后残差骤降


def test_jump_verdict_triangle():
    # 断层缝：ratio 大 + 平移消不掉 + NCC 高 -> 判断层
    video = torch.cat([seg_normal(12), seg_jump(12, t_start=12, skip=3)])
    d = sd.frame_diffs(video)
    med = float(torch.tensor(d).median())
    m = sd.diagnose_seam(video, 12, 1, d, med, window=6, radius=16, kmax=8,
                         fps=24, ratio_th=3.0)
    assert m["ratio"] >= 3.0                     # 缝差确实大（≈4 帧演化）
    assert m["ncc"] >= 0.5                       # 同一场景延续
    assert m["is_jump"] and 2 <= m["jump_frames"] <= 5


def test_jump_verdict_not_fooled_by_warp():
    # 瞬移缝：平移可消 -> 不判断层，走瞬移分支
    video = torch.cat([seg_normal(12), seg_warp(12, t_start=12)])
    d = sd.frame_diffs(video)
    med = float(torch.tensor(d).median())
    m = sd.diagnose_seam(video, 12, 1, d, med, window=6, radius=16, kmax=8,
                         fps=24, ratio_th=3.0)
    assert not m["is_jump"] and m["shift_fix"]


def test_dup_probe_detects_stutter():
    video = torch.cat([seg_normal(12), seg_stutter(12, t_start=12, back=2)])
    errs, k_star, dup = sd.dup_probe(video, 12, kmax=8, radius=16)
    assert dup and 1 <= k_star <= 2             # 缝后 1~2 帧重播了缝前内容


def test_color_and_ncc():
    a = seg_tint(6, t_start=0, blue=0.0)
    b = seg_tint(6, t_start=6, blue=0.35)
    both = torch.cat([a, b])
    ma, _ = sd.window_color(both, 0, 6)
    mb, _ = sd.window_color(both, 6, 12)
    d_e = math.sqrt(sum((x - y) ** 2 for x, y in zip(ma, mb)))
    assert d_e > 0.05                            # 颜色漂移检出
    assert sd.ncc(a[-1], b[0]) > 0.9             # 同一画面：NCC 高
    other = frames(lambda t: circle_frame(t, r0=26.0, grow=0.0), 4)
    assert sd.ncc(a[-1], other[0]) < 0.5         # 换内容：NCC 低


def test_audio_probe_loudness_jump():
    sr = 44100
    t = torch.arange(sr) / sr
    quiet = (0.05 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(0)
    loud = (0.5 * torch.sin(2 * math.pi * 440 * t)).unsqueeze(0)
    wav = torch.cat([quiet, loud], dim=1)       # [1, 2*sr]
    p = sd.audio_probe(wav, sr, sample_i=sr)
    assert p["rms_db"] is not None and 15 < p["rms_db"] < 25   # ~+20dB
    assert p["peak"] > 0.4


def test_analyze_full_report():
    segs = [seg_normal(12),
            seg_jump(12, t_start=12, skip=3),
            seg_tint(12, t_start=27, blue=0.3)]
    video = torch.cat(segs)
    sr = 44100
    t = torch.arange(video.shape[0] * 100) / sr
    wav = (0.2 * torch.sin(2 * math.pi * 300 * t)).unsqueeze(0)
    report, gallery, metrics = sd.analyze(
        video, wav, sr, fps=24, segs=segs, window=6, radius=16, kmax=8, ratio_th=3.0)
    assert len(metrics) == 2
    assert metrics[0]["is_jump"]                 # 缝 1：断层
    assert metrics[1]["d_e"] > 0.05              # 缝 2：颜色漂移
    assert "时间断层" in report and "颜色漂移" in report
    assert "帧数守恒" in report and report.count("✓") >= 1
    assert "拼装" in report                      # 分段一致性块存在
    assert gallery is not None and gallery.shape[0] == 1 and gallery.shape[-1] == 3


def test_analyze_auto_mode_and_consistency():
    video = torch.cat([seg_normal(10), seg_jump(6, t_start=10, skip=5), seg_normal(6, t_start=21)])
    report, gallery, metrics = sd.analyze(
        video, None, 0, fps=24, segs=[], window=6, radius=16, kmax=8, ratio_th=3.0)
    assert any(m["is_jump"] for m in metrics)    # 自动模式也定位到断层缝
    assert "自动检测" in report
    # 假分段（帧数不守恒）-> 分段定位被拒，明示后转自动检测
    segs_bad = [video[:5]]
    report2, _, _ = sd.analyze(video, None, 0, 24, segs_bad, 6, 16, 8, 3.0)
    assert "不守恒" in report2 and "自动检测" in report2


def test_join_consistency_metrics():
    segs = [seg_normal(6, t_start=0), seg_normal(6, t_start=6)]
    video = torch.cat(segs).clone()
    video[6] = video[6] * 0.9                    # 模拟 smoothstep 混合改动缝后帧
    d = sd.frame_diffs(video)
    m = sd.diagnose_seam(video, 6, 1, d, float(torch.tensor(d).median()),
                         window=4, radius=8, kmax=4, fps=24, ratio_th=3.0,
                         wav=None, sr=None, seg_pair=(segs[0][-1], segs[1][0]))
    assert m["join"] is not None
    assert m["join"][0] < 1e-4                   # 缝前帧与段尾一致
    assert m["join"][1] > 1e-4                   # 缝后帧与段首不同 -> 混合生效


def test_save_report():
    import shutil
    import tempfile
    import types
    root = tempfile.mkdtemp()
    sys.modules["folder_paths"] = types.SimpleNamespace(
        get_output_directory=lambda: root)
    try:
        rel = sd.save_report("体检内容 abc")
        assert rel and rel.startswith("h3_seam_doctor") and rel.endswith(".txt")
        with open(os.path.join(root, rel), encoding="utf-8") as f:
            assert f.read() == "体检内容 abc"
    finally:
        del sys.modules["folder_paths"]
        shutil.rmtree(root)
    assert sd.save_report("x") is None            # 无 folder_paths 环境 -> None 不炸


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
