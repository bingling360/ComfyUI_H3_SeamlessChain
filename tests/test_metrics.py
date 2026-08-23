"""接缝评测基线单测（合成序列方向性验证）：python tests/test_metrics.py

合成"匀速平移纹理"序列构造可控接缝：
- 好缝（速度连续）→ 光流/相机 z 低；
- 坏缝（位置瞬移 / 速度跳变 / 颜色漂移）→ 对应维度 z 显著升高。
缺 cv2 时光流/相机断言自动 SKIP；无 torch 全 SKIP（与其他测试同策略）。
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    if not hasattr(torch, "manual_seed"):
        # 合集运行时 test_node_structure 的假 torch stub 可能已在 sys.modules——视为无 torch
        raise ImportError
except ImportError:
    torch = None

if torch is not None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ComfyUI_H3_SeamlessChain import metrics

    torch.manual_seed(7)

    def _canvas(h=120, w=160):
        """噪声纹理画布：高频角点利于 ORB/光流，噪声保证帧间估计有自然方差。"""
        # 独立生成器避免合集运行时其他测试消耗全局 RNG 后造成接缝阈值抖动。
        gen = torch.Generator().manual_seed(7)
        return torch.rand(h, w, generator=gen) * 0.6 + 0.2

    def _pan_frames(canvas, x0, speed, n, color_shift=0.0):
        """从画布按 x0 + i*speed 取窗，n 帧 [n,h,w,3] 0-1。"""
        h, w = canvas.shape
        fw = w // 2
        out = []
        for i in range(n):
            x = int(round(x0 + i * speed))
            x = max(0, min(x, w - fw))
            g = canvas[:, x:x + fw]
            rgb = torch.stack([g + color_shift, g, g - color_shift * 0.5], dim=-1)
            out.append(rgb.clamp(0.0, 1.0))
        return torch.stack(out)


def _good_chain():
    """速度连续的 80 帧：缝在 40，B[0] 恰好续 A[-1] 的下一位置。"""
    canvas = _canvas()
    a = _pan_frames(canvas, 0, 1.0, 40)
    b = _pan_frames(canvas, 40, 1.0, 40)          # A 末帧位置 39 → B 首 40，无缝
    return torch.cat([a, b], dim=0), 40


def _teleport_seam():
    """位置瞬移缝：B 从 A 末帧位置 +7px 开始，速度不变 → 缝对速度 7px/帧。"""
    canvas = _canvas()
    a = _pan_frames(canvas, 0, 1.0, 40)
    b = _pan_frames(canvas, 46, 1.0, 40)          # A 末 39 → B 首 46：跳 7px
    return torch.cat([a, b], dim=0), 40


def _color_drift_seam():
    """颜色漂移缝：B 延续位置但整体偏红。"""
    canvas = _canvas()
    a = _pan_frames(canvas, 0, 1.0, 40)
    b = _pan_frames(canvas, 40, 1.0, 40, color_shift=0.12)
    return torch.cat([a, b], dim=0), 40


def test_robust_z_math():
    if torch is None:
        return print("SKIP (no torch)")
    base = [1.0] * 30 + [0.9, 1.1]
    assert metrics._robust_z(1.0, base) is not None and abs(metrics._robust_z(1.0, base)) < 1.0
    assert metrics._robust_z(6.0, base) > 5.0            # 离群值大幅升高
    assert metrics._robust_z(1.0, [1.0, None, 1.0]) is None   # 基线不足 → None
    print("PASS test_robust_z_math")


def test_good_seam_low_z():
    if torch is None:
        return print("SKIP (no torch)")
    video, c = _good_chain()
    res = metrics.evaluate_seams(video, [c])
    row = res["seams"][0]
    if metrics.cv2 is not None:
        assert row["flow_z"] is not None and row["flow_z"] < 2.0, row
        assert abs(row["cam_z"] or 0.0) < 3.0, row
    assert abs(row["lpips_z"] or 0.0) < 3.0, row
    assert abs(row["emb_z"] or 0.0) < 3.0, row
    print(f"PASS test_good_seam_low_z {row}")


def test_teleport_seam_flow_z_spikes():
    if torch is None:
        return print("SKIP (no torch)")
    if metrics.cv2 is None:
        return print("SKIP (no cv2)")
    video, c = _teleport_seam()
    res = metrics.evaluate_seams(video, [c])
    row = res["seams"][0]
    assert row["flow_z"] is not None and row["flow_z"] > 3.0, row
    assert (row["cam_z"] or 0.0) > 3.0, row
    # 好缝对照组：同一画布速度连续处 z 应远低于坏缝
    video2, c2 = _good_chain()
    row2 = metrics.evaluate_seams(video2, [c2])["seams"][0]
    assert row["flow_z"] > row2["flow_z"] + 2.0
    print(f"PASS test_teleport_seam_flow_z_spikes flow_z={row['flow_z']:.1f} cam_z={row['cam_z']:.1f}")


def test_color_drift_emb_z_spikes():
    if torch is None:
        return print("SKIP (no torch)")
    video, c = _color_drift_seam()
    res = metrics.evaluate_seams(video, [c])
    row = res["seams"][0]
    assert row["emb_z"] is not None and row["emb_z"] > 3.0, row
    print(f"PASS test_color_drift_emb_z_spikes emb_z={row['emb_z']:.1f}")


def test_evaluate_local_concat():
    if torch is None:
        return print("SKIP (no torch)")
    canvas = _canvas()
    prev = _pan_frames(canvas, 0, 1.0, 24)
    cur = _pan_frames(canvas, 30, 1.0, 48)            # 跳 7px 的局部缝
    row = metrics.evaluate_local(prev, cur)
    assert row["c"] == 24
    if metrics.cv2 is not None:
        assert row["flow_z"] is not None and row["flow_z"] > 3.0, row
    print(f"PASS test_evaluate_local_concat {row}")


def test_summary_and_flags():
    if torch is None:
        return print("SKIP (no torch)")
    video, c = _teleport_seam()
    res = metrics.evaluate_seams(video, [c, 60])
    assert set(res["flags"]) >= {"flow", "cam", "lpips", "emb"}
    assert "flow_z_mean" in res["summary"] or metrics.cv2 is None
    # 越界缝下标自动剔除；空缝列表不崩
    assert metrics.evaluate_seams(video, [0, 9999])["seams"] == []
    assert metrics.evaluate_seams(video[:1], [1])["seams"] == []
    txt = metrics.fmt_seam_z(res["seams"][0])
    assert isinstance(txt, str) and len(txt) > 0
    print(f"PASS test_summary_and_flags flags={res['flags']}")


if __name__ == "__main__":
    if torch is None:
        print("SKIP (no torch) — 用带 torch 的 venv 跑：D:\\本地部署文件集合\\comfyui\\comfy shili\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe")
    else:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
        print("all tests passed")
