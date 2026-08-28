"""桥区软着陆单测：python tests/test_bridge.py（需 torch；stub 环境自动跳过）。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

try:
    import torch
    if not hasattr(torch, "arange"):
        raise ImportError
    from ComfyUI_H3_SeamlessChain import bridge
    from ComfyUI_H3_SeamlessChain.grid import latent_t_to_frames
except ImportError:
    torch = None


class _NT:
    """NestedTensor 替身：bridge 只用构造与 unbind（stub 环境可跑）。"""

    def __init__(self, tensors):
        self.tensors = list(tensors)
        self.is_nested = True

    def unbind(self):
        return self.tensors


def _install_nested():
    bridge.NestedTensor = _NT


# 合集运行（pytest）时也要装好替身：bridge 的 NestedTensor 是模块级属性，
# 装配函数（`nested_pair`）在调用时才解析它。
_install_nested()


def _video(t=37, h=48, w=84, fill=None):
    x = torch.zeros(1, 24, t, h, w)
    if fill is not None:
        x[...] = float(fill)
    return x


def _audio(t=207, fill=None):
    x = torch.zeros(1, 32, 2, t)
    if fill is not None:
        x[...] = float(fill)
    return x


def test_hold_plan():
    assert bridge.hold_plan(0) == (0, 0, 0)
    assert bridge.hold_plan(5) == (5, 2, 8)        # 1+4 帧 / round(5*5/3)
    assert bridge.hold_plan(9) == (9, 3, 15)
    assert bridge.hold_plan(22) == (22, 7, 37)     # ctx 默认档 = 7 token
    # token↔帧网格是 1,5,9,13,17,18,22,…（首 token 1 帧、其后 4 帧，每 5 token 循环）；
    # 21 不在网格上，向下落到 18
    assert bridge.hold_plan(21) == (18, 6, 30)
    for hf, ht, _ in (bridge.hold_plan(n) for n in (5, 9, 13, 17, 18, 22)):
        assert latent_t_to_frames(ht) == hf        # 帧数必须落到真实 token 网格


def test_curve_value():
    assert bridge.curve_value(0.0) == 0.0
    assert bridge.curve_value(1.0) == 0.0          # hold：整窗钉死
    assert bridge.curve_value(0.5, "linear") == 0.5
    assert abs(bridge.curve_value(0.5, "smoothstep") - 0.5) < 1e-6
    assert abs(bridge.curve_value(0.5, "ease_in") - 0.25) < 1e-6
    assert bridge.curve_value(-1.0, "linear") == 0.0     # 越界钳制
    assert bridge.curve_value(9.0, "linear") == 1.0


def test_video_hold_mask_shape_and_boundary():
    m = bridge.video_hold_mask(37, 48, 84, 22)
    assert tuple(m.shape) == (1, 1, 37, 48, 84)
    # 22 帧 = 前 7 个 token（token 中心帧 0.5/3/7/11/15/17.5/20，第 8 个中心 24 ≥ 22）
    assert float(m[0, 0, :7].max()) == 0.0
    assert float(m[0, 0, 7:].min()) == 1.0
    # 逐 token 边界不按序号线性取：第 6、7 个 token 中心 17.5 / 20 仍在窗内
    assert float(m[0, 0, 5, 0, 0]) == 0.0 and float(m[0, 0, 6, 0, 0]) == 0.0


def test_video_hold_mask_ramp_monotone():
    m = bridge.video_hold_mask(37, 8, 8, 22, curve="smoothstep")
    col = m[0, 0, :, 0, 0]
    assert float(col[0]) < float(col[3]) < float(col[6]) < 1.0
    assert float(col[7]) == 1.0                    # 窗外完全生成
    assert (col.diff() >= 0).all()                 # 单调不减


def test_audio_hold_mask_stereo_channel_major():
    m = bridge.audio_hold_mask(207, 15)
    assert tuple(m.shape) == (1, 1, 2, 207)
    assert torch.equal(m[0, 0, 0], m[0, 0, 1])     # 双声道同斜坡
    assert float(m[0, 0, 0, :15].max()) == 0.0
    assert float(m[0, 0, 0, 15:].min()) == 1.0
    m2 = bridge.audio_hold_mask(207, 15, curve="linear")
    row = m2[0, 0, 0]
    assert (row.diff() >= 0).all() and float(row[-1]) == 1.0


def test_build_initial_latent_head_is_prev_tail():
    v, a = _video(), _audio()
    bv, ba = _video(t=7, fill=3.0), _audio(t=37, fill=5.0)
    out = bridge.build_initial_latent(v, a, bv, ba)
    vv, aa = out.unbind()
    assert torch.equal(vv[:, :, :7], bv)           # 头部 == 上段尾
    assert float(vv[:, :, 7:].abs().max()) == 0.0  # 其余为空（采样器按掩码补噪）
    assert torch.equal(aa[..., :37], ba)
    assert float(aa[..., 37:].abs().max()) == 0.0
    assert vv.dtype == v.dtype and vv.shape == v.shape


def test_build_initial_latent_clamps_oversize_bridge():
    v, a = _video(t=7), _audio(t=37)
    out = bridge.build_initial_latent(v, a, _video(t=37, fill=1.0), _audio(t=207, fill=2.0))
    vv, aa = out.unbind()
    assert vv.shape == v.shape and aa.shape == a.shape      # 超长桥被截断而非报错


def test_hold_slices_take_tail():
    guide = {"latent": _video(t=7), "audio_latent": _audio(t=37)}
    guide["latent"][:, :, 0] = 1.0
    guide["latent"][:, :, -1] = 9.0
    bv, ba = bridge.hold_slices(guide, 3, 15)
    assert bv.shape[2] == 3
    assert float(bv[0, 0, -1, 0, 0]) == 9.0        # 取的是桥尾（=上段尾）
    assert ba.shape[-1] == 15
    assert bridge.hold_slices(guide, 0, 0) == (None, None)
    assert bridge.hold_slices(None, 3, 15) == (None, None)


def test_resolve_crop_matches_existing_behaviour():
    """现状路径必须与改动前逐字节一致（这是回归底线）。"""
    # 事实首段 / 独立镜头段：不裁头也不多采样
    assert bridge.resolve_crop(0, 22, has_source=False) == (0, 0)
    assert bridge.resolve_crop(9, 22, has_source=False) == (0, 0)
    # 有上段、软桥未生效：裁 ctx、多采样 ctx
    assert bridge.resolve_crop(0, 22, has_source=True) == (22, 22)
    assert bridge.resolve_crop(0, 39, has_source=True) == (39, 39)


def test_resolve_crop_soft_bridge_shares_one_value():
    """软桥段：裁头量 == 采样额外量 == 钉住帧数。

    回放分支与生成分支共用本函数的 skip_f。若回放时按 ctx 裁，会把软桥段
    「段长+hold」帧的 latent 多裁 (ctx-hold) 帧——成片凭空变短且不报错。
    """
    skip, extra = bridge.resolve_crop(9, 22, has_source=True)
    assert (skip, extra) == (9, 9)
    assert skip == extra                     # 存下的 latent 长度与裁剪量必须同源
    assert skip < 22                         # 软桥的核心收益：少烧帧
    for hold in (5, 9, 13, 17, 18, 22):
        s, e = bridge.resolve_crop(hold, 22, has_source=True)
        assert s == e == hold


def test_plan_for_guide_clamps_to_bridge():
    """桥比计划短时按桥收窄——否则会把没有内容的行钉到全零 latent（黑帧/静音）。"""
    guide = {"latent": _video(t=7), "audio_latent": _audio(t=37)}
    assert bridge.plan_for_guide(guide, 22) == (22, 7, 37)     # 桥够长：照计划
    short = {"latent": _video(t=2), "audio_latent": _audio(t=8)}
    assert bridge.plan_for_guide(short, 22) == (5, 2, 8)       # 只有 2 token = 5 帧
    assert bridge.plan_for_guide(short, 5) == (5, 2, 8)
    assert bridge.plan_for_guide({}, 9) == (0, 0, 0)           # 无桥：不钉
    assert bridge.plan_for_guide(guide, 0) == (0, 0, 0)
    # 收窄后装配出的掩码与内容必须等长（不能钉空行）
    latent = {"samples": _NT((_video(), _audio()))}
    out, mask, info = bridge.apply_soft_bridge(latent, short, 22)
    mv, ma = mask.unbind()
    assert info == {"hold_frames": 5, "hold_tokens": 2, "hold_audio_tokens": 8}
    assert float(mv[0, 0, :2, 0, 0].max()) == 0.0
    assert float(mv[0, 0, 2:, 0, 0].min()) == 1.0
    vv, aa = out["samples"].unbind()
    assert torch.equal(vv[:, :, :2], short["latent"])          # 钉住区有真实内容
    assert float(ma[0, 0, 0, :8].max()) == 0.0


def test_apply_soft_bridge():
    _install_nested()
    v, a = _video(), _audio()
    latent = {"samples": _NT((v, a))}
    guide = {"latent": _video(t=7, fill=2.0), "audio_latent": _audio(t=37, fill=4.0)}
    out, mask, info = bridge.apply_soft_bridge(latent, guide, 22)
    assert info == {"hold_frames": 22, "hold_tokens": 7, "hold_audio_tokens": 37}
    assert "noise_mask" in out and out["noise_mask"] is mask
    mv, ma = mask.unbind()
    assert tuple(mv.shape) == (1, 1, 37, 48, 84)
    assert tuple(ma.shape) == (1, 1, 2, 207)
    vv, aa = out["samples"].unbind()
    assert torch.equal(vv[:, :, :7], guide["latent"])
    # 原 latent 字典不被改动（软桥关闭时仍可走原路径）
    assert "noise_mask" not in latent and latent["samples"] is not out["samples"]
    # 钉住帧数不足 1 token -> 抛错由调用方降级
    try:
        bridge.apply_soft_bridge(latent, guide, 0)
        raise AssertionError("应抛 RuntimeError")
    except RuntimeError:
        pass


def test_probe_soft_bridge_degrades_without_comfy():
    """无 ComfyUI 环境：探测必须返回等级 0 而不是抛异常。"""
    level, reason = bridge.probe_soft_bridge(force=True)
    if level != bridge.LEVEL_OFF:
        assert level in (bridge.LEVEL_SAMPLER, bridge.LEVEL_FULL)
        assert reason
    else:
        assert reason


if __name__ == "__main__":
    if torch is None:
        print("test_bridge: 无 torch 环境，跳过")
    else:
        _install_nested()
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"  ok  {name}")
        print("test_bridge: 全部通过")
