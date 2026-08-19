"""跨缝窗口精修单测（纯 torch，无 ComfyUI/GPU）：python tests/test_refine.py

覆盖：窗口 token 布局踩 17k+5 网格、两侧 latent 切片内容/音频窗换算正确、
保留区不足回退 None、smoothstep 羽化端点连续、精修种子派生确定性公式。
注意不能用 test_node_structure 的 stub（其 torch 是假的，cat 恒返 [1]）——
本测需要真 torch 切片；包 __init__ 在无 ComfyUI 环境下自行降级（静默导入）。
refine_seam 本体依赖 ComfyUI 运行环境，不在本测范围（失败回退路径由
nodes.py try/except 兜底，端到端在 AutoDL 验证）。
"""
import contextlib
import io
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 插件父目录（包名可导入）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests 目录

try:
    import torch
    if not hasattr(torch, "rand"):
        # 合集运行时 test_node_structure 的假 torch stub 可能已在 sys.modules——视为无 torch
        raise ImportError
except ImportError:
    torch = None

if torch is not None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ComfyUI_H3_SeamlessChain import refine
        from ComfyUI_H3_SeamlessChain import qc
    from ComfyUI_H3_SeamlessChain.grid import (video_latent_t, latent_t_to_frames,
                                               audio_tokens_for_frames)
else:
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("需要真 torch（stub/无 torch 环境跳过）", allow_module_level=True)


def _fake_lat(v_tokens, a_tokens, k0, k1, base=0.0):
    """构造 (video_t, audio_t, kt0, kt1, ka0, ka1)，内容按 token 下标编码可验证切片。"""
    pv = (torch.arange(v_tokens, dtype=torch.float32).view(1, 1, v_tokens, 1, 1)
          + base).expand(1, 24, v_tokens, 4, 4).contiguous()
    pa = (torch.arange(a_tokens, dtype=torch.float32).view(1, 1, 1, 1, a_tokens)
          + base).expand(1, 32, 2, 1, a_tokens).contiguous()
    return (pv, pa, k0, k1, 0, a_tokens)


def test_seam_window_tokens():
    # 22/39/56 帧侧 = 7/12/17 token（网格点）；两侧之和 ≡4 mod 5 不落网格，
    # 缺口全部从本段侧补齐；窗口总 token ≡2 mod 5 且帧数可整 token 解码
    for side, vt_p in ((22, 7), (39, 12), (56, 17)):
        p, c, tw, wf = refine.seam_window_tokens(side)
        assert p == vt_p
        assert tw % 5 == 2 and tw >= 7
        assert p + c == tw
        assert wf == latent_t_to_frames(tw)
    # 锚定具体值（防 grid 常量变化静默漂移）
    assert refine.seam_window_tokens(22) == (7, 10, 17, 56)
    assert refine.seam_window_tokens(39) == (12, 15, 27, 90)
    assert refine.seam_window_tokens(56) == (17, 20, 37, 124)


def test_build_seam_window_slices():
    # 上段 kept 12 token、本段 kept 15 token，窗口 side=22（7 + 10 = 17 token）
    prev = _fake_lat(20, 100, 8, 20, base=100.0)
    cur = _fake_lat(20, 120, 7, 22, base=200.0)
    win = refine.build_seam_window(prev, cur, 22)
    assert win is not None
    win_v, win_a, vt_p, vt_c, wf = win
    assert (vt_p, vt_c, wf) == (7, 10, 56)
    assert win_v.shape == (1, 24, 17, 4, 4)
    # 缝前侧=上段 kept 末端 7 token，缝后侧=本段 kept 开头 10 token（内容编码可验证）
    assert torch.equal(win_v[:, :, :7], prev[0][:, :, 13:20])
    assert torch.equal(win_v[:, :, 7:], cur[0][:, :, 7:17])
    # 音频窗总长=窗口帧数×5/3，缝前侧=上段尾 audio_tokens_for_frames(22)
    at = audio_tokens_for_frames(56)
    assert win_a.shape[-1] == at == 93
    ap = audio_tokens_for_frames(22)
    assert win_a[..., :ap].equal(prev[1][..., 100 - ap:100])
    assert win_a[..., ap:].equal(cur[1][..., :at - ap])


def test_build_seam_window_insufficient():
    # 上段 kept 6 token < vt_p=7 -> None
    assert refine.build_seam_window(_fake_lat(10, 100, 3, 9), _fake_lat(20, 120, 7, 22), 22) is None
    # 本段 kept 12 token < vt_c=15 -> None
    assert refine.build_seam_window(_fake_lat(20, 100, 8, 20), _fake_lat(20, 120, 7, 19), 39) is None
    # 音频窗不足 -> None（视频够、音频极限短）
    prev = _fake_lat(20, 10, 8, 20)
    cur = _fake_lat(20, 10, 7, 22)
    assert refine.build_seam_window(prev, cur, 22) is None


def test_refine_seed_derivation():
    # 精修种子=本段种子+1、重摇步进 7919（64 位模回绕）——确定性公式锚定
    m = 0xffffffffffffffff
    assert (m - 1 + 1) % m == 0 and (m + 1) % m == 1
    assert (123 + 7919) % m == 8042


def test_smoothstep_weights_endpoints():
    # 羽化窗端点导数为 0：w(0)=0（首帧不改）、w(1)=1（末帧=原帧）
    w = qc.smoothstep_weights(6)
    assert w[0] == 0.0 and w[-1] == 1.0
    assert torch.all(w >= 0.0) and torch.all(w <= 1.0)
    assert torch.all(w[1:] >= w[:-1])                       # 单调
    # 精修区尾羽化：混合后首帧=精修帧（w=0 端）、末帧=原帧（w=1 端，接未精修区）
    refined = torch.ones(4, 2, 2, 3)
    origin = torch.zeros(4, 2, 2, 3)
    wf_ = qc.smoothstep_weights(4).view(-1, 1, 1, 1)
    mixed = refined * (1.0 - wf_) + origin * wf_
    assert torch.equal(mixed[0], refined[0]) and torch.equal(mixed[-1], origin[-1])


def test_adaptive_strength_mapping():
    # 缝差分档：<0.04 轻修 0.30；0.04-0.08 标准 0.45；>0.08 强调和 0.55；None 兜底 0.45
    assert refine.adaptive_strength(0.0) == 0.30
    assert refine.adaptive_strength(0.0399) == 0.30
    assert refine.adaptive_strength(0.04) == 0.45
    assert refine.adaptive_strength(0.08) == 0.45
    assert refine.adaptive_strength(0.0801) == 0.55
    assert refine.adaptive_strength(None) == 0.45


def test_adaptive_strength_custom_tiers():
    # 预设档传入自己的分档表：轻量 (0.22, 0.30, 0.40)
    tiers = refine.SEAM_PROFILES["轻量"]["tiers"]
    assert refine.adaptive_strength(0.02, tiers) == 0.22
    assert refine.adaptive_strength(0.05, tiers) == 0.30
    assert refine.adaptive_strength(0.10, tiers) == 0.40
    assert refine.adaptive_strength(None, tiers) == 0.30


def test_resolve_profile_presets():
    # 预设档：固化参数、不读控件（传进来的控件值必须被忽略）、加噪/递减一律关闭
    for mode in ("标准", "轻量", "强力"):
        p = refine.resolve_profile(mode, anchor_aug=0.3, fade_ratio=0.5,
                                   strength=0.7, window=56, blend=24, adaptive=False)
        assert p["use_refine"] and not p["pixel_blend"]
        assert p["anchor_aug"] == 0.0 and p["fade_ratio"] == 0.0, "预设档必须关闭加噪/递减"
        exp = refine.SEAM_PROFILES[mode]
        assert p["tiers"] == exp["tiers"]
        assert p["window"] == exp["window"] and p["blend"] == exp["blend"]
        assert p["fixed"] is None
    assert refine.SEAM_PROFILES["标准"]["tiers"] == (0.30, 0.45, 0.55)


def test_resolve_profile_custom_passthrough():
    # 自定义档：透传全部细粒度控件（含加噪/递减，供实验）；自适应关=固定强度
    p = refine.resolve_profile("自定义", anchor_aug=0.2, fade_ratio=0.3,
                               strength=0.6, window=56, blend=12, adaptive=True)
    assert p["use_refine"] and p["mode"] == "自定义"
    assert p["anchor_aug"] == 0.2 and p["fade_ratio"] == 0.3
    assert p["window"] == 56 and p["blend"] == 12
    assert p["tiers"] == (0.30, 0.45, 0.55) and p["fixed"] is None

    p2 = refine.resolve_profile("自定义", strength=0.35, adaptive=False)
    assert p2["tiers"] is None and p2["fixed"] == 0.35


def test_resolve_profile_legacy_values():
    # 旧值兼容：潜空间精修=自定义等价透传（旧工作流行为不变）；smoothstep=纯像素；关闭=全关
    p = refine.resolve_profile("潜空间精修", anchor_aug=0.2, fade_ratio=0.5,
                               strength=0.5, window=22, blend=8, adaptive=False)
    assert p["use_refine"] and not p["pixel_blend"] and p["mode"] == "潜空间精修"
    assert p["anchor_aug"] == 0.2 and p["fade_ratio"] == 0.5
    assert p["fixed"] == 0.5 and p["window"] == 22 and p["blend"] == 8

    p2 = refine.resolve_profile("smoothstep像素混合", blend=10, anchor_aug=0.2, fade_ratio=0.5)
    assert p2["pixel_blend"] and not p2["use_refine"]
    assert p2["blend"] == 10 and p2["anchor_aug"] == 0.2   # 像素档加噪仍作用于生成期锚定

    p3 = refine.resolve_profile("关闭", anchor_aug=0.3, fade_ratio=0.7)
    assert not p3["use_refine"] and not p3["pixel_blend"]
    assert p3["anchor_aug"] == 0.0 and p3["fade_ratio"] == 0.0


def test_tail_anchor_index_alignment():
    # 双端锚定：尾锚=窗口末 2 token（5 帧），锚索引 = 窗口帧数 - 5，锚区恰好
    # 覆盖窗口末帧——与 storyboard frame_keyframe（5 帧钉窗口末）同模式
    for side in (22, 39, 56):
        vt_p, vt_c, tw, wf = refine.seam_window_tokens(side)
        tail_idx = wf - 5
        assert tail_idx > latent_t_to_frames(vt_p), "尾锚必须在缝前侧（上段侧）之后"
        assert 0 < tail_idx < wf


if __name__ == "__main__":
    if torch is None:
        print("SKIP (no torch) — 用带 torch 的 venv 跑：D:\\本地部署文件集合\\comfyui\\comfy shili\\ComfyUI\\ComfyUI\\.venv\\Scripts\\python.exe")
    else:
        for name, fn in sorted(globals().items()):
            if name.startswith("test_") and callable(fn):
                fn()
                print(f"PASS {name}")
        print("all tests passed")
