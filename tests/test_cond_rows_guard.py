"""cond_video_rows_guard / cond_audio_rows_guard 单测：python tests/test_cond_rows_guard.py

背景（2026-08-28 报障）：带参考图的多段链，第二段起必崩——

    RuntimeError: shape mismatch: value tensor of shape [405, 96]
    cannot be broadcast to indexing result of shape [810, 96]

405 = 864×480 基础画幅单帧的 2×2 patch 行数（latent 30×54 → 15×27），
810 = 「段间桥 keyframe + 参考图」两份。成因是 ComfyUI 0.33.0~0.33.4 的
MiniMaxH3.extra_conds 在 refs 分支用「=」覆盖了 keyframes 分支刚写入的
cond_video_latents（0.34.2 起才改成 +=），keyframe latent 被整个丢掉，
而 PackedLayout 仍按 keyframes+refs 两者预留行数。

本测用假 dit（真 patchify 逻辑，不依赖 torch——只需行数口径）验证：
  1. 官方 buggy payload（只有 refs）→ 守卫重建为 kf+refs，行数翻倍正确；
  2. 官方正确 payload（0.34.2+ 已是 kf+refs）→ 守卫零干预（身份相同）；
  3. 只有 keyframes 或只有 refs → 不改动；
  4. restore 后 dit 恢复原方法；
  5. 音频守卫同类覆盖（keyframe 音频被丢）→ 按 keyframe+refs 顺序重建。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests.test_node_structure import _install_stubs  # noqa: E402

_install_stubs(with_audio_support=False)

from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes  # noqa: E402


# ---- 假 dit：复刻官方 _cond_video_rows / _cond_audio_rows 的行数口径 ----
# patchify_video: [1, C, T, H, W] -> (T/1) * (H/2) * (W/2) 行，每行 C*4 维
# pack_audio:     [1, C, 2, T] -> T*2 行


class _Z:
    """假 latent：只需 shape（[1,24,T,H,W] 或音频 [1,32,2,T]）与 .to()。"""

    def __init__(self, shape, tag=""):
        self.shape = tuple(shape)
        self.tag = tag

    def to(self, *_a, **_k):
        return self


class _Rows:
    """假张量：守卫只按 .shape[0] 取行数（音频守卫的口径）。"""

    def __init__(self, n, order=()):
        self.shape = (n, 96)
        self.order = list(order)      # 记录本轮喂进来的 latent 顺序，供断言

    def __int__(self):
        return self.shape[0]


class _FakeDit:
    patch_size = (1, 2, 2)

    def __init__(self):
        self.seen = None      # 最近一次调用实际收到的 payload

    def _cond_video_rows(self, payload, device):
        self.seen = payload
        rows = 0
        order = []
        for z in payload.get("cond_video_latents", []):
            _, _c, t, h, w = z.shape
            pt, ph, pw = self.patch_size
            rows += (t // pt) * (h // ph) * (w // pw)
            order.append(z)
        return _Rows(rows, order) if rows else None

    def _cond_audio_rows(self, payload, device):
        self.seen = payload
        rows = 0
        order = []
        for z in payload.get("cond_audio_latents", []):
            rows += int(z.shape[-1]) * 2
            order.append(z)
        return _Rows(rows, order) if rows else None


# 864×480 基础画幅：latent 30×54 → 15×27 = 405 行/帧
FRAME_ROWS = 405


def _kf_latent(tag="kf"):
    return _Z((1, 24, 1, 30, 54), tag)


def _ref_latent(tag="ref"):
    return _Z((1, 24, 1, 30, 54), tag)


# ---- 1. 官方 0.33.x 的 buggy payload：refs 分支覆盖了 keyframes 分支 ----


def test_guard_repairs_clobbered_keyframe_latent():
    dit = _FakeDit()
    kf, ref = _kf_latent(), _ref_latent()
    # extra_conds 0.33.x 的实际产物：keyframes/refs 都在，latents 只剩 refs
    payload = {"keyframes": [{"resolved_frame_index": 0, "latent": kf}],
               "refs": [{"kind": "image", "latent_h": 30, "latent_w": 54, "latent": ref}],
               "cond_video_latents": [ref]}
    assert int(dit._cond_video_rows(payload, "cuda")) == FRAME_ROWS      # 修复前：405
    restore = plugin_nodes.cond_video_rows_guard(dit)
    try:
        got = dit._cond_video_rows(payload, "cuda")
        assert int(got) == 2 * FRAME_ROWS, f"守卫后应补齐到 810 行，实际 {int(got)}"
        # 顺序必须是 keyframe 在前、refs 在后（PackedLayout 段顺序）
        assert got.order == [kf, ref]
    finally:
        restore()
    # 守卫不改动调用方的 payload 本体（写时复制）
    assert payload["cond_video_latents"] == [ref]


def test_guard_restores_original_method():
    dit = _FakeDit()
    orig = dit._cond_video_rows
    restore = plugin_nodes.cond_video_rows_guard(dit)
    assert dit._cond_video_rows is not orig
    restore()
    # 绑定方法每次取值都是新对象，比相等不比同一性
    assert dit._cond_video_rows == orig


# ---- 2. 官方 0.34.2+ 已是 kf+refs：守卫零干预 ----


def test_guard_noop_when_already_correct():
    dit = _FakeDit()
    kf, ref = _kf_latent(), _ref_latent()
    payload = {"keyframes": [{"resolved_frame_index": 0, "latent": kf}],
               "refs": [{"kind": "image", "latent": ref}],
               "cond_video_latents": [kf, ref]}
    restore = plugin_nodes.cond_video_rows_guard(dit)
    try:
        assert int(dit._cond_video_rows(payload, "cuda")) == 2 * FRAME_ROWS
        assert payload["cond_video_latents"] == [kf, ref]   # 原样
    finally:
        restore()


# ---- 3. 只有一侧时不动（首段只有参考图 / 无参考图只有桥）----


def test_guard_noop_single_side():
    dit = _FakeDit()
    restore = plugin_nodes.cond_video_rows_guard(dit)
    try:
        ref = _ref_latent()
        p_ref = {"refs": [{"kind": "image", "latent": ref}], "cond_video_latents": [ref]}
        assert int(dit._cond_video_rows(p_ref, "cuda")) == FRAME_ROWS

        kf = _kf_latent()
        p_kf = {"keyframes": [{"resolved_frame_index": 0, "latent": kf}],
                "cond_video_latents": [kf]}
        assert int(dit._cond_video_rows(p_kf, "cuda")) == FRAME_ROWS

        # 完全无条件（纯 t2v）：cond_video_rows 为 None，守卫不得臆造
        assert dit._cond_video_rows({}, "cuda") is None
    finally:
        restore()


# ---- 4. 无 latent 的 keyframe（纯音频锚）不参与视频行 ----


def test_guard_skips_latentless_keyframe():
    dit = _FakeDit()
    ref = _ref_latent()
    payload = {"keyframes": [{"resolved_frame_index": 0, "audio_latent": _Z((1, 32, 2, 4))}],
               "refs": [{"kind": "image", "latent": ref}],
               "cond_video_latents": [ref]}
    restore = plugin_nodes.cond_video_rows_guard(dit)
    try:
        # want == supplied（都是 [ref]）→ 不改动，行数仍是 405
        assert int(dit._cond_video_rows(payload, "cuda")) == FRAME_ROWS
    finally:
        restore()


# ---- 5. 音频守卫同类覆盖（keyframe 音频被共存插件丢掉）----


def test_audio_guard_rebuilds_in_layout_order():
    dit = _FakeDit()
    kf_a, ref_a = _Z((1, 32, 2, 8), "kf_a"), _Z((1, 32, 2, 5), "ref_a")
    payload = {"keyframes": [{"resolved_frame_index": 0, "audio_latent": kf_a}],
               "refs": [{"kind": "audio", "ref_audio_t": 5, "audio_latent": ref_a}],
               "cond_audio_latents": [ref_a]}
    assert int(dit._cond_audio_rows(payload, "cuda")) == 10
    restore = plugin_nodes.cond_audio_rows_guard(dit)
    try:
        got = dit._cond_audio_rows(payload, "cuda")
        assert int(got) == 16 + 10, f"应为 kf(8*2)+ref(5*2)=26，实际 {int(got)}"
        assert got.order == [kf_a, ref_a]
    finally:
        restore()


if __name__ == "__main__":
    cases = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in cases:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(cases)} 项全部通过")
