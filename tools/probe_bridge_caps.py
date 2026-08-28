#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""软桥能力自检：确认本机 ComfyUI 是否支持「段首 latent 钉住 + 逐 token 掩码」。

## 为什么需要它

插件的「桥区软着陆」依赖两条链路同时可用：

- 链路①（采样器）：`latent["noise_mask"]` 被 `common_ksampler` 取出 →
  `CFGGuider.sample` 支持 NestedTensor 双掩码（video/audio 各一份）→
  `KSamplerX0Inpaint` 每步把掩码为 0 的行钉回 `latent_image`。
  这条链路负责「保留」能否成立。
- 链路②（模型层）：`MiniMaxH3Model.forward(denoise_mask=, audio_denoise_mask=)`
  由 `MiniMaxH3.extra_conds` 透传，模型把该行 timestep 钳到
  `1 - m*sigma`（clamp 0.999 / 1.0）。这条链路负责「模型是否知道这些行是干净的」，
  只影响质量，不影响可用性。

两条都通 = 等级 2（完整软桥）；只有①= 等级 1（保留生效、模型不知情，仍可用）；
①不通 = 等级 0（自动回退现有「钉桥 + 裁头」路径）。

## 用法（autodl / 本机均可）

    cd <你的 ComfyUI 目录>
    python <插件目录>/tools/probe_bridge_caps.py

脚本只读不写、不加载模型权重，秒级完成；输出可直接贴回 issue / 报告。
"""
import inspect
import os
import sys

LEVEL_NAMES = {
    0: "0 = 不可用（自动回退现有钉桥+裁头路径）",
    1: "1 = 仅采样器噪声掩码（保留生效，模型侧无逐行 timestep）",
    2: "2 = 采样器 + 模型逐行 timestep（完整软桥）",
}


def _line(title, value):
    print(f"  {title:<34} {value}")


def _source_of(obj):
    try:
        return inspect.getsource(obj)
    except Exception:
        return ""


def _has_param(func, name):
    try:
        return name in inspect.signature(func).parameters
    except Exception:
        return False


def _comfy_version():
    for mod in ("comfyui_version",):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", None)
            if v:
                return str(v)
        except Exception:
            pass
    try:  # 兜底：从仓库根目录的 pyproject.toml 读
        import comfy
        root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(comfy.__file__))))
        p = os.path.join(root, "pyproject.toml")
        with open(p, "r", encoding="utf-8") as f:
            for ln in f:
                if ln.strip().startswith("version"):
                    return ln.split("=", 1)[1].strip().strip('"')
    except Exception:
        pass
    return "未知"


def main():
    print("=" * 78)
    print("H3 Seamless Chain —— 软桥能力自检")
    print("=" * 78)

    try:
        import torch
    except Exception as e:  # pragma: no cover
        print(f"[致命] 无法导入 torch：{e}")
        return 2
    _line("torch", torch.__version__)

    try:
        import comfy  # noqa: F401
    except Exception as e:
        print(f"[致命] 无法导入 comfy（请在 ComfyUI 目录下运行，或设置 PYTHONPATH）：{e}")
        return 2
    _line("ComfyUI 版本", _comfy_version())

    # ---------- 1. 官方 AddGuide 节点 ----------
    print("\n[1] 官方 MiniMaxH3AddGuide（任意帧锚定 / 多锚点串联）")
    try:
        from comfy_extras import nodes_minimax_h3 as _h3
        has_guide = hasattr(_h3, "MiniMaxH3AddGuide")
        _line("MiniMaxH3AddGuide", "存在" if has_guide else "缺失（ComfyUI < v0.34.0？）")
    except Exception as e:
        has_guide = False
        _line("MiniMaxH3AddGuide", f"导入失败：{e}")

    # ---------- 2. 模型层逐行 timestep ----------
    print("\n[2] 模型层（MiniMaxH3Model）逐 token 掩码参数")
    model_level = 0
    try:
        from comfy.ldm.minimax import model as _mm
        fwd = getattr(_mm, "MiniMaxH3Model").forward
        v_ok = _has_param(fwd, "denoise_mask")
        a_ok = _has_param(fwd, "audio_denoise_mask")
        _line("forward(denoise_mask=)", "有" if v_ok else "无")
        _line("forward(audio_denoise_mask=)", "有" if a_ok else "无")
        if v_ok and a_ok:
            model_level = 1
    except Exception as e:
        _line("MiniMaxH3Model.forward", f"导入/解析失败：{e}")

    # ---------- 3. extra_conds 是否透传 ----------
    print("\n[3] MiniMaxH3.extra_conds 是否透传掩码到模型")
    extra_ok = False
    try:
        import comfy.model_base as _mb
        cls = getattr(_mb, "MiniMaxH3", None)
        if cls is None:
            _line("class MiniMaxH3", "未找到（版本过旧）")
        else:
            src = _source_of(getattr(cls, "extra_conds", None))
            if not src:
                _line("extra_conds", "无法取源码")
            else:
                v_in = "denoise_mask" in src
                a_in = "audio_denoise_mask" in src
                _line("源码含 denoise_mask", "是" if v_in else "否")
                _line("源码含 audio_denoise_mask", "是" if a_in else "否")
                extra_ok = v_in and a_in
                if extra_ok:
                    model_level = 2
    except Exception as e:
        _line("comfy.model_base", f"导入失败：{e}")

    # ---------- 4. 采样器 NestedTensor 双掩码 ----------
    print("\n[4] 采样器：NestedTensor 双掩码（video/audio 各一份）")
    nested_ok = False
    try:
        import comfy.samplers as _sp
        src = _source_of(getattr(_sp, "CFGGuider").sample)
        nested_ok = "denoise_mask.is_nested" in src
        _line("CFGGuider.sample 支持 is_nested", "是" if nested_ok else "否")
    except Exception as e:
        _line("comfy.samplers", f"导入失败：{e}")

    # ---------- 5. common_ksampler 取掩码的入口 ----------
    print("\n[5] 掩码入口：latent['noise_mask']（官方 SetLatentNoiseMask 协议）")
    latent_key_ok = False
    try:
        import nodes as _cn
        src = _source_of(getattr(_cn, "common_ksampler"))
        latent_key_ok = '"noise_mask" in latent' in src or "'noise_mask' in latent" in src
        _line("读取 latent['noise_mask']", "是" if latent_key_ok else "否（需改走 conditioning 路径）")
        cond_key = '"denoise_mask" in positive' in src or "'denoise_mask' in positive' " in src
        _line("读取 positive[0][1]['denoise_mask']", "是" if cond_key else "否")
    except Exception as e:
        _line("nodes.common_ksampler", f"导入失败：{e}")

    # ---------- 6. 功能性探针：掩码整形 ----------
    print("\n[6] 功能性探针：掩码整形到 H3 的 video/audio latent 形状")
    shape_ok = False
    try:
        import comfy.sampler_helpers as _sh
        v = torch.zeros(1, 1, 2, 2, 2)
        a = torch.zeros(1, 1, 2, 4)
        mv = _sh.prepare_mask(v, (1, 24, 2, 2, 2), "cpu")
        ma = _sh.prepare_mask(a, (1, 32, 2, 4), "cpu")
        v_ok = tuple(mv.shape) == (1, 24, 2, 2, 2)
        a_ok = tuple(ma.shape) == (1, 32, 2, 4)
        _line("video mask -> (1,24,T,H,W)", f"{tuple(mv.shape)} {'OK' if v_ok else 'FAIL'}")
        _line("audio mask -> (1,32,2,Ta)", f"{tuple(ma.shape)} {'OK' if a_ok else 'FAIL'}")
        shape_ok = v_ok and a_ok
    except Exception as e:
        _line("prepare_mask", f"失败：{e}")

    # ---------- 7. 结论 ----------
    print("\n[7] 结论")
    if latent_key_ok and nested_ok and shape_ok:
        level = 1 + (1 if model_level == 2 else 0)
    else:
        level = 0
    _line("软桥等级", LEVEL_NAMES[level])
    if level == 0:
        print("  → 插件将自动回退现有「钉桥 + 裁头」路径，行为与本版本之前完全一致。")
    elif level == 1:
        print("  → 可用：段首 latent 会被钉住；建议升级 ComfyUI 到含模型层掩码的版本以获得更好质量。")
    else:
        print("  → 完全可用：保留区在模型侧也被标记为干净行，续拍衔接质量最佳。")
    if not has_guide:
        print("  → 注意：未检测到 MiniMaxH3AddGuide，多锚点/任意帧引导语义对齐功能不可用（不影响软桥）。")
    print("\n" + "=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
