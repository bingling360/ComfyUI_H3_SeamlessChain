#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""一键修补 ComfyUI 核心：MiniMaxH3.extra_conds 的 cond_video_latents 覆盖 bug。

## 症状

带参考素材（参考图/参考视频）的多段链，非首段起采样即崩：

    File ".../comfy/ldm/minimax/model.py", line 581, in _forward
        all_video_rows[~img_update] = cond_video_rows
    RuntimeError: shape mismatch: value tensor of shape [405, 96]
    cannot be broadcast to indexing result of shape [810, 96]

## 成因

ComfyUI 0.33.0 ~ 0.33.4 的 `comfy/model_base.py` → `MiniMaxH3.extra_conds`：

    keyframes = kwargs.get("minimax_keyframes", None)
    if keyframes is not None:
        payload["cond_video_latents"] = [kf["latent"] for kf in keyframes]   # 写入
    refs = kwargs.get("minimax_refs", None)
    if refs is not None:
        payload["cond_video_latents"] = [r["latent"] for r in refs ...]      # 覆盖！

refs 分支用「=」而不是「+=」，keyframe 的 latent 被整个丢掉；而
`PackedLayout` 仍按 keyframes + refs 两者预留行数 → 供给行数只有应有的一半。

- 上游 0.34.2 起已修为 `payload.get("cond_video_latents", []) + [...]`
- 0.33.x / 0.34.0 / 0.34.1 仍带病

## 用法

    python tools/patch_comfyui_cond_video_latents.py "G:/comfyui/ComfyUI-aki-v3/ComfyUI"

- 幂等：已修过 / 已是新版写法 → 直接报告跳过，不重复改
- 改前自动备份 `comfy/model_base.py.bak`（已有 .bak 不覆盖）
- 也可不传参数，脚本会尝试从常见安装路径里找

插件侧已内置运行时兜底 `cond_video_rows_guard`（无需改核心也能跑）；
本脚本是「治本」，顺带修好其他同样混用 keyframes+refs 的 H3 节点。
改完必须重启 ComfyUI。
"""
import os
import sys

OLD = 'payload["cond_video_latents"] = [r["latent"] for r in refs if "latent" in r]'
NEW = ('payload["cond_video_latents"] = payload.get("cond_video_latents", [])'
       ' + [r["latent"] for r in refs if "latent" in r]')

# 上游 0.34.2+ 的正确写法（识别为「已修」的几种形态）
FIXED_MARKERS = (
    "payload.get(\"cond_video_latents\", [])",
    "payload.get('cond_video_latents', [])",
)

CANDIDATES = (
    "G:/comfyui/ComfyUI-aki-v3/ComfyUI",
    "D:/Comfy-Desktop/ComfyUI-Shared/ComfyUI",
    "D:/comfyui/ComfyUI",
    "C:/comfyui/ComfyUI",
)


def _candidate_roots(explicit=None):
    if explicit:
        yield explicit
        yield os.path.join(explicit, "ComfyUI")
        return
    for c in CANDIDATES:
        yield c
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for up in range(4):
        here = os.path.dirname(here)
        yield os.path.join(here, "ComfyUI")
        yield here


def _find_target(root):
    """返回 (model_base.py 路径)；root 下找不到返回 None。"""
    if not root:
        return None
    for rel in ("comfy/model_base.py", "ComfyUI/comfy/model_base.py"):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    return None


def main(argv):
    explicit = argv[1] if len(argv) > 1 else None
    target = None
    for root in _candidate_roots(explicit):
        target = _find_target(root)
        if target:
            break
    if not target:
        print("[跳过] 没找到 comfy/model_base.py。")
        print("       显式指定：python tools/patch_comfyui_cond_video_latents.py "
              "\"<你的 ComfyUI 目录>\"")
        return 0

    # newline="" —— 原样读写，不把 LF 统一改写成 CRLF（否则整文件 diff 全是换行）
    with open(target, "r", encoding="utf-8", newline="") as f:
        src = f.read()

    if "class MiniMaxH3" not in src:
        print(f"[跳过] {target} 里没有 MiniMaxH3（ComfyUI 太旧？）")
        return 0

    if NEW in src or any(m in src for m in FIXED_MARKERS):
        print(f"[已修] {target}")
        print("       cond_video_latents 已是追加写法，无需改动。")
        return 0

    if OLD not in src:
        print(f"[警告] {target}")
        print("       没匹配到预期的待修行，可能版本不同。请人工确认 "
              "MiniMaxH3.extra_conds 里 refs 分支是否用了「=」。")
        return 1

    bak = target + ".bak"
    if not os.path.exists(bak):
        with open(bak, "w", encoding="utf-8", newline="") as f:
            f.write(src)
        print(f"[备份] {bak}")

    with open(target, "w", encoding="utf-8", newline="") as f:
        f.write(src.replace(OLD, NEW, 1))
    print(f"[已修] {target}")
    print("       MiniMaxH3.extra_conds: refs 分支 cond_video_latents 由「=」改为「+=」")
    print("       请重启 ComfyUI 生效。")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
