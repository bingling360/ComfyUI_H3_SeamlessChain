"""A/B 实验统一出口：对比两个存档目录的 manifest，量化接缝与丢帧行为差异。

用法（任意 python，纯标准库，无 ComfyUI 依赖）：
    python tools/ab_report.py <存档A目录> <存档B目录> [--md 报告.md]

输出三块：
1. 每缝对比：帧差 + 五维 z-score（光流/加速度/LPIPS/嵌入/相机，|z|<2 合格）
2. 帧数守恒与丢弃：计划时长 vs 实际、trims 明细、全链丢弃占比
3. 参数指纹差异：A/B 唯一不同之处（单变量实验的自检——指纹应只差在
   实验开关上，否则结论不可信）

A/B 规程（README「评测与 A/B」）：同种子同提示词跑两版，只改一个开关；
验收线：边界光流 z < 2.0、累计丢弃 ≤ 预算、精修后缝差 ≤ 精修前。
"""

import argparse
import json
import math
import os
import sys

Z_KEYS = ["flow_z", "flow_accel_z", "lpips_z", "emb_z", "cam_z", "pose_z"]
Z_LABELS = {"flow_z": "光流", "flow_accel_z": "加速度", "lpips_z": "LPIPS",
            "emb_z": "嵌入", "cam_z": "相机", "pose_z": "姿态"}


def load(root):
    path = os.path.join(root, "manifest.json")
    if not os.path.isfile(path):
        sys.exit(f"找不到 {path}——请传存档目录（含 manifest.json）")
    with open(path, "r", encoding="utf-8-sig") as f:
        return json.load(f)


def fmt(v, digits=2):
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:+.{digits}f}"
    return str(v)


def fmt_abs(v, digits=3):
    return "—" if v is None else f"{v:.{digits}f}"


def seam_rows(mani):
    """[(段号, seams[i], seam_metrics[i] or None)]，跳过首段占位 None。"""
    seams = mani.get("seams", [])
    zm = mani.get("seam_metrics", [])
    rows = []
    for i, s in enumerate(seams):
        if not s:
            continue
        z = zm[i] if i < len(zm) else None
        rows.append((i + 1, s, z))
    return rows


def drop_stats(mani):
    trims = mani.get("trims", [])
    params = mani.get("params", {})
    length = params.get("length")
    done = mani.get("done", len(trims))
    total = mani.get("total", done)
    drop = sum(t for t in trims if t)
    planned = sum(length for _ in trims) if length else None
    return {"drop": drop, "trims": trims, "planned": planned,
            "kept": (planned - drop) if planned else None,
            "done": done, "total": total, "length": length}


def build(tag_a, mani_a, tag_b, mani_b):
    L = []
    w = L.append
    w("══ H3 无缝链 A/B 对比 ══")
    w(f"A = {tag_a}    B = {tag_b}")
    w("")

    # —— 1. 参数指纹差异（单变量自检）
    pa, pb = mani_a.get("params", {}), mani_b.get("params", {})
    keys = list(dict.fromkeys(list(pa) + list(pb)))
    diffs = [(k, pa.get(k, "<无>"), pb.get(k, "<无>")) for k in keys if pa.get(k) != pb.get(k)]
    w("── 参数差异（应为空或只有实验开关；种子里程碑见 seeds）──")
    if diffs:
        for k, va, vb in diffs:
            w(f"  {k}: A={va!r}  B={vb!r}")
    else:
        w("  （指纹完全一致）")
    seeds_a, seeds_b = mani_a.get("seeds", []), mani_b.get("seeds", [])
    same_seeds = seeds_a and seeds_b and seeds_a[:min(len(seeds_a), len(seeds_b))] == \
        seeds_b[:min(len(seeds_a), len(seeds_b))]
    w(f"  种子序列前缀一致：{'是' if same_seeds else '否（A/B 不同种子，仅可比分布不可比逐段）'}")
    w("")

    # —— 2. 每缝对比表
    rows_a, rows_b = seam_rows(mani_a), seam_rows(mani_b)
    w("── 接缝对比（帧差越小越好；z 绝对值越小越好，|z|<2 合格）──")
    if not rows_a and not rows_b:
        w("  （两个存档都没有已测接缝——需 ≥2 段且完成生成）")
    header = f"  {'缝':<4} {'A帧差':>7} {'B帧差':>7} {'Δ':>7} " + \
        " ".join(f"{Z_LABELS[k]} A/B".ljust(13) for k in Z_KEYS)
    w(header)
    for idx in sorted({r[0] for r in rows_a} | {r[0] for r in rows_b}):
        da = next((r for r in rows_a if r[0] == idx), None)
        db = next((r for r in rows_b if r[0] == idx), None)
        dva = da[1][0] if da else None
        dvb = db[1][0] if db else None
        delta = ("—" if dva is None or dvb is None
                 else f"{dvb - dva:+.3f}" if abs(dvb - dva) >= 0.0005 else "0.000")
        zcells = []
        for k in Z_KEYS:
            za = da[2].get(k) if da and da[2] else None
            zb = db[2].get(k) if db and db[2] else None
            zcells.append(f"{fmt(za)}/{fmt(zb)}".ljust(13))
        w(f"  {idx:<4} {fmt_abs(dva):>7} {fmt_abs(dvb):>7} {delta:>7} " + " ".join(zcells))

    def mean(vals):
        vals = [v for v in vals if v is not None]
        return sum(vals) / len(vals) if vals else None

    for name, rows in (("A", rows_a), ("B", rows_b)):
        ds = [r[1][0] for r in rows if r[1]]
        fzs = [r[2].get("flow_z") for r in rows if r[2]]
        if ds:
            w(f"  {name} 汇总：接缝 {len(ds)} 个 · 平均帧差 {mean(ds):.3f} · 最差 {max(ds):.3f}"
              + (f" · 光流 z 均值 {mean(fzs):+.1f} / 最差 {max(v for v in fzs if v is not None):+.1f}σ"
                 if any(v is not None for v in fzs) else ""))
    w("")

    # —— 3. 帧数守恒与丢弃
    w("── 帧数守恒与丢弃（止损验证：A/B 的 drop 应 ≤ params.drop_budget）──")
    for name, mani in (("A", mani_a), ("B", mani_b)):
        st = drop_stats(mani)
        budget = mani.get("params", {}).get("drop_budget")
        pct = (st["drop"] / st["planned"] * 100.0) if st["planned"] else None
        w(f"  {name}: 完成 {st['done']}/{st['total']} 段 · 计划 {st['planned'] or '—'} 帧"
          f" · 丢弃 {st['drop']} 帧（{st['drop'] / 24:.1f}s"
          + (f"，占 {pct:.1f}%" if pct is not None else "") + ")"
          + (f" · 预算 {budget} 帧 {'✓ 未超' if st['drop'] <= budget else '⚠ 超预算'}"
             if budget is not None else ""))
        nz = [(i + 1, t) for i, t in enumerate(st["trims"]) if t]
        if nz:
            w(f"     trims 明细：{'  '.join(f'段{i}:{t}' for i, t in nz)}")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser(description="H3 无缝链 A/B 存档对比")
    ap.add_argument("archive_a", help="存档 A 目录（基准版）")
    ap.add_argument("archive_b", help="存档 B 目录（实验版）")
    ap.add_argument("--md", default="", help="可选：同时写 markdown 报告到该路径")
    args = ap.parse_args()

    mani_a, mani_b = load(args.archive_a), load(args.archive_b)
    text = build(os.path.basename(os.path.normpath(args.archive_a)), mani_a,
                 os.path.basename(os.path.normpath(args.archive_b)), mani_b)
    print(text)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write("```\n" + text + "\n```\n")
        print(f"\n已写入 {args.md}")


if __name__ == "__main__":
    main()
