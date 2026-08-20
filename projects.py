"""游戏式项目存档：output/h3_projects/ 下一个项目一个文件夹。

list_projects 扫描磁盘实时列项目（不依赖索引文件或前端 localStorage，
永不失真）；delete_project 删项目 = 删整个文件夹（视频/提示词/latent 全删）。
本模块不导入 routes（routes 单向导入本模块，避免循环依赖）。
"""

import json
import os
import shutil
import time

from . import checkpoint


def safe_name(name) -> str:
    """项目目录名安全校验：拒空、路径分隔、盘符、与点开头（防目录穿越）。"""
    s = str(name or "").strip()
    if not s or s.startswith(".") or "/" in s or "\\" in s or ":" in s or ".." in s:
        return ""
    return s


def _cover(root: str, manifest: dict) -> str:
    """第一张实际存在的段缩略图文件名（列表里可能有空串占位）。"""
    for name in (manifest.get("thumbs") or []):
        if name and os.path.isfile(os.path.join(root, name)):
            return name
    return ""


def list_projects() -> list:
    """扫描项目根目录 -> 摘要列表（按 updated_at 倒序）。

    无 manifest.json 或 manifest 损坏的目录直接跳过（不是本项目系统的产物）。
    """
    root = checkpoint.projects_root()
    if not os.path.isdir(root):
        return []
    out = []
    for name in os.listdir(root):
        pdir = os.path.join(root, name)
        if not os.path.isdir(pdir) or not safe_name(name):
            continue
        try:
            with open(os.path.join(pdir, "manifest.json"), "r", encoding="utf-8") as f:
                manifest = json.load(f)
        except Exception:
            continue
        if not isinstance(manifest, dict):
            continue
        updated = manifest.get("updated_at")
        if not isinstance(updated, (int, float)):
            try:
                updated = os.path.getmtime(os.path.join(pdir, "manifest.json"))
            except OSError:
                updated = 0
        out.append({
            "dir": name,
            "title": manifest.get("title") or name,
            "done": int(manifest.get("done") or 0),
            "total": int(manifest.get("total") or 0),
            "updated_at": updated,
            "cover": _cover(pdir, manifest),
            "finals": list(manifest.get("finals") or []),
            "params": manifest.get("params") or {},
        })
    out.sort(key=lambda p: p["updated_at"] or 0, reverse=True)
    return out


def read_project(name: str):
    """读单个项目 manifest 全文；不存在/非法返回 None。"""
    if not safe_name(name):
        return None
    root = os.path.join(checkpoint.projects_root(), safe_name(name))
    manifest = checkpoint.load_manifest(root)
    return manifest if isinstance(manifest, dict) else None


def save_prompts(name: str, prompts):
    """把导演台当前提示词组回写进项目 manifest（提示词的持久源=项目文件夹）。

    只更新 prompts / total / updated_at（及 done 超界钳制），不动
    params / seeds / prompt_hashes / finals——运行时的重做判定依旧按
    节点控件提示词 vs prompt_hashes 逐段比对，改词段落照常自动重做。
    序章项目（has_prologue）自动补回「序章」占位头，与运行时写盘格式一致。
    目录或 manifest 不存在（未跑过的指纹目录）返回 None，调用方按无项目跳过。
    """
    name = safe_name(name)
    if not name or not isinstance(prompts, list):
        return None
    root = os.path.join(checkpoint.projects_root(), name)
    manifest = checkpoint.load_manifest(root)
    if manifest is None:
        return None
    seg_prompts = [str(p) for p in prompts][:64]
    off = 1 if manifest.get("has_prologue") else 0
    manifest["prompts"] = (["「序章（上传视频）」"] if off else []) + seg_prompts
    manifest["total"] = len(seg_prompts) + off
    manifest["done"] = min(int(manifest.get("done") or 0), manifest["total"])
    manifest["updated_at"] = time.time()
    checkpoint.save_manifest(root, manifest)
    return manifest


def create_project(name: str):
    """新建项目：当场建文件夹 + 写初始 manifest（0 段草稿态），列表立即可见。

    游戏存档槽语义——此前文件夹要等首次运行 ckpt_dir() 才建、manifest 要等
    首段采样完才写，点完「新建项目」磁盘上什么都没有，项目也不在列表里。
    params 留空：首跑 assert_match 对空旧档按「沿用当前值」放行（旧档缺键
    视为一致），跑完由真实参数覆写；title/created_at 首跑会被继承保留。
    已存在（含跑过一段以上的正式项目）直接幂等返回现 manifest，不重写。
    """
    name = safe_name(name)
    if not name:
        return None
    root = os.path.join(checkpoint.projects_root(), name)
    manifest = checkpoint.load_manifest(root)
    if manifest is not None:
        return manifest
    os.makedirs(root, exist_ok=True)
    now = time.time()
    manifest = {
        "schema": checkpoint.SCHEMA, "done": 0, "total": 0,
        "title": name, "created_at": now, "updated_at": now,
        "prompts": [], "params": {}, "finals": [],
    }
    checkpoint.save_manifest(root, manifest)
    return manifest


def delete_project(name: str):
    """删除项目 = 删除整个项目文件夹；被删的是当前链时清掉状态指针。

    返回被删目录的相对路径（output 内）；目录不存在返回 None。
    抛出的 OSError（如文件被占用）由调用方（路由层）捕获反馈。
    """
    name = safe_name(name)
    if not name or name == "h3chain_state.json":
        return None
    root = checkpoint.projects_root()
    target = os.path.join(root, name)
    if not os.path.isdir(target):
        return None
    shutil.rmtree(target)
    state = checkpoint.load_state()
    if isinstance(state, dict) and state.get("dir") == name:
        try:
            os.remove(os.path.join(root, "h3chain_state.json"))
        except OSError:
            pass
    return f"h3_projects/{name}"
