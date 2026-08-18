/**
 * H3 Seamless Chain —— 「长片导演台」（全屏一体化控制台 + 侧栏迷你入口）
 *
 * 参照「H3 一体化总控导演台」状态驱动架构：所有输入（模式/提示词/首帧/参考图）
 * 存入节点「导演台状态」JSON widget，节点 execute 读取 JSON 加载素材，
 * 不在画布上创建/连接任何辅助节点（LoadImage / PrimitiveStringMultiline）。
 *
 * 状态 JSON schema v2（存入「导演台状态」widget）：
 *   { mode:"文生视频"|"首帧视频"|"多参视频",
 *     prompts:["段1文本","段2文本",...],
 *     first_frame:"input目录下的文件名",
 *     ref_assets:[ { file:"文件名", label:"角色1" }, ... ],   // 标签素材池（真源）
 *     ref_images:["文件名1",...],                             // 兼容保留 = ref_assets 的文件序列
 *     segments:[
 *       { scene_prompt:"场景描述", character_prompt:"角色描述",
 *         seconds:6.5,        // 本段时长（秒），null=跟随节点「每段时长」默认
 *         refs:["角色1","场景1"] },  // 本段引用的素材标签，[]=引用全部
 *       ...
 *     ] }
 *
 * 标签引用：提示词写 [[角色1]]，后端按段压实重编号为 <Picture k>；
 * 原生 <Picture N> 写法继续兼容。v1 状态（只有 ref_images）自动迁移默认标签「图片N」。
 *
 * 模式互斥在 JSON 层完成：切模式即清空不相容字段，后端复验不匹配报错。
 * 分段处理中心：segments 与 prompts 等长，每段的 scene_prompt/character_prompt
 * 由后端组合到该段主提示词前（scene → character → 主提示词），不影响核心采样。
 *
 * 配套工作流（web/h3_default_workflow.js）：画布无 H3 节点时可一键载入官方风格
 * 预置工作流；「素材池 · 自动管理」组的 LoadImage 常驻预连，导演台上传素材时
 * 点亮对应节点（mode=0），删除时隐藏（mode=2+折叠）——连线始终存在，只做显隐。
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3SeamlessChainSampler";
const SAVER_TYPE = "H3ChainSaver";
const W_REROLL = "重跑起始段";
const W_SEED = "种子";
const W_INSERTS = "插入视频";
const W_DIR_NAMES = ["存档目录", "断点目录"];
const W_MODE = "生成模式";
const W_DS = "导演台状态";
const W_AR = "宽高比";
const W_MP = "百万像素";
const W_DUR = "每段时长";
const W_WIDTH = "宽度";
const W_HEIGHT = "高度";
const AR_LIST = ["自定义", "21:9", "16:9", "9:16", "4:3", "3:4", "1:1"];
const AR_RATIO = { "21:9": 21 / 9, "16:9": 16 / 9, "9:16": 9 / 16, "4:3": 4 / 3, "3:4": 3 / 4, "1:1": 1 };
const MP_LIST = ["0.25", "0.5", "0.75", "1.0"];
const QUICK_LABELS = ["角色1", "角色2", "场景1", "场景2", "风格", "道具"];
/* 引用语模板库：插入到提示词光标处，[[标签]] 由后端按段压实为 <Picture k> */
const REF_TEMPLATES = [
    ["插入 [[标签]]", (l) => `[[${l}]]`],
    ["主角出场", (l) => `主角：[[${l}]] 全程出镜，保持外观与服饰一致`],
    ["配角出场", (l) => `画面中出现的 [[${l}]] 为次要角色，保持一致`],
    ["场景还原", (l) => `场景以 [[${l}]] 为准，延续其环境与光照`],
    ["风格参考", (l) => `整体画风与色调参考 [[${l}]]`],
    ["镜头参考", (l) => `运镜方式参考 [[${l}]]`],
];
const MODES = [
    ["文生视频", "文生", "纯文本，fl2va UNET，不接图片"],
    ["首帧视频", "首帧", "首帧图片起手，fl2va UNET"],
    ["多参视频", "多参", "参考图片，ref2va UNET，提示词用 <Picture 1..9>"],
];
const MODE_DEFAULT = "文生视频";
const MAX_SEG = 64;
const LS_PROJECTS = "h3d_projects";

let miniBox = null;
let desk = null;
let pendingReset = false;
let refreshTimer = null;
let deleteRouteOk = true;
let ledPhase = "idle";
let ledText = "待命";

/* ---------- 小工具 ---------- */

function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
}

function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

function viewUrl(subfolder, filename) {
    return `/api/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(filename)}`;
}

async function fetchJson(subfolder, filename) {
    try {
        const r = await fetch(viewUrl(subfolder, filename));
        if (!r.ok) return null;
        return JSON.parse(await r.text());
    } catch (e) {
        return null;
    }
}

function fmtTime(v) {
    if (!v) return "";
    if (typeof v === "number") {
        const d = new Date(v * 1000);
        const pad = (x) => String(x).padStart(2, "0");
        return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
    }
    return String(v).replace("T", " ").slice(5, 16);
}

function badge(text, cls) {
    return `<span class="h3d-chip ${cls || ""}">${escapeHtml(text)}</span>`;
}

/* ---------- 官方参数换算（与 nodes.py _resolve_canvas/_snap_seconds 同公式） ---------- */

function resolveCanvas(ar, mp) {
    const r = AR_RATIO[String(ar)];
    if (!r) return null;
    const total = parseFloat(mp) * 1e6;
    if (!isFinite(total) || total <= 0) return null;
    const w0 = Math.sqrt(total * r), h0 = Math.sqrt(total / r);
    const s = Math.min(1, 1344 / Math.max(w0, h0), 768 / Math.min(w0, h0));
    const w = Math.max(32, Math.ceil(w0 * s / 32) * 32);
    const h = Math.max(32, Math.ceil(h0 * s / 32) * 32);
    return [w, h];
}

function snapFrames(seconds) {
    const f = Math.max(5, Math.round(Number(seconds) * 24));
    const k = Math.max(0, Math.round((f - 5) / 17));
    return 17 * k + 5;
}

/** 宽高比+百万像素 -> 显示徽章文案；自定义/非法返回 null */
function canvasBadgeText(node) {
    const ar = String(getWidgetValue(node, W_AR) ?? "");
    if (!AR_RATIO[ar]) return null;
    const mp = String(getWidgetValue(node, W_MP) ?? "0.5");
    const c = resolveCanvas(ar, mp);
    return c ? `${ar} · ${mp}MP → ${c[0]}×${c[1]}` : null;
}

/** 反推：宽高完全命中某 AR×MP 组合则返回 [ar, mp]，否则 null（旧工作流迁移用） */
function matchCanvasCombo(w, h) {
    for (const ar of Object.keys(AR_RATIO)) {
        for (const mp of MP_LIST) {
            const c = resolveCanvas(ar, mp);
            if (c && c[0] === w && c[1] === h) return [ar, mp];
        }
    }
    return null;
}

/* ---------- 素材标签工具 ---------- */

function cleanLabel(text) {
    return String(text ?? "").trim().replace(/[[\]]/g, "").slice(0, 12);
}

function uniqueLabelFrom(taken, base) {
    if (!taken.has(base)) return base;
    let n = 2;
    while (taken.has(`${base}${n}`)) n += 1;
    return `${base}${n}`;
}

/** 在 textarea 光标处插入文本（未聚焦则追加到末尾），返回新值 */
function insertAtCursor(ta, text) {
    if (!ta) return "";
    const s = ta.selectionStart ?? ta.value.length;
    const e = ta.selectionEnd ?? s;
    ta.value = ta.value.slice(0, s) + text + ta.value.slice(e);
    const pos = s + text.length;
    ta.focus();
    ta.setSelectionRange(pos, pos);
    return ta.value;
}

function inputViewUrl(name) {
    const norm = String(name ?? "").replaceAll("\\", "/");
    const parts = norm.split("/");
    const file = parts.pop() ?? "";
    const sub = parts.join("/");
    return `/api/view?type=input&subfolder=${encodeURIComponent(sub)}&filename=${encodeURIComponent(file)}`;
}

async function uploadToInput(file) {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const r = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

/* ---------- 画布节点操作 ---------- */

function findNode() {
    const nodes = app.graph?._nodes || [];
    return nodes.find((n) => n.type === NODE_TYPE) || null;
}

function findSaver() {
    const nodes = app.graph?._nodes || [];
    return nodes.find((n) => n.type === SAVER_TYPE) || null;
}

function dirWidget(node) {
    return (node.widgets || []).find((w) => W_DIR_NAMES.includes(w.name)) || null;
}

function getWidgetValue(node, name) {
    const w = (node.widgets || []).find((w) => w.name === name);
    return w ? w.value : null;
}

function setWidgetValue(node, name, value) {
    const w = (node.widgets || []).find((w) => w.name === name);
    if (!w) return false;
    w.value = value;
    if (typeof w.callback === "function") {
        try { w.callback(value); } catch (e) { /* callback 可选 */ }
    }
    node.setDirtyCanvas(true, true);
    return true;
}

function getDirValue(node) {
    const w = node && dirWidget(node);
    return w ? String(w.value ?? "").trim() : "";
}

function setDirValue(node, value) {
    const w = node && dirWidget(node);
    if (!w) return false;
    w.value = value;
    if (typeof w.callback === "function") {
        try { w.callback(value); } catch (e) { /* callback 可选 */ }
    }
    node.setDirtyCanvas(true, true);
    return true;
}

function hasInserts(node) {
    return !!(node && (node.widgets || []).some((w) => w.name === W_INSERTS));
}

/* ---- 旧工作流迁移：widget 改名/新增后 widgets_values 按位错位，载入前重排 ----
 * 老格式：[宽, 高, 每段帧数, 引导帧数, 种子, (ctrl)?, 步数, CFG, …]
 * 新格式：[宽高比, 百万像素, 宽, 高, 每段时长, 引导帧数, 种子, ctrl, 步数, CFG, …]
 * 识别：老格式 wv[0] 是数字（宽度）；新格式 wv[0] 是宽高比字符串 */
const SEED_CTRL_VALUES = ["fixed", "increment", "decrement", "randomize"];

function remapOldWidgetValues(wv) {
    const [w, h, frames, guide, seed, maybeCtrl, ...rest] = wv;
    const hasCtrl = typeof maybeCtrl === "string" && SEED_CTRL_VALUES.includes(maybeCtrl);
    const ctrl = hasCtrl ? maybeCtrl : "fixed";
    const tail = hasCtrl ? rest : [maybeCtrl, ...rest];
    const combo = matchCanvasCombo(Number(w), Number(h));
    const framesNum = Number(frames);
    const secs = isFinite(framesNum) && framesNum > 0
        ? Math.max(0.5, Math.min(15, Math.round((framesNum / 24) * 10) / 10)) : 5.0;
    return [
        combo ? combo[0] : "自定义",          // 宽高比：能命中 AR×MP 组合则迁移，否则保持自定义画幅
        combo ? combo[1] : "0.5",            // 百万像素
        Number(w), Number(h),                // 宽/高（自定义模式继续生效，存档指纹不变）
        secs, guide, Number(seed), ctrl, ...tail,
    ];
}

function migrateGraphWidgets(graphData) {
    if (!graphData || !Array.isArray(graphData.nodes)) return graphData;
    let migrated = 0;
    for (const n of graphData.nodes) {
        if (n.type === NODE_TYPE && Array.isArray(n.widgets_values)
            && n.widgets_values.length && typeof n.widgets_values[0] !== "string") {
            try {
                n.widgets_values = remapOldWidgetValues(n.widgets_values);
                migrated += 1;
            } catch (e) { /* 解析失败保持原样，交给兜底修正 */ }
        }
    }
    if (migrated) console.log(`[h3-director] 已迁移 ${migrated} 个旧版 H3 节点的参数（宽高/帧数 → 宽高比+百万像素+秒）`);
    return graphData;
}

/** 兜底：宽高比控件值非法（错位载入/手工改坏）时修正，避免后端换算报错 */
function fixInvalidArWidget(node) {
    const w = (node.widgets || []).find((x) => x.name === W_AR);
    if (!w) return;
    const v = String(w.value ?? "");
    if (AR_LIST.includes(v)) return;
    const width = Number(getWidgetValue(node, W_WIDTH));
    const height = Number(getWidgetValue(node, W_HEIGHT));
    const combo = (Number.isFinite(width) && Number.isFinite(height)) ? matchCanvasCombo(width, height) : null;
    const fixed = combo ? combo[0] : "自定义";
    console.warn(`[h3-director] 「宽高比」控件值「${v}」无效，已修正为「${fixed}」（旧工作流请用 Load 按钮载入以完整迁移）`);
    setWidgetValue(node, W_AR, fixed);
    if (combo) setWidgetValue(node, W_MP, combo[1]);
}

/* ---------- 导演台状态（JSON widget 驱动，不操作画布连线） ---------- */

function defaultDs() {
    return { mode: MODE_DEFAULT, prompts: [""], first_frame: "", ref_images: [], ref_assets: [], segments: [] };
}

function defaultSegment() {
    return { scene_prompt: "", character_prompt: "", seconds: null, refs: [] };
}

function getDs(node) {
    if (!node) return defaultDs();
    const w = (node.widgets || []).find((x) => x.name === W_DS);
    if (!w) return defaultDs();
    try {
        const raw = w.value ? JSON.parse(w.value) : {};
        const prompts = Array.isArray(raw.prompts) && raw.prompts.length ? raw.prompts.map(String) : [""];

        /* v2 标签素材池；v1（只有 ref_images）自动迁移默认标签「图片N」 */
        let refAssets = Array.isArray(raw.ref_assets)
            ? raw.ref_assets.filter((a) => a && typeof a === "object" && a.file)
                .map((a) => ({ file: String(a.file), label: cleanLabel(a.label) || "" }))
            : null;
        if (!refAssets) {
            const legacy = Array.isArray(raw.ref_images) ? raw.ref_images.filter(String).map(String) : [];
            refAssets = legacy.map((file, i) => ({ file, label: `图片${i + 1}` }));
        }
        const taken = new Set();
        refAssets.forEach((a) => {
            a.label = cleanLabel(a.label) || `图片${taken.size + 1}`;
            a.label = uniqueLabelFrom(taken, a.label);
            taken.add(a.label);
        });

        let segments = Array.isArray(raw.segments) ? raw.segments : [];
        while (segments.length < prompts.length) segments.push(defaultSegment());
        if (segments.length > prompts.length) segments = segments.slice(0, prompts.length);
        const validLabels = new Set(refAssets.map((a) => a.label));
        segments = segments.map((s) => {
            const sec = Number(s?.seconds);
            return {
                scene_prompt: typeof s?.scene_prompt === "string" ? s.scene_prompt : "",
                character_prompt: typeof s?.character_prompt === "string" ? s.character_prompt : "",
                seconds: (isFinite(sec) && sec > 0) ? Math.min(15, Math.max(0.5, sec)) : null,
                refs: Array.isArray(s?.refs) ? s.refs.map(String).filter((l) => validLabels.has(l)) : [],
            };
        });
        return {
            mode: MODES.some(([m]) => m === raw.mode) ? raw.mode : MODE_DEFAULT,
            prompts,
            first_frame: typeof raw.first_frame === "string" ? raw.first_frame : "",
            ref_images: refAssets.map((a) => a.file),
            ref_assets: refAssets,
            segments,
        };
    } catch (e) {
        return defaultDs();
    }
}

function setDs(node, ds) {
    const w = (node.widgets || []).find((x) => x.name === W_DS);
    if (!w) return false;
    /* ref_images 兼容字段 = ref_assets 文件序列（旧版前端/后端仍可读） */
    if (Array.isArray(ds.ref_assets)) {
        ds.ref_assets = ds.ref_assets.filter((a) => a && a.file);
        ds.ref_images = ds.ref_assets.map((a) => String(a.file));
    }
    w.value = JSON.stringify(ds);
    if (typeof w.callback === "function") {
        try { w.callback(w.value); } catch (e) { /* callback 可选 */ }
    }
    node.setDirtyCanvas(true, true);
    node.graph?.change?.();
    return true;
}

/** 写回模式控件（combo widget 同步，便于画布侧也能看到） */
function syncModeWidget(node, mode) {
    setWidgetValue(node, W_MODE, mode);
}

/** 模式互斥：在 JSON 状态层清空不相容字段 */
function applyModeDs(ds, mode) {
    if (mode === "文生视频") {
        ds.first_frame = "";
        ds.ref_assets = [];
    } else if (mode === "首帧视频") {
        ds.ref_assets = [];
    } else if (mode === "多参视频") {
        ds.first_frame = "";
    }
    ds.mode = mode;
}

function getMode(node) {
    return getDs(node).mode;
}

function setMode(node, mode) {
    const ds = getDs(node);
    applyModeDs(ds, mode);
    setDs(node, ds);
    syncModeWidget(node, mode);
    syncMirrors(node, ds);
    return true;
}

/* ---- 提示词（状态驱动） ---- */

function getPrompts(node) {
    return getDs(node).prompts;
}

function setPromptText(node, idx, text) {
    const ds = getDs(node);
    if (idx < 0 || idx >= ds.prompts.length) return false;
    ds.prompts[idx] = text;
    setDs(node, ds);
    return true;
}

function addPromptSegment(node) {
    const ds = getDs(node);
    if (ds.prompts.length >= MAX_SEG) { alert(`最多 ${MAX_SEG} 段提示词`); return; }
    ds.prompts.push("");
    if (!ds.segments) ds.segments = [];
    ds.segments.push(defaultSegment());
    setDs(node, ds);
    scheduleRefresh(60);
}

function removePromptSegment(node, idx) {
    const ds = getDs(node);
    if (idx < 0 || idx >= ds.prompts.length) return;
    ds.prompts.splice(idx, 1);
    if (ds.segments) ds.segments.splice(idx, 1);
    setDs(node, ds);
    scheduleRefresh(60);
}

function clearPrompts(node) {
    const ds = getDs(node);
    ds.prompts = ds.prompts.map(() => "");
    // 清文本但保留每段时长/素材引用（结构设置跨项目沿用）
    if (ds.segments) ds.segments = ds.segments.map((s) => ({
        ...defaultSegment(), seconds: s?.seconds ?? null, refs: Array.isArray(s?.refs) ? s.refs : [],
    }));
    setDs(node, ds);
}

/* ---- 分段处理中心：场景/角色提示词 + 每段时长 + 段级素材引用（状态驱动） ---- */

function getSegment(node, idx) {
    const ds = getDs(node);
    if (idx < 0 || idx >= (ds.segments || []).length) return defaultSegment();
    return ds.segments[idx];
}

function setSegmentField(node, idx, field, text) {
    const ds = getDs(node);
    if (idx < 0 || idx >= (ds.segments || []).length) return false;
    ds.segments[idx][field] = text;
    setDs(node, ds);
    return true;
}

/** 本段时长（秒）；v=null 表示跟随节点「每段时长」默认 */
function setSegmentSeconds(node, idx, v) {
    const ds = getDs(node);
    if (idx < 0 || idx >= (ds.segments || []).length) return false;
    const num = Number(v);
    ds.segments[idx].seconds = (v === "" || v == null || !isFinite(num) || num <= 0)
        ? null : Math.min(15, Math.max(0.5, num));
    setDs(node, ds);
    return true;
}

/** 勾选/取消本段引用的素材标签；全部取消 = 引用全部（与后端约定一致） */
function toggleSegmentRef(node, idx, label) {
    const ds = getDs(node);
    if (idx < 0 || idx >= (ds.segments || []).length) return;
    const seg = ds.segments[idx];
    if (!Array.isArray(seg.refs)) seg.refs = [];
    const pos = seg.refs.indexOf(label);
    if (pos >= 0) seg.refs.splice(pos, 1);
    else seg.refs.push(label);
    setDs(node, ds);
}

/* ---- 首帧/参考图片（状态驱动 + 配套工作流 LoadImage 镜像点亮/隐藏） ---- */

function setFirstFrame(node, filename) {
    const ds = getDs(node);
    ds.first_frame = filename || "";
    setDs(node, ds);
    syncMirrors(node, ds);
    scheduleRefresh(120);
}

function addRefImage(node, filename) {
    const ds = getDs(node);
    if (ds.ref_assets.length >= 9) { alert("参考图片最多 9 张"); return; }
    const taken = new Set(ds.ref_assets.map((a) => a.label));
    const label = uniqueLabelFrom(taken, `图片${ds.ref_assets.length + 1}`);
    ds.ref_assets.push({ file: filename, label });
    setDs(node, ds);
    syncMirrors(node, ds);
    scheduleRefresh(120);
}

function removeRefImage(node, idx) {
    const ds = getDs(node);
    if (idx < 0 || idx >= ds.ref_assets.length) return;
    const gone = ds.ref_assets.splice(idx, 1)[0];
    if (gone && gone.label) {
        for (const s of ds.segments || []) {
            if (Array.isArray(s.refs)) s.refs = s.refs.filter((l) => l !== gone.label);
        }
    }
    setDs(node, ds);
    syncMirrors(node, ds);
    scheduleRefresh(120);
}

/** 重命名素材标签：唯一性自动加后缀；同步重命名所有段级引用 */
function renameAssetLabel(node, idx, text) {
    const ds = getDs(node);
    const asset = ds.ref_assets[idx];
    if (!asset) return "";
    const clean = cleanLabel(text);
    if (!clean) return asset.label;   // 空输入保持原标签
    const old = asset.label;
    const taken = new Set(ds.ref_assets.filter((_a, i) => i !== idx).map((a) => a.label));
    asset.label = uniqueLabelFrom(taken, clean);
    if (old !== asset.label) {
        for (const s of ds.segments || []) {
            if (Array.isArray(s.refs)) s.refs = s.refs.map((l) => (l === old ? asset.label : l));
        }
    }
    setDs(node, ds);
    return asset.label;
}

function removeFirstFrame(node) {
    const ds = getDs(node);
    ds.first_frame = "";
    setDs(node, ds);
    syncMirrors(node, ds);
    scheduleRefresh(120);
}

/* ---- 配套工作流：素材节点镜像（连线常驻，仅切换 点亮/隐藏） ---- */

function mirrorNodeByTitle(title) {
    return (app.graph?._nodes || []).find((n) => n.getTitle?.() === title || n.title === title) || null;
}

/** 点亮/隐藏配套工作流的素材节点：有图=mode0+展开，无图=mode2+折叠（连线与配置保留） */
function setMirrorImage(title, filename) {
    const m = mirrorNodeByTitle(title);
    if (!m) return false;
    const w = (m.widgets || []).find((x) => x.name === "image") || (m.widgets || [])[0];
    if (w && filename && String(w.value ?? "") !== String(filename)) {
        w.value = filename;
        if (typeof w.callback === "function") { try { w.callback(filename); } catch (e) { /* 可选 */ } }
    }
    const want = filename ? 0 : 2;
    if (m.mode !== want) m.mode = want;
    m.flags = m.flags || {};
    m.flags.collapsed = !filename;
    m.setDirtyCanvas?.(true, true);
    return true;
}

/** 全量同步镜像：首帧图 + 参考图·1..9（仅导演台素材操作时调用，不动手摆工作流） */
function syncMirrors(node, ds) {
    if (!node || !ds) return;
    setMirrorImage("首帧图", ds.mode === "首帧视频" ? (ds.first_frame || "") : "");
    const refs = Array.isArray(ds.ref_assets) ? ds.ref_assets : [];
    for (let i = 0; i < 9; i++) {
        const active = ds.mode === "多参视频" && refs[i] ? String(refs[i].file) : "";
        setMirrorImage(`参考图·${i + 1}`, active);
    }
}

/** 一键载入配套默认工作流（画布无 H3 节点时的引导入口） */
async function loadDefaultWorkflow() {
    if (!window.H3_DEFAULT_WORKFLOW) {
        alert("配套工作流模板未加载：请确认插件 web/ 目录含 h3_default_workflow.js 并刷新页面");
        return;
    }
    if (!confirm("将载入配套工作流并替换当前画布（未保存的画布修改会丢失），继续？")) return;
    try {
        await app.loadGraphData(JSON.parse(JSON.stringify(window.H3_DEFAULT_WORKFLOW)));
    } catch (e) {
        console.warn("[h3-director] loadGraphData 返回值非 Promise（旧版前端），忽略", e);
    }
    await new Promise((r) => setTimeout(r, 150));
    const node = findNode();
    if (node && app.canvas?.centerOnNode) { try { app.canvas.centerOnNode(node); } catch (e) { /* 可选 */ } }
    const ds = getDs(node);
    syncMirrors(node, ds);
    scheduleRefresh(300);
}

async function pickImage(target) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
        const f = input.files && input.files[0];
        if (!f) return;
        try {
            const name = await uploadToInput(f);
            if (target === "first") {
                setFirstFrame(node, name);
            } else {
                addRefImage(node, name);
            }
        } catch (e) {
            alert(`上传失败：${e}`);
        }
    };
    input.click();
}

/* ---- 「插入视频」控件（不变，仍走画布控件） ---- */

function parseInsertsSpec(text) {
    const items = [], bad = [];
    for (const raw of String(text ?? "").split("\n")) {
        const line = raw.trim();
        if (!line || line.startsWith("#")) continue;
        const i = line.indexOf("|");
        const pos = Number(line.slice(0, i).trim());
        const file = line.slice(i + 1).trim();
        if (i < 0 || !Number.isInteger(pos) || pos < 1 || !file) { bad.push(line); continue; }
        items.push([pos, file]);
    }
    items.sort((a, b) => a[0] - b[0]);
    return { items, bad };
}

function insertLines(node) {
    return String(getWidgetValue(node, W_INSERTS) ?? "").split("\n").filter((l) => l.trim());
}

function appendInsert(pos, file) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (!setWidgetValue(node, W_INSERTS, [...insertLines(node), `${pos}|${file}`].join("\n"))) {
        alert("节点上没有「插入视频」控件：当前引擎未合并插入视频能力，或旧工作流需重新添加该节点");
        return;
    }
    scheduleRefresh();
}

function removeInsert(pos, file) {
    const node = findNode();
    if (!node) return;
    setWidgetValue(node, W_INSERTS,
        insertLines(node).filter((l) => l.trim() !== `${pos}|${file}`).join("\n"));
    scheduleRefresh();
}

function pickInsertVideo(pos) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (!hasInserts(node)) {
        alert("当前引擎分支没有「插入视频」控件：合并插入视频技术改进后此功能自动启用");
        return;
    }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "video/*";
    input.onchange = async () => {
        const f = input.files && input.files[0];
        if (!f) return;
        try {
            const name = await uploadToInput(f);
            appendInsert(pos, name);
        } catch (e) {
            alert(`上传失败：${e}`);
        }
    };
    input.click();
}

/* ---- 提示词写回防抖 ---- */

const _taTimers = new Map();

function debouncePromptWrite(node, idx, text) {
    const key = `p${idx}`;
    const old = _taTimers.get(key);
    if (old) clearTimeout(old);
    _taTimers.set(key, setTimeout(() => {
        _taTimers.delete(key);
        setPromptText(node, idx, text);
    }, 350));
}

function debounceSegmentWrite(node, idx, field, text) {
    const key = `s${idx}_${field}`;
    const old = _taTimers.get(key);
    if (old) clearTimeout(old);
    _taTimers.set(key, setTimeout(() => {
        _taTimers.delete(key);
        setSegmentField(node, idx, field, text);
    }, 350));
}

/* ---- 段落计划（状态驱动） ---- */

function planFromDs(node) {
    const ds = getDs(node);
    const prompts = ds.prompts.map((t, i) => ({ text: t.trim(), idx: i }));
    const { items } = parseInsertsSpec(getWidgetValue(node, W_INSERTS));
    const plan = [];
    let pi = 0;
    for (const [pos, file] of items) {
        while (plan.length + 1 < pos) {
            if (pi < prompts.length) plan.push({ kind: "prompt", ...prompts[pi++] });
            else break;
        }
        plan.push({ kind: "insert", pos, file });
    }
    while (pi < prompts.length) plan.push({ kind: "prompt", ...prompts[pi++] });
    const drafts = prompts.filter((p) => !p.text).map((p) => `段 ${p.idx + 1}`);
    return { plan, drafts, ds };
}

function queuePrompt() {
    const node = findNode();
    if (node) {
        const ds = getDs(node);
        setDs(node, ds);
        syncModeWidget(node, ds.mode);
    }
    setLed("running", "已提交队列");
    app.queuePrompt();
    scheduleRefresh();
}

function scheduleRefresh(delay = 900) {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, delay);
}

/* ---------- 项目记忆 ---------- */

function lsProjects() {
    try { return JSON.parse(localStorage.getItem(LS_PROJECTS) || "[]"); } catch (e) { return []; }
}

function rememberProject(dir) {
    if (!dir) return;
    try {
        const arr = lsProjects().filter((d) => d !== dir);
        arr.unshift(dir);
        localStorage.setItem(LS_PROJECTS, JSON.stringify(arr.slice(0, 50)));
    } catch (e) { /* 隐私模式等场景静默降级 */ }
}

/* ---------- 动作 ---------- */

function doReroll(segNo) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const okReroll = setWidgetValue(node, W_REROLL, segNo);
    setWidgetValue(node, W_SEED, Math.floor(Math.random() * 2 ** 48));
    pendingReset = okReroll;
    queuePrompt();
}

function switchProject(dir) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点（只读模式）"); return; }
    if (!setDirValue(node, dir)) { alert("节点上没有「存档目录/断点目录」控件"); return; }
    setWidgetValue(node, W_REROLL, 0);
    rememberProject(dir);
    scheduleRefresh();
}

/* ---------- LED 状态灯 ---------- */

function setLed(phase, text) {
    ledPhase = phase;
    if (text) ledText = text;
    paintLeds();
}

function paintLeds() {
    document.querySelectorAll(".h3d-led").forEach((node) => {
        node.className = `h3d-led ${ledPhase}`;
        const em = node.querySelector("em");
        if (em) em.textContent = ledText;
    });
}

/* ---------- 数据汇总 ---------- */

function saverPrefix() {
    const node = findSaver();
    const w = node && (node.widgets || []).find((w) => w.name === "输出前缀");
    return String((w && w.value) || "h3_chain").trim() || "h3_chain";
}

function paramsSummary(node, mf) {
    const p = (mf && mf.params) || {};
    const gw = (name) => {
        const w = node && (node.widgets || []).find((x) => x.name === name);
        return w ? String(w.value ?? "").trim() : "";
    };
    /* 画幅：优先 宽高比+百万像素 换算，自定义/非法回落 宽×高（或存档指纹） */
    const ar = AR_RATIO[gw(W_AR)] ? gw(W_AR) : "";
    let geo = "";
    if (ar) {
        const c = resolveCanvas(ar, gw(W_MP) || "0.5");
        geo = c ? `${c[0]}×${c[1]}` : "—";
    } else {
        const w = p.width || gw(W_WIDTH);
        const h = p.height || gw(W_HEIGHT);
        geo = w && h ? `${w}×${h}` : "—";
    }
    const durRaw = Number(p.length ? p.length / 24 : gw(W_DUR));
    const len = isFinite(durRaw) && durRaw > 0
        ? `${(p.length ? p.length / 24 : durRaw).toFixed(1)}s/段${p.length ? `(${p.length}f)` : ""}` : "";
    const ctx = p.ctx || gw("引导帧数");
    return {
        geo,
        len,
        ctx: ctx ? `引导${ctx}帧` : "",
    };
}

/** 项目总时长（秒）：生成段按 ds.segments[i].seconds（缺省=节点默认），插入段按默认估 */
function chainSeconds(node, ds, plan) {
    const defRaw = Number(getWidgetValue(node, W_DUR));
    const def = isFinite(defRaw) && defRaw > 0 ? defRaw : 5.0;
    let total = 0;
    for (let i = 0; i < (plan || []).length; i++) {
        const it = plan[i];
        if (it.kind === "prompt" && it.idx !== undefined) {
            const s = ds?.segments?.[it.idx];
            total += s?.seconds ?? def;
        } else {
            total += def;   // 插入视频时长未知，按默认估
        }
    }
    return total;
}

async function collectData() {
    const node = findNode();
    if (node) fixInvalidArWidget(node);
    const state = await fetchJson("checkpoints", "h3chain_state.json");
    const idx = await fetchJson("checkpoints", "h3chain_index.json");
    let mf = null;
    if (state && state.dir) mf = await fetchJson(`checkpoints/${state.dir}`, "manifest.json");
    const prefix = saverPrefix();
    const history = await fetchJson(prefix, "h3saver_history.json");

    let plan = null;
    let drafts = [];
    let ds = defaultDs();
    if (node) {
        ({ plan, drafts, ds } = planFromDs(node));
    }
    if (!plan || !plan.length) {
        const total = (state && state.total) ?? (mf && mf.total) ?? 0;
        if (total) {
            plan = Array.from({ length: total }, (_v, i) => {
                const ins = ((mf && mf.inserts) || []).find((x) => x.slot === i);
                return ins ? { kind: "insert", pos: i + 1, file: ins.file || "" }
                           : { kind: "prompt", text: ((mf && mf.prompts) || [])[i] || "" };
            });
        } else {
            plan = [];
        }
    }
    return { node, state, idx, mf, plan, drafts, history, prefix, ds };
}

function statusLine(state, mf, plan) {
    const total = plan ? plan.length : (state?.total ?? mf?.total ?? 0);
    const done = state?.done ?? mf?.done ?? 0;
    if (!total) return { text: "尚未配置段落：在流水线卡片填写提示词，或点「＋ 添加一段」", next: false };
    if (done >= total) return { text: `本链已全部完成（${total}/${total} 段）✓ 可「＋ 新建项目」开下一条`, next: false };
    const anchored = done > 0 ? `，锚定段 ${done} 尾部` : "";
    return { text: `已载入段 1–${done}（存档回放）→ 本次将生成段 ${done + 1}${anchored}`, next: true };
}

function segMediaHtml(state, mf, idx) {
    const done = mf?.done ?? 0;
    if (idx >= done || !state?.dir) return null;
    const thumbFile = (mf.thumbs || [])[idx];
    const videoFile = (mf.videos || [])[idx];
    const thumbSrc = thumbFile ? viewUrl(state.dir, thumbFile) : "";
    if (videoFile) {
        return `<video class="h3d-segvideo" controls preload="metadata"${thumbSrc ? ` poster="${thumbSrc}"` : ""} src="${viewUrl(state.dir, videoFile)}"></video>`;
    }
    if (thumbSrc) return `<img loading="lazy" src="${thumbSrc}" alt="段${idx + 1}">`;
    return null;
}

/* ---------- 样式（参照一体化总控导演台，前缀 h3d-） ---------- */

const STYLE_ID = "h3-director-style";

function injectStyles() {
    if (document.getElementById(STYLE_ID)) return;
    const style = el("style");
    style.id = STYLE_ID;
    style.textContent = `
    :root{--h3d-ink:#0b0d17;--h3d-panel:#141726;--h3d-panel2:#1a1e32;--h3d-line:#2a2f4a;--h3d-cyan:#a78bfa;--h3d-copper:#e6b566;--h3d-bone:#eef0fa;--h3d-muted:#9aa0be;--h3d-ok:#5fd4a2;--h3d-warn:#f0a35e;--h3d-danger:#f27d7d}
    .h3d-page *,.h3d-mini *,.h3d-dialog *{box-sizing:border-box}
    @keyframes h3d-blink{50%{opacity:.35}}
    .h3d-led{display:inline-flex;gap:7px;align-items:center;color:var(--h3d-muted);font-size:12px;white-space:nowrap}
    .h3d-led i{width:8px;height:8px;border-radius:50%;background:#6a6f8f;flex:none}
    .h3d-led.running i{background:var(--h3d-cyan);box-shadow:0 0 10px #a78bfacc;animation:h3d-blink 1s infinite}
    .h3d-led.done i{background:var(--h3d-ok);box-shadow:0 0 12px #5fd4a288}
    .h3d-led.error i{background:var(--h3d-warn);box-shadow:0 0 12px #f0a35e88}
    .h3d-btn{cursor:pointer;border:1px solid #343a5e;border-radius:6px;background:#171b2e;color:var(--h3d-bone);padding:6px 11px;font-size:12px;font-family:inherit;transition:border-color .12s,filter .12s}
    .h3d-btn:hover{filter:brightness(1.18)}
    .h3d-btn:disabled{opacity:.5;cursor:not-allowed;filter:none}
    .h3d-btn-cyan{border-color:#565092;background:#242045;color:#bfaeff}
    .h3d-btn-danger{border-color:#6e3a4a;background:#2e1826;color:#ff9a9a}
    .h3d-btn-cta{border:0;background:linear-gradient(135deg,#f0c274,#e6b566 55%,#d99e4a);color:#1a1408;font-weight:700;box-shadow:0 2px 14px #e6b56633}
    .h3d-btn-cta:hover{filter:brightness(1.08);box-shadow:0 3px 18px #e6b56644}
    .h3d-chip{font-size:10.5px;padding:2px 8px;border-radius:10px;border:1px solid #3a4066;background:#1d2138;color:#b6bcdc;white-space:nowrap}
    .h3d-chip.ok{border-color:#2f6e57;background:#17342a;color:#7fe0b0}
    .h3d-chip.media{border-color:#7a5f36;background:#352a19;color:#e9c07a}
    .h3d-chip.cyan{border-color:#565092;background:#242045;color:#bfaeff}
    .h3d-chip.warn{border-color:#6e3a4a;background:#45202c;color:#ff8585}

    /* ---- 侧栏迷你入口卡 ---- */
    .h3d-mini{display:flex;flex-direction:column;gap:9px;padding:10px;font-size:12px;color:var(--h3d-bone);background:linear-gradient(150deg,#131626,#1c2038);border:1px solid #31375a;border-radius:10px}
    .h3d-mini-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
    .h3d-mini-brand{font-weight:700;letter-spacing:.04em;min-width:0}
    .h3d-mini-brand small{display:block;margin-top:3px;color:var(--h3d-cyan);font-weight:500;font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-rail{display:flex;gap:3px}
    .h3d-mini-rail span{flex:1;height:4px;border-radius:4px;background:#33385a}
    .h3d-mini-rail span.done{background:var(--h3d-cyan)}
    .h3d-mini-rail span.next{background:#4d4680;animation:h3d-blink 1.2s infinite}
    .h3d-mini-cards{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}
    .h3d-mini-card{min-height:62px;padding:9px;border:1px solid #2f3352;border-radius:8px;background:#10131f;font-size:11px;color:var(--h3d-muted);line-height:1.65;overflow:hidden}
    .h3d-mini-count{display:grid;place-items:center;padding:9px 10px;min-width:66px;border:1px solid #2f3352;border-radius:8px;background:#10131f;font:700 20px/1.15 ui-monospace,Consolas;color:var(--h3d-cyan);text-align:center}
    .h3d-mini-count small{font-size:10px;color:var(--h3d-muted);font-weight:400}
    .h3d-mini-foot{display:flex;flex-direction:column;gap:8px}
    .h3d-mini-params{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-open{width:100%;padding:9px}

    /* ---- 全屏导演台 ---- */
    .h3d-page{position:fixed;inset:0;z-index:1000000;background:var(--h3d-ink);color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif;display:grid;grid-template-rows:58px 1fr 58px}
    .h3d-topbar{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--h3d-line);background:linear-gradient(90deg,#10131f,#171b30)}
    .h3d-top-left{min-width:0;display:flex;gap:14px;align-items:center}
    .h3d-kicker{color:var(--h3d-copper);font:700 11px/1 ui-monospace,Consolas;letter-spacing:.18em;white-space:nowrap}
    .h3d-title{font-size:17px;font-weight:700;white-space:nowrap}
    .h3d-sub{min-width:0;color:var(--h3d-muted);font-family:ui-monospace,Consolas;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-top-right{display:flex;gap:14px;align-items:center;justify-self:end}
    .h3d-close{width:36px;height:36px;border:1px solid var(--h3d-line);border-radius:7px;background:#181c30;color:var(--h3d-bone);cursor:pointer;font-size:15px}
    .h3d-close:hover{filter:brightness(1.2)}
    .h3d-stage{min-height:0;display:grid;grid-template-columns:minmax(255px,300px) minmax(400px,1fr) minmax(255px,300px);gap:1px;background:var(--h3d-line)}
    .h3d-col{min-width:0;min-height:0;background:var(--h3d-panel);overflow:auto}
    .h3d-sechead{position:sticky;top:0;z-index:3;padding:14px 16px 10px;background:#141726ed;backdrop-filter:blur(8px);border-bottom:1px solid #272c47}
    .h3d-sechead strong{display:block}
    .h3d-sechead small{color:var(--h3d-muted)}

    .h3d-projlist{display:grid;gap:7px;padding:12px}
    .h3d-proj{display:grid;gap:3px;padding:9px 10px 9px 13px;border:1px solid #2f3352;border-radius:8px;background:#10131f;cursor:pointer;box-shadow:inset 3px 0 0 #33385a;text-align:left;font-family:inherit;color:inherit}
    .h3d-proj:hover{background:#151929}
    .h3d-proj.active{box-shadow:inset 3px 0 0 var(--h3d-cyan);border-color:#565092}
    .h3d-proj-name{font-weight:700;word-break:break-all;font-size:12.5px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;color:var(--h3d-bone)}
    .h3d-proj-meta{color:var(--h3d-muted);font-size:11px;font-family:ui-monospace,Consolas}
    .h3d-newrow{display:flex;gap:8px;padding:0 12px 12px;flex-wrap:wrap}
    .h3d-meter{margin:0 12px 12px;padding:11px;border:1px solid #343a5e;border-radius:8px;background:#10131f}
    .h3d-meter strong{display:block;color:var(--h3d-cyan);font:700 16px/1.3 ui-monospace,Consolas}
    .h3d-meter p{margin:4px 0 0;color:var(--h3d-muted);font-size:11px}
    .h3d-drafts{margin:0 12px 12px;color:var(--h3d-muted);font-size:11px;line-height:1.7}

    .h3d-center-pad{padding:14px 18px 20px}
    .h3d-statusbar{display:flex;align-items:center;gap:10px;padding:10px 13px;border:1px solid #565092;border-radius:8px;background:linear-gradient(90deg,#242045,#1a1e32);line-height:1.6}
    .h3d-statusbar .h3d-st-text{flex:1;min-width:0}
    .h3d-statusbar .h3d-chip{align-self:center}
    .h3d-refresh{padding:4px 10px;flex:none}
    .h3d-rail{display:flex;gap:4px;margin:12px 0 14px}
    .h3d-rail span{flex:1;height:5px;border-radius:4px;background:#33385a}
    .h3d-rail span.done{background:var(--h3d-cyan)}
    .h3d-rail span.next{background:#4d4680;animation:h3d-blink 1.2s infinite}
    .h3d-cards{display:grid;gap:10px}
    .h3d-card{display:grid;grid-template-columns:150px minmax(0,1fr);gap:11px;padding:10px;border:1px solid #2f3352;border-radius:9px;background:#10131f}
    .h3d-card.todo{opacity:.72}
    .h3d-card.todo:hover{opacity:1}
    .h3d-thumb{width:150px;aspect-ratio:16/9;border-radius:6px;overflow:hidden;background:#080a12;display:grid;place-items:center;color:#565b78;font:700 11px ui-monospace,Consolas}
    .h3d-thumb video,.h3d-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-thumb video.h3d-segvideo{object-fit:contain;background:#000}
    .h3d-cbody{min-width:0;display:flex;flex-direction:column;gap:5px}
    .h3d-ctitle{display:flex;gap:7px;align-items:center;flex-wrap:wrap;font-weight:700}
    .h3d-cmeta{color:var(--h3d-muted);font:11px ui-monospace,Consolas;word-break:break-all}
    .h3d-cprompt{color:#c6cbe4;font-size:12px;line-height:1.65;word-break:break-word}
    .h3d-ta{width:100%;min-height:84px;resize:vertical;border:1px solid #333a5c;border-radius:6px;background:#131626;color:var(--h3d-bone);padding:8px 9px;font:12.5px/1.65 "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none;box-sizing:border-box}
    .h3d-ta:focus{border-color:#b3a1ff;box-shadow:0 0 0 2px #a78bfa33}
    .h3d-ta:disabled{color:#565b78;cursor:not-allowed;background:#0d0f1a}
    .h3d-ta-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:4px}
    .h3d-ta-hint{color:var(--h3d-muted);font-size:10.5px}
    .h3d-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:2px}
    .h3d-hint{color:var(--h3d-muted);font-size:11px;align-self:center}
    .h3d-editor{display:grid;gap:7px;margin-top:4px;padding:10px;border:1px solid #3a4066;border-radius:8px;background:#0f1220}
    .h3d-editor textarea{width:100%;min-height:96px;resize:vertical;border:1px solid #333a5c;border-radius:6px;background:#131626;color:var(--h3d-bone);padding:9px;font:13px/1.7 "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none}
    .h3d-editor textarea:focus{border-color:#b3a1ff;box-shadow:0 0 0 2px #a78bfa33}
    .h3d-editor-row{display:flex;gap:7px;flex-wrap:wrap}
    .h3d-addrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .h3d-drawer{margin-top:14px;border:1px solid #303656;border-radius:8px;background:#10131f}
    .h3d-drawer summary{padding:10px 12px;cursor:pointer;color:var(--h3d-cyan)}
    .h3d-drawer pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;font-size:11.5px;max-height:300px;overflow:auto;color:#bcc3de}

    .h3d-hist{padding:12px;display:grid;gap:10px;align-content:start}
    .h3d-empty{padding:16px 10px;border:1px dashed #3a4066;border-radius:8px;color:var(--h3d-muted);text-align:center;background:#10131f;line-height:1.8}
    .h3d-result{padding:8px;border:1px solid #363c60;border-radius:9px;background:#0f1220}
    .h3d-result.current{border-color:#565092}
    .h3d-result video{display:block;width:100%;max-height:230px;border-radius:6px;background:#05060c}
    .h3d-result-meta{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:7px}
    .h3d-result-name{min-width:0;color:#bcc3de;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-result-info{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-result-acts{display:flex;gap:6px;flex:none}
    .h3d-dl{padding:4px 8px;border:1px solid #565092;border-radius:6px;color:#bfaeff;text-decoration:none;background:#242045;font-size:11px}
    .h3d-dl:hover{filter:brightness(1.15)}
    .h3d-foot{padding:2px 16px 14px;color:var(--h3d-muted);font-size:10.5px;word-break:break-all;line-height:1.8}

    /* ---- 分段处理中心面板 ---- */
    .h3d-seg-panel{margin-top:6px;border:1px solid #303656;border-radius:7px;background:#0e1122;overflow:hidden}
    .h3d-seg-panel summary{padding:7px 10px;cursor:pointer;color:var(--h3d-cyan);font-size:11.5px;font-weight:600;user-select:none;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
    .h3d-seg-panel summary::-webkit-details-marker{display:none}
    .h3d-seg-panel summary::before{content:"▸";font-size:10px;transition:transform .15s}
    .h3d-seg-panel[open] summary::before{transform:rotate(90deg)}
    .h3d-seg-panel.has-content{border-color:#565092}
    .h3d-seg-body{display:grid;gap:7px;padding:0 10px 10px}
    .h3d-seg-label{display:block;margin-bottom:2px;color:var(--h3d-muted);font-size:10.5px;font-weight:600}
    .h3d-seg-ta{min-height:48px !important;font-size:12px !important;line-height:1.55 !important}

    /* ---- 每段时长 + 段级引用素材 ---- */
    .h3d-secs{width:58px;border:1px solid #333a5c;border-radius:5px;background:#131626;color:var(--h3d-bone);padding:2px 4px;font:11px ui-monospace,Consolas;text-align:right;outline:none}
    .h3d-secs:focus{border-color:#b3a1ff}
    .h3d-secs-hint{color:var(--h3d-muted);font:10px ui-monospace,Consolas;white-space:nowrap}
    .h3d-refrow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:4px;padding:7px 8px;border:1px solid #303656;border-radius:7px;background:#0e1122}
    .h3d-refrow>label{color:var(--h3d-muted);font-size:10.5px;font-weight:600;flex:none}
    .h3d-refchip{display:inline-flex;gap:5px;align-items:center;padding:3px 8px 3px 3px;border:1px solid #343a5e;border-radius:12px;background:#171b2e;color:#9aa3c4;cursor:pointer;font-size:11px;font-family:inherit}
    .h3d-refchip:hover{border-color:#565092}
    .h3d-refchip.on{border-color:#2f6e57;background:#12291f;color:#7fe0b0}
    .h3d-refchip img{width:20px;height:20px;border-radius:9px;object-fit:cover;background:#080a12}
    .h3d-reftpl{max-width:118px;border:1px solid #333a5c;border-radius:6px;background:#131626;color:var(--h3d-bone);padding:3px 4px;font-size:11px;outline:none;margin-left:auto}
    .h3d-reftpl:focus{border-color:#b3a1ff}

    /* ---- 素材标签编辑 ---- */
    .h3d-labelinp{width:100%;border:1px solid #333a5c;border-radius:5px;background:#131626;color:var(--h3d-bone);padding:4px 6px;font:600 12px "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none}
    .h3d-labelinp:focus{border-color:#b3a1ff;box-shadow:0 0 0 2px #a78bfa33}
    .h3d-quicklbl{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px}
    .h3d-quicklbl button{padding:2px 7px;border:1px solid #343a5e;border-radius:10px;background:#171b2e;color:#9aa0be;cursor:pointer;font-size:10px;font-family:inherit}
    .h3d-quicklbl button:hover{border-color:#565092;color:#d6dcf5}
    .h3d-asset-usage{margin-top:5px;display:flex;gap:4px;flex-wrap:wrap}

    /* ---- 链参数：换算徽章 + 高级设置折叠 ---- */
    .h3d-convbadge{grid-column:1/-1;margin:-4px 0 0;padding:8px 10px;border:1px dashed #565092;border-radius:7px;background:#1e1c38;color:#bfaeff;font:11.5px ui-monospace,Consolas;word-break:break-all}
    .h3d-adv{grid-column:1/-1;border:1px solid #303656;border-radius:8px;background:#0e1120;overflow:hidden}
    .h3d-adv summary{padding:8px 10px;cursor:pointer;color:var(--h3d-muted);font-size:11.5px;font-weight:600;user-select:none}
    .h3d-adv summary:hover{color:#d6dcf5}
    .h3d-adv summary::-webkit-details-marker{display:none}
    .h3d-adv summary::before{content:"⚙ ";}
    .h3d-adv[open] summary{border-bottom:1px solid #272c47;color:var(--h3d-cyan)}
    .h3d-adv-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:10px}
    .h3d-adv .h3d-param{margin:0}
    .h3d-param{margin:0}
    .h3d-param .h3d-hint{display:block;margin-top:4px}

    .h3d-loadwf{display:flex;flex-direction:column;gap:10px;align-items:center;padding:22px 14px;border:1px dashed #565092;border-radius:10px;background:#1e1c38;text-align:center;line-height:1.9}
    .h3d-loadwf p{margin:0;color:var(--h3d-muted);font-size:11.5px}

    .h3d-footer{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 20px;border-top:1px solid var(--h3d-line);background:#10131f}
    .h3d-footinfo{display:flex;gap:16px;color:var(--h3d-muted);flex-wrap:wrap;min-width:0;font-size:11.5px;align-items:center}
    .h3d-footinfo b{color:var(--h3d-bone)}
    .h3d-run{min-width:150px;padding:11px 18px}

    /* ---- 素材与参考 / 链参数 ---- */
    .h3d-lsec,.h3d-rsec{display:flex;flex-direction:column;min-height:0}
    .h3d-col.left .h3d-sechead,.h3d-col.right .h3d-sechead{position:static;backdrop-filter:none}
    .h3d-modebar{display:flex;gap:6px;padding:10px 12px 6px;border-bottom:1px solid var(--h3d-line);background:#0d0f1a}
    .h3d-mode{flex:1;min-width:0;padding:8px 4px;border:1px solid #2f3352;border-radius:7px;background:#10131f;color:#9aa3c4;cursor:pointer;font:700 12px "Microsoft YaHei UI","Segoe UI",sans-serif;transition:all .12s}
    .h3d-mode:hover{border-color:#565092;color:#d6dcf5}
    .h3d-mode.active{color:#16102e;background:var(--h3d-cyan);border-color:var(--h3d-cyan);box-shadow:0 0 0 1px var(--h3d-cyan) inset,0 0 14px #a78bfa44}
    .h3d-mode small{display:block;font-weight:400;font-size:9.5px;opacity:.72;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-mode.active small{opacity:.85}
    .h3d-assets{display:grid;gap:8px;padding:12px}
    .h3d-asset{display:grid;grid-template-columns:64px minmax(0,1fr) auto;gap:9px;align-items:center;padding:8px;border:1px solid #2f3352;border-radius:8px;background:#10131f;box-shadow:inset 3px 0 0 #33385a}
    .h3d-asset.on{box-shadow:inset 3px 0 0 var(--h3d-cyan)}
    .h3d-asset-thumb{width:64px;height:52px;border-radius:5px;background:#1e2238;display:grid;place-items:center;overflow:hidden;color:#6a6f8f;font:700 10px ui-monospace,Consolas}
    .h3d-asset-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-asset-copy{min-width:0}
    .h3d-asset-copy strong{display:block;font-size:12px}
    .h3d-asset-copy small{display:block;margin-top:3px;color:var(--h3d-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-asset-acts{display:flex;gap:5px}
    .h3d-asset-acts .h3d-btn{padding:4px 8px;font-size:11px}
    .h3d-addasset{justify-self:start;padding:6px 12px}
    .h3d-params{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px}
    .h3d-param label{display:block;margin-bottom:4px;color:var(--h3d-muted);font-size:11px}
    .h3d-select,.h3d-seedrow input{width:100%;border:1px solid #333a5c;border-radius:6px;background:#131626;color:var(--h3d-bone);padding:6px 7px;font-size:12px;outline:none;font-family:inherit}
    .h3d-select:focus,.h3d-seedrow input:focus{border-color:#b3a1ff}
    .h3d-seedrow{display:flex;gap:5px}
    .h3d-seedrow input{flex:1;min-width:0}
    .h3d-seedrow .h3d-btn{padding:4px 8px;flex:none}
    .h3d-psec .h3d-foot,.h3d-asec .h3d-foot,.h3d-projsec .h3d-foot{padding:0 16px 14px}

    /* ---- 新建项目模态 ---- */
    .h3d-overlay{position:fixed;z-index:1000003;inset:0;display:grid;place-items:center;padding:24px;background:#07060fd0;backdrop-filter:blur(10px)}
    .h3d-dialog{width:min(470px,calc(100vw - 40px));border:1px solid #3f4570;border-radius:14px;background:linear-gradient(160deg,#1c2038,#12141f 62%);box-shadow:0 22px 64px #000b;padding:20px;color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif}
    .h3d-dialog h3{margin:0 0 6px;font-size:16px}
    .h3d-dialog .h3d-lead{color:var(--h3d-muted);margin:0 0 14px;line-height:1.75;font-size:12px}
    .h3d-dialog input[type=text]{width:100%;border:1px solid #333a5c;border-radius:6px;background:#131626;color:var(--h3d-bone);padding:9px 10px;font:13px ui-monospace,Consolas;outline:none;margin-bottom:6px}
    .h3d-dialog input[type=text]:focus{border-color:#b3a1ff}
    .h3d-err{color:#ff8585;font-size:11px;min-height:16px;margin-bottom:6px}
    .h3d-check{display:flex;gap:8px;align-items:center;color:var(--h3d-muted);font-size:12px;margin-bottom:14px;cursor:pointer}
    .h3d-dialog-row{display:flex;gap:8px;justify-content:flex-end}

    .h3d-fab{position:fixed;right:16px;top:120px;z-index:80;width:44px;height:44px;border-radius:50%;border:1px solid #565092;background:#242045;color:#bfaeff;cursor:pointer;font-size:17px}
    .h3d-fab:hover{filter:brightness(1.2)}

    @media(max-width:1200px){.h3d-sub{display:none}}
    @media(max-width:1050px){.h3d-stage{grid-template-columns:240px minmax(0,1fr)}.h3d-col.right{grid-column:1/-1;max-height:260px}}
    @media(max-width:760px){.h3d-page{grid-template-rows:auto 1fr 64px}.h3d-topbar{grid-template-columns:1fr auto;padding:8px 12px;flex-wrap:wrap;gap:8px}.h3d-title{font-size:15px}.h3d-footer{flex-direction:column;padding:8px 12px;gap:8px}}
    `;
    document.head.appendChild(style);
}

/* ---------- 侧栏迷你卡 ---------- */

function renderMini(data) {
    if (!miniBox) return;
    const { state, mf, plan, node } = data;
    miniBox.innerHTML = "";
    const card = el("div", "h3d-mini");

    const total = plan ? plan.length : 0;
    const done = state?.done ?? mf?.done ?? 0;

    const head = el("div", "h3d-mini-head");
    const brand = el("div", "h3d-mini-brand", "长片导演台");
    brand.append(el("small", "", escapeHtml(state?.dir || (node ? getDirValue(node) || "未命名链" : "无节点"))));
    head.append(brand, el("span", `h3d-led ${ledPhase}`, '<i></i><em></em>'));

    const rail = el("div", "h3d-mini-rail");
    const cells = Math.max(1, Math.min(total, MAX_SEG));
    for (let i = 0; i < cells; i++) {
        rail.append(el("span", i < done ? "done" : i === done && done < total ? "next" : ""));
    }

    const cards = el("div", "h3d-mini-cards");
    const info = el("div", "h3d-mini-card", escapeHtml(statusLine(state, mf, plan).text));
    const count = el("div", "h3d-mini-count", `${done}<small>/${total || "?"} 段</small>`);
    cards.append(info, count);

    const foot = el("div", "h3d-mini-foot");
    const ps = paramsSummary(node, mf);
    const totalSec = plan?.length ? `共${chainSeconds(node, data.ds, plan).toFixed(1)}s/${plan.length}段` : "";
    foot.append(el("div", "h3d-mini-params",
        escapeHtml([ps.geo, ps.len, ps.ctx, totalSec].filter(Boolean).join(" · ") || "参数待首次运行后显示")));
    const open = el("button", "h3d-btn h3d-btn-cta h3d-mini-open", "打开长片导演台");
    open.onclick = openDesk;
    foot.append(open);

    card.append(head, rail, cards, foot);
    miniBox.append(card);
    paintLeds();
}

/* ---------- 全屏导演台 ---------- */

function openDesk() {
    if (desk) return;
    const page = el("section", "h3d-page");
    page.tabIndex = -1;

    /* 顶栏 */
    const topbar = el("header", "h3d-topbar");
    const left = el("div", "h3d-top-left");
    left.append(
        el("span", "h3d-kicker", "H3 · SEAMLESS CHAIN"),
        el("span", "h3d-title", "长片一体化导演台"),
        el("span", "h3d-sub", "项目控制台 · 存档续拍 · 逐段审片"),
    );
    const right = el("div", "h3d-top-right");
    const ledWrap = el("span", `h3d-led ${ledPhase}`, '<i></i><em></em>');
    const sub = el("span", "h3d-sub h3d-top-project", "");
    const close = el("button", "h3d-close", "✕");
    close.title = "关闭导演台（Esc）";
    right.append(ledWrap, sub, close);
    topbar.append(left, right);

    /* 主舞台三栏（侧栏各含两个固定分区，渲染器只填充分区内容） */
    const stage = el("div", "h3d-stage");
    const colL = el("aside", "h3d-col left");
    const lProj = el("section", "h3d-lsec h3d-projsec");
    const lAssets = el("section", "h3d-lsec h3d-asec");
    colL.append(lProj, lAssets);
    const colC = el("section", "h3d-col center");
    colC.append(el("div", "h3d-sechead", "<strong>段落流水线</strong><small>卡片 = 生成顺序；✏ 改词 · 🎲 重摇 · 📎 插视频 · 🎬 分段处理（场景/角色）</small>"));
    const colR = el("aside", "h3d-col right");
    const rParams = el("section", "h3d-rsec h3d-psec");
    const rHist = el("section", "h3d-rsec h3d-hsec");
    colR.append(rParams, rHist);
    stage.append(colL, colC, colR);

    /* 页脚 */
    const footer = el("footer", "h3d-footer");
    const footInfo = el("div", "h3d-footinfo", "");
    const run = el("button", "h3d-btn h3d-btn-cta h3d-run", "▶ 开始生成");
    footer.append(footInfo, run);

    page.append(topbar, stage, footer);
    document.body.append(page);
    page.focus();

    desk = {
        page,
        zones: {
            project: sub,
            colC,
            lProj, lAssets,
            rParams, rHist,
            footInfo, run,
        },
        cardsSig: "",
        histSig: "",
        close() {
            page.remove();
            desk = null;
        },
    };
    close.onclick = () => desk && desk.close();
    page.addEventListener("keydown", (e) => {
        if (e.key === "Escape") desk && desk.close();
    });
    refresh();
}

function isVideoPlaying(root) {
    return [...root.querySelectorAll("video")].some((v) => !v.paused && !v.ended);
}

function cardsSignature(data) {
    const { state, plan, ds } = data;
    return JSON.stringify({
        dir: state?.dir ?? "",
        done: state?.done ?? 0,
        total: plan?.length ?? 0,
        review: !!state?.review,
        reroll: state?.reroll ?? 0,
        plan: (plan || []).map((it) => it.kind === "insert" ? ["i", it.pos, it.file] : ["p", it.text]),
        segs: (ds?.segments || []).map((s) => [s.scene_prompt ?? "", s.character_prompt ?? "",
                                               s.seconds ?? 0, (s.refs || []).join(",")]),
        labels: (ds?.ref_assets || []).map((a) => a.label),
        mode: ds?.mode ?? "",
        dur: String(getWidgetValue(data.node, W_DUR) ?? ""),
    });
}

function updateDesk(data) {
    if (!desk) return;
    const { node, state, idx, mf, plan, drafts, history, prefix } = data;
    const z = desk.zones;

    /* 顶栏项目名 + LED */
    const dirName = state?.dir || (node ? getDirValue(node) : "") || "无当前链";
    z.project.textContent = `项目 · ${dirName}`;
    paintLeds();

    /* 左栏：项目与链 + 素材与参考（标签输入聚焦时跳过重渲，防丢焦） */
    renderLeftColumn(z.lProj, data);
    if (!z.lAssets.querySelector(".h3d-labelinp:focus")) {
        renderAssetsZone(z.lAssets, data);
    }

    /* 中栏：状态条 + 进度轨 + 段落卡片（有编辑器/播放中时跳过重渲） */
    renderCenterColumn(z.colC, data);

    /* 右栏：链参数（编辑中不重建）+ 成片历史（播放中不重建） */
    const psig = paramsSig(node);
    const pgrid = z.rParams.querySelector(".h3d-params");
    if (!(pgrid && pgrid.contains(document.activeElement)) && z.rParams.dataset.sig !== psig) {
        z.rParams.dataset.sig = psig;
        renderParamsZone(z.rParams, data);
    }
    const histSig = prefix + "|" + (history ? history.length + ":" + (history[0]?.file ?? "") : "-");
    if (histSig !== desk.histSig || !z.rHist.querySelector(".h3d-hist")) {
        if (!isVideoPlaying(z.rHist)) {
            desk.histSig = histSig;
            renderHistoryZone(z.rHist, data);
        }
    }

    /* 页脚 CTA + 提示 */
    renderFooter(z, data);
}

/* ---- 左栏 ---- */

function renderLeftColumn(sec, data) {
    const { node, state, idx } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>项目与链</strong><small>切换项目 = 换存档目录，提示词自动沿用</small>"));

    const list = el("div", "h3d-projlist");
    const projects = mergeProjects(idx, state);
    if (!projects.length) {
        list.append(el("div", "h3d-empty",
            "暂无历史项目：跑一次（开审片/存档）后会自动登记；也可点下方「＋ 新建项目」直接开跑。"));
    }
    for (const p of projects) {
        const active = state?.dir && p.dir === state.dir;
        const btn = el("button", "h3d-proj" + (active ? " active" : ""));
        const nodeDir = node ? getDirValue(node) : "";
        btn.innerHTML = `
            <span class="h3d-proj-name">${escapeHtml(p.dir)}
                ${p.dir === nodeDir && nodeDir ? badge("已指向", "cyan") : ""}
                ${p.draft ? badge("草稿", "") : ""}
            </span>
            <span class="h3d-proj-meta">${p.done ?? "?"}/${p.total ?? "?"} 段${p.updated_at ? " · " + escapeHtml(fmtTime(p.updated_at)) : ""}</span>`;
        btn.title = "点击把「存档目录」切到该项目（共享参数需与其一致，否则后端会提示换链）";
        btn.onclick = () => switchProject(p.dir);
        list.append(btn);
    }
    sec.append(list);

    const newrow = el("div", "h3d-newrow");
    const newBtn = el("button", "h3d-btn h3d-btn-cyan", "＋ 新建项目");
    newBtn.title = "新开一条视频链：换存档目录名，提示词沿用当前内容作底稿";
    newBtn.onclick = openNewProjectModal;
    newrow.append(newBtn);
    sec.append(newrow);

    if (state?.dir) {
        const done = state.done ?? 0;
        const total = state.total ?? 0;
        const ps = paramsSummary(node, data.mf);
        const totalSec = data.plan?.length
            ? `共${chainSeconds(node, data.ds, data.plan).toFixed(1)}s` : "";
        const meter = el("div", "h3d-meter");
        meter.innerHTML = `<strong>${done}/${total || "?"} 段${state.review ? " · 审片中" : ""}</strong>
            <p>${escapeHtml([ps.geo, ps.len, ps.ctx, totalSec].filter(Boolean).join(" · ") || "参数待首次运行后显示")}</p>`;
        sec.append(meter);
        sec.append(el("div", "h3d-foot", `链目录：output/checkpoints/${escapeHtml(state.dir)}`));
    }
}

function mergeProjects(idx, state) {
    const map = new Map();
    for (const p of (idx?.projects) || []) {
        if (p?.dir) map.set(p.dir, { dir: p.dir, done: p.done, total: p.total, updated_at: p.updated_at, draft: false });
    }
    if (state?.dir && !map.has(state.dir)) {
        map.set(state.dir, { dir: state.dir, done: state.done, total: state.total, updated_at: state.updated_at, draft: false });
    }
    const known = new Set(map.keys());
    for (const dir of lsProjects()) {
        if (dir && !known.has(dir)) map.set(dir, { dir, done: null, total: null, updated_at: null, draft: true });
    }
    const arr = [...map.values()];
    arr.sort((a, b) => (b.updated_at ?? 0) - (a.updated_at ?? 0));
    return arr.slice(0, 40);
}

/* ---- 中栏 ---- */

function renderCenterColumn(colC, data) {
    const sig = cardsSignature(data);
    // 编辑中（常驻 textarea 聚焦 / 临时编辑器打开 / 视频播放中）跳过重建，防丢焦丢草稿
    const taFocus = !!colC.querySelector(".h3d-ta:focus");
    const locked = taFocus || !!colC.querySelector(".h3d-editor") || isVideoPlaying(colC);
    if (!locked && (sig !== desk.cardsSig || !colC.querySelector(".h3d-center-pad"))) {
        desk.cardsSig = sig;
        [...colC.children].slice(1).forEach((n) => n.remove());
        colC.append(buildCenterBody(data));
    }
}

function buildCenterBody(data) {
    const { node, state, mf, plan, drafts } = data;
    const wrap = el("div", "h3d-center-pad");
    const total = plan ? plan.length : 0;
    const done = state?.done ?? mf?.done ?? 0;

    /* 状态条 */
    const bar = el("div", "h3d-statusbar");
    const st = statusLine(state, mf, plan);
    bar.append(el("div", "h3d-st-text", escapeHtml(st.text)));
    if (state?.review) bar.insertAdjacentHTML("beforeend", badge("逐段审片", "cyan"));
    if (state?.reroll > 0) bar.insertAdjacentHTML("beforeend", badge(`重跑起始段=${state.reroll}`, "warn"));
    const refreshBtn = el("button", "h3d-btn h3d-refresh", "↻");
    refreshBtn.title = "重新读取节点与存档数据";
    refreshBtn.onclick = refresh;
    bar.append(refreshBtn);
    wrap.append(bar);

    /* 段落进度轨（按各段时长加权，宽度≈时长比例） */
    if (total > 0) {
        const defRaw = Number(getWidgetValue(node, W_DUR));
        const def = isFinite(defRaw) && defRaw > 0 ? defRaw : 5.0;
        const rail = el("div", "h3d-rail");
        for (let i = 0; i < Math.min(total, MAX_SEG); i++) {
            const it = plan[i];
            const sec = it?.kind === "prompt" && it.idx !== undefined
                ? (data.ds.segments[it.idx]?.seconds ?? def) : def;
            const sp = el("span", i < done ? "done" : i === done ? "next" : "");
            sp.style.flexGrow = String(Math.max(1, sec));
            sp.title = `段 ${i + 1} · ${sec}s${it?.kind === "insert" ? "（插入视频，按默认估）" : ""}`;
            rail.append(sp);
        }
        wrap.append(rail);
    }

    /* 未填写提示 */
    if (drafts && drafts.length) {
        wrap.append(el("div", "h3d-drafts",
            `未填写（运行时跳过）：${drafts.map(escapeHtml).join("、")}`));
    }

    /* 段落卡片 */
    wrap.append(buildCards(data));

    /* 添加行 */
    const addrow = el("div", "h3d-addrow");
    if (node) {
        const addSeg = el("button", "h3d-btn", "＋ 添加一段");
        addSeg.title = "新增一段提示词到导演台状态";
        addSeg.onclick = () => addPromptSegment(node);
        addrow.append(addSeg);
        if (hasInserts(node)) {
            if (total) {
                const tail = insertButton("尾部插入视频", total + 1);
                tail.title = `上传视频追加到成片末尾（${total + 1}|文件名）`;
                addrow.append(tail);
            }
        }
    } else if (window.H3_DEFAULT_WORKFLOW) {
        const wf = el("button", "h3d-btn h3d-btn-cyan", "⚡ 一键载入配套工作流");
        wf.title = "画布上还没有 H3 链节点：载入官方风格预置工作流（模型加载 + 主节点 + 常驻连线的素材池）";
        wf.onclick = loadDefaultWorkflow;
        addrow.append(wf);
    }
    wrap.append(addrow);

    /* 上次运行报告 */
    if (state?.report) {
        const det = el("details", "h3d-drawer");
        det.innerHTML = `<summary>上次运行报告</summary><pre>${escapeHtml(state.report)}</pre>`;
        wrap.append(det);
    }
    return wrap;
}

function buildCards(data) {
    const { node, state, mf, plan } = data;
    const done = mf?.done ?? state?.done ?? 0;
    const seeds = mf?.seeds || [];
    const seams = mf?.seams || [];
    const bridges = mf?.bridge_scores || [];
    const canInsert = hasInserts(node);
    const wrap = el("div", "h3d-cards");

    if (!plan.length) {
        wrap.append(el("div", "h3d-empty",
            node
                ? "还没有段落：点上方「＋ 添加一段」开始填写提示词。"
                : "画布上未找到 H3 Seamless Chain 节点，且暂无历史链数据。添加节点后这里成为段落流水线。"));
        return wrap;
    }

    plan.forEach((it, idx) => {
        const isDone = idx < done;
        const isInsert = it.kind === "insert";
        const isHead = isInsert && it.pos === 1;
        const card = el("div", "h3d-card" + (isDone ? "" : " todo"));
        const media = segMediaHtml(state, mf, idx);
        const thumb = el("div", "h3d-thumb");
        thumb.innerHTML = media
            ?? `<div>${isDone ? "…" : isInsert ? "插入段" : "待生成"}</div>`;
        const img = thumb.querySelector("img");
        if (img) img.onerror = () => { img.replaceWith(el("div", "", "⚠ 加载失败")); };

        const body = el("div", "h3d-cbody");
        const stateChip = !isDone
            ? badge(isInsert ? "插入段（未跑）" : "待生成", "")
            : isHead ? badge("片头（上传）", "media")
            : isInsert ? badge("插入段", "media") : badge("已完成", "ok");
        const title = el("div", "h3d-ctitle",
            `<span>段 ${idx + 1}${isInsert ? `（位置 ${it.pos}）` : ""}</span>${stateChip}`);

        /* 每段时长（秒）：留空=跟随节点默认；显示吸附后的帧数 */
        let secsHint = null;
        if (node && it.idx !== undefined) {
            const segData = data.ds.segments[it.idx] || defaultSegment();
            const defRaw = Number(getWidgetValue(node, W_DUR));
            const defSec = isFinite(defRaw) && defRaw > 0 ? defRaw : 5.0;
            const secsInp = document.createElement("input");
            secsInp.type = "number";
            secsInp.className = "h3d-secs";
            secsInp.min = "0.5"; secsInp.max = "15"; secsInp.step = "0.1";
            secsInp.placeholder = String(defSec);
            secsInp.value = segData.seconds ?? "";
            secsInp.title = "本段时长（秒）：留空=跟随右栏「每段时长」默认；内部自动吸附 17k+5 帧网格(@24fps)";
            secsHint = el("span", "h3d-secs-hint",
                `≈${snapFrames(segData.seconds ?? defSec)}帧`);
            const syncHint = (v) => {
                const num = Number(v);
                const sec = (v === "" || !isFinite(num) || num <= 0) ? defSec : num;
                secsHint.textContent = `≈${snapFrames(sec)}帧`;
            };
            secsInp.addEventListener("input", () => syncHint(secsInp.value));
            secsInp.addEventListener("change", () => {
                setSegmentSeconds(node, it.idx, secsInp.value);
                syncHint(secsInp.value);
            });
            title.append(secsInp, secsHint, el("span", "h3d-secs-hint", "秒"));
        }
        body.append(title);

        const seedTxt = isDone && !isInsert && seeds[idx] != null ? `种子 ${seeds[idx]}` : "";
        const seamTxt = isDone && seams[idx]
            ? `接缝 ${seams[idx][0]}${seams[idx][1] == null ? "" : ` / ${seams[idx][1]}dB`}` : "";
        const bridgeTxt = isDone && !isInsert && bridges[idx] != null ? `桥分 ${bridges[idx]}` : "";
        const meta = [seedTxt, seamTxt, bridgeTxt].filter(Boolean).join(" · ");
        if (meta) body.append(el("div", "h3d-cmeta", escapeHtml(meta)));

        /* 提示词：常驻 textarea（插入段显示文件名只读）—— 直接在台内编辑，实时写回 JSON 状态 */
        if (isInsert) {
            body.append(el("div", "h3d-cprompt", escapeHtml(it.file)));
        } else {
            const ta = document.createElement("textarea");
            ta.className = "h3d-ta";
            ta.rows = 4;
            ta.value = it.text || "";
            const pool = (data.ds.ref_assets || []);
            const poolHint = pool.length
                ? `用 [[${pool[0].label}]] 这样的标签引用素材，或手写 <Picture N>`
                : "上传参考图后可用 [[标签]] 引用";
            ta.placeholder = `第 ${idx + 1} 段画面描述：顺着上一段结尾继续；${poolHint}`;
            if (!node || it.idx === undefined) {
                ta.disabled = true;
                ta.title = it.idx === undefined ? "历史段只读（来自存档，非导演台状态）" : "画布上未找到节点";
            } else {
                ta.addEventListener("input", () => debouncePromptWrite(node, it.idx, ta.value));
                ta.addEventListener("blur", () => {
                    const key = `p${it.idx}`;
                    const t = _taTimers.get(key);
                    if (t) { clearTimeout(t); _taTimers.delete(key); }
                    setPromptText(node, it.idx, ta.value);
                    scheduleRefresh(200);
                });
            }
            body.append(ta);

            /* 引用素材：勾选本段用哪些图（后端按勾选顺序压实 <Picture k>） */
            if (node && it.idx !== undefined && data.ds.mode === "多参视频" && pool.length) {
                const seg = data.ds.segments[it.idx] || defaultSegment();
                const row = el("div", "h3d-refrow");
                row.append(el("label", "", "引用素材"));
                pool.forEach((a) => {
                    const on = (seg.refs || []).includes(a.label);
                    const chip = el("button", "h3d-refchip" + (on ? " on" : ""));
                    chip.type = "button";
                    chip.title = `勾选后本段仅引用这些图（按勾选顺序编号 <Picture k>）；提示词写 [[${a.label}]] 引用`;
                    const im = document.createElement("img");
                    im.loading = "lazy";
                    im.src = inputViewUrl(a.file);
                    im.onerror = () => im.remove();
                    chip.append(im, document.createTextNode(a.label));
                    chip.onclick = () => { toggleSegmentRef(node, it.idx, a.label); scheduleRefresh(80); };
                    row.append(chip);
                });
                if ((seg.refs || []).length) {
                    const reset = el("button", "h3d-btn", "↺ 全部");
                    reset.type = "button";
                    reset.style.padding = "3px 8px";
                    reset.style.fontSize = "10.5px";
                    reset.title = "清空段级选择 = 引用全部素材（与旧版行为一致）";
                    reset.onclick = () => {
                        const ds = getDs(node);
                        if (ds.segments[it.idx]) ds.segments[it.idx].refs = [];
                        setDs(node, ds);
                        scheduleRefresh(80);
                    };
                    row.append(reset);
                } else {
                    row.insertAdjacentHTML("beforeend",
                        '<span class="h3d-secs-hint">未勾选=引用全部</span>');
                }
                /* 引用语模板：把现成句式插入主提示词光标处 */
                const tpl = document.createElement("select");
                tpl.className = "h3d-reftpl";
                const optEmpty = document.createElement("option");
                optEmpty.value = "";
                optEmpty.textContent = "引用语…";
                tpl.append(optEmpty);
                REF_TEMPLATES.forEach(([name, fn], ti) => {
                    const o = document.createElement("option");
                    o.value = String(ti);
                    o.textContent = name;
                    tpl.append(o);
                });
                tpl.onchange = () => {
                    const ti = Number(tpl.value);
                    tpl.value = "";
                    if (!Number.isInteger(ti) || !REF_TEMPLATES[ti]) return;
                    const target = (seg.refs || [])[0] || pool[0].label;
                    const phrase = REF_TEMPLATES[ti][1](target);
                    insertAtCursor(ta, (ta.value && !/[\s,，。;；]$/.test(ta.value.slice(-1)) ? "\n" : "") + phrase);
                    setPromptText(node, it.idx, ta.value);
                    scheduleRefresh(150);
                };
                row.append(tpl);
                body.append(row);
            }

            /* 分段处理中心：场景/角色提示词（可展开折叠） */
            if (node && it.idx !== undefined) {
                const seg = (data.ds.segments && data.ds.segments[it.idx]) || defaultSegment();
                const hasSeg = !!(seg.scene_prompt || seg.character_prompt);
                const det = el("details", "h3d-seg-panel" + (hasSeg ? " has-content" : ""));
                det.open = hasSeg;
                det.innerHTML = `<summary>分段处理 · 场景 & 角色${hasSeg ? ' <span class="h3d-chip cyan">已填写</span>' : ""}</summary>`;
                const segBody = el("div", "h3d-seg-body");

                const sceneLabel = el("label", "h3d-seg-label", "场景提示词");
                const sceneTa = document.createElement("textarea");
                sceneTa.className = "h3d-ta h3d-seg-ta";
                sceneTa.rows = 2;
                sceneTa.value = seg.scene_prompt || "";
                sceneTa.placeholder = "本段场景描述（环境、光照、氛围等），留空则不附加";
                sceneTa.addEventListener("input", () => debounceSegmentWrite(node, it.idx, "scene_prompt", sceneTa.value));
                sceneTa.addEventListener("blur", () => {
                    const key = `s${it.idx}_scene_prompt`;
                    const t = _taTimers.get(key);
                    if (t) { clearTimeout(t); _taTimers.delete(key); }
                    setSegmentField(node, it.idx, "scene_prompt", sceneTa.value);
                    scheduleRefresh(200);
                });
                segBody.append(sceneLabel, sceneTa);

                const charLabel = el("label", "h3d-seg-label", "角色提示词");
                const charTa = document.createElement("textarea");
                charTa.className = "h3d-ta h3d-seg-ta";
                charTa.rows = 2;
                charTa.value = seg.character_prompt || "";
                charTa.placeholder = "本段角色描述（外貌、动作、表情等），留空则不附加";
                charTa.addEventListener("input", () => debounceSegmentWrite(node, it.idx, "character_prompt", charTa.value));
                charTa.addEventListener("blur", () => {
                    const key = `s${it.idx}_character_prompt`;
                    const t = _taTimers.get(key);
                    if (t) { clearTimeout(t); _taTimers.delete(key); }
                    setSegmentField(node, it.idx, "character_prompt", charTa.value);
                    scheduleRefresh(200);
                });
                segBody.append(charLabel, charTa);

                det.append(segBody);
                body.append(det);
            }
        }

        const actions = el("div", "h3d-actions");
        if (!node) {
            actions.append(el("span", "h3d-hint", "只读（画布上未找到节点）"));
        } else if (isInsert) {
            if (!isDone && canInsert) {
                const rm = el("button", "h3d-btn h3d-btn-danger", "✕ 移除插入");
                rm.title = "从「插入视频」清单删除这一条（其后段落自动重做）";
                rm.onclick = () => removeInsert(it.pos, it.file);
                actions.append(rm);
            } else if (!isDone) {
                actions.append(el("span", "h3d-hint", "当前引擎未合并插入能力"));
            } else {
                actions.append(el("span", "h3d-hint", "更换/移除插入 = 其后重做"));
            }
            if (canInsert) actions.append(insertButton("在此段后插入", idx + 2));
        } else {
            if (isDone) {
                const rerollBtn = el("button", "h3d-btn", "🎲 重摇此段");
                rerollBtn.title = `设「重跑起始段」=${idx + 1} 并换种子重新生成该段及之后`;
                rerollBtn.onclick = () => doReroll(idx + 1);
                actions.append(rerollBtn);
            } else if (canInsert) {
                actions.append(insertButton(`插视频到段 ${idx + 1} 前`, idx + 1));
            }
            if (node && it.idx !== undefined) {
                const rm = el("button", "h3d-btn h3d-btn-danger", "✕ 删除此段");
                rm.title = "删除这一段提示词（其后段落自动前移）";
                rm.onclick = () => removePromptSegment(node, it.idx);
                actions.append(rm);
            }
        }
        body.append(actions);

        card.append(thumb, body);
        wrap.append(card);
    });
    return wrap;
}

function insertButton(label, pos) {
    const b = el("button", "h3d-btn", `📎 ${label}`);
    b.title = `选择本地视频上传到 input 目录，并写入「插入视频」= ${pos}|文件名`;
    b.onclick = () => pickInsertVideo(pos);
    return b;
}

/* ---- 素材与参考（状态驱动：标签素材池存 JSON，缩略图直接回显；配套工作流节点做画布镜像） ---- */

function assetCard(title, file, onRemove) {
    const card = el("div", "h3d-asset" + (file ? " on" : ""));
    const thumb = el("div", "h3d-asset-thumb", file ? "" : "<span>IMG</span>");
    if (file) {
        const im = document.createElement("img");
        im.loading = "lazy";
        im.src = inputViewUrl(file);
        im.alt = title;
        im.onerror = () => { thumb.replaceChildren(el("span", "", "⚠")); };
        thumb.append(im);
    }
    const copy = el("div", "h3d-asset-copy");
    copy.innerHTML = `<strong>${escapeHtml(title)}</strong><small>${escapeHtml(file || "未设置")}</small>`;
    card.append(thumb, copy);
    if (file && onRemove) {
        const acts = el("div", "h3d-asset-acts");
        const rm = el("button", "h3d-btn h3d-btn-danger", "✕");
        rm.title = "移除";
        rm.onclick = onRemove;
        acts.append(rm);
        card.append(acts);
    }
    return card;
}

/** 多参模式：带标签编辑的素材卡（重命名同步所有段级引用） */
function labeledAssetCard(node, ds, idx) {
    const a = ds.ref_assets[idx];
    const file = a.file;
    const card = el("div", "h3d-asset on");
    const thumb = el("div", "h3d-asset-thumb");
    const im = document.createElement("img");
    im.loading = "lazy";
    im.src = inputViewUrl(file);
    im.alt = a.label;
    im.onerror = () => { thumb.replaceChildren(el("span", "", "⚠")); };
    thumb.append(im);

    const copy = el("div", "h3d-asset-copy");
    const labelInp = document.createElement("input");
    labelInp.className = "h3d-labelinp";
    labelInp.value = a.label;
    labelInp.spellcheck = false;
    labelInp.maxLength = 12;
    labelInp.title = "素材标签：提示词用 [[标签]] 引用；回车或失焦提交，重名自动加后缀";
    const commit = () => {
        const next = renameAssetLabel(node, idx, labelInp.value);
        if (next !== labelInp.value) labelInp.value = next;
        scheduleRefresh(120);
    };
    labelInp.addEventListener("change", commit);
    labelInp.addEventListener("keydown", (e) => { if (e.key === "Enter") { e.preventDefault(); labelInp.blur(); } });
    const quick = el("div", "h3d-quicklbl");
    QUICK_LABELS.forEach((q) => {
        const b = el("button", "", q);
        b.type = "button";
        b.title = `把标签设为「${q}」（被占用时自动加后缀）`;
        b.onclick = () => {
            const taken = new Set(ds.ref_assets.filter((_x, i) => i !== idx).map((x) => x.label));
            const want = uniqueLabelFrom(taken, q);
            labelInp.value = want;
            const got = renameAssetLabel(node, idx, want);
            labelInp.value = got;
            scheduleRefresh(120);
        };
        quick.append(b);
    });
    const usage = el("div", "h3d-asset-usage");
    const usedBy = (ds.segments || []).filter((s) => (s.refs || []).includes(a.label)).length;
    const defaultAll = (ds.segments || []).every((s) => !(s.refs || []).length);
    usage.innerHTML = `<span class="h3d-chip">全局图${idx + 1}</span>`
        + (usedBy ? `<span class="h3d-chip ok">${usedBy} 段指定</span>` : "")
        + (defaultAll && (ds.segments || []).length ? '<span class="h3d-chip cyan">默认全段引用</span>' : "");
    copy.append(labelInp, quick, usage,
        el("small", "", `${escapeHtml(file)} · 提示词写 [[${escapeHtml(a.label)}]]`));

    const acts = el("div", "h3d-asset-acts");
    const rm = el("button", "h3d-btn h3d-btn-danger", "✕");
    rm.title = "移除该素材（画布镜像节点隐藏，连线保留）";
    rm.onclick = () => { removeRefImage(node, idx); };
    acts.append(rm);
    card.append(thumb, copy, acts);
    return card;
}

function renderAssetsZone(sec, data) {
    const { node, ds } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>素材池 · 分段处理中心</strong><small>上传素材 → 打标签 → 各段卡片勾选引用</small>"));

    /* 模式条 */
    const mode = ds.mode;
    const mbar = el("div", "h3d-modebar");
    for (const [code, label, desc] of MODES) {
        const b = el("button", "h3d-mode" + (code === mode ? " active" : ""));
        const sub = code === "多参视频" ? "ref2va · [[标签]]"
            : code === "首帧视频" ? "fl2va · 首帧" : "fl2va · 纯文本";
        b.innerHTML = `${escapeHtml(label)}<small>${sub}</small>`;
        b.title = desc;
        b.onclick = () => { if (!node) return; setMode(node, code); scheduleRefresh(120); };
        mbar.append(b);
    }
    sec.append(mbar);

    const box = el("div", "h3d-assets");
    if (!node) {
        if (window.H3_DEFAULT_WORKFLOW) {
            const guide = el("div", "h3d-loadwf");
            guide.innerHTML = "<p>画布上还没有 H3 Seamless Chain 节点</p>"
                + "<p>一键载入配套工作流：模型加载 + 主节点 + 常驻连线的素材池<br>（素材节点默认隐藏，上传素材时自动点亮，不再动态建线）</p>";
            const wf = el("button", "h3d-btn h3d-btn-cyan", "⚡ 一键载入配套工作流");
            wf.onclick = loadDefaultWorkflow;
            guide.append(wf);
            box.append(guide);
        } else {
            box.append(el("div", "h3d-empty", "画布上未找到 H3 Seamless Chain 节点"));
        }
        sec.append(box);
        return;
    }

    if (mode === "文生视频") {
        box.append(el("div", "h3d-empty",
            "文生模式：无需图片素材，提示词直接描述画面即可。<br>需要起手图请切「首帧」，需要多参考请切「多参」。"));
    } else if (mode === "首帧视频") {
        const file = ds.first_frame;
        const card = assetCard("首帧图片", file,
            file ? () => { removeFirstFrame(node); } : null);
        card.querySelector("small").textContent = file || "未设置（第 1 段的起始帧）";
        const acts = card.querySelector(".h3d-asset-acts") || el("div", "h3d-asset-acts");
        const up = el("button", "h3d-btn h3d-btn-cyan", file ? "替换" : "上传");
        up.title = "上传图片到 input 目录，文件名存入导演台状态，画布「首帧图」节点同步点亮";
        up.onclick = () => pickImage("first");
        acts.insertBefore(up, acts.firstChild);
        if (!card.querySelector(".h3d-asset-acts")) card.append(acts);
        box.append(card);
    } else {
        /* 多参：标签素材池（状态驱动 + 画布镜像） */
        ds.ref_assets.forEach((_a, i) => {
            box.append(labeledAssetCard(node, ds, i));
        });
        if (ds.ref_assets.length < 9) {
            const add = el("button", "h3d-btn h3d-addasset", "＋ 添加参考图片");
            add.title = "上传图片 → 存入标签素材池（配套工作流对应节点自动点亮；手摆工作流不受影响）";
            add.onclick = () => pickImage("ref");
            box.append(add);
        }
    }
    sec.append(box);
    sec.append(el("div", "h3d-foot",
        "标签素材池：每张图打标签（角色1 / 场景1…），段落卡片里勾选本段引用哪些，提示词写 [[标签]] 引用（后端按段重编号为 &lt;Picture k&gt;）。<br>"
        + "配套工作流下素材节点由导演台点亮/隐藏——连线常驻，只是隐藏，请勿删除。"));
}

/* ---- 链参数（面板直写画布控件）：常规五项 + 高级设置收纳其余全部 ---- */

const PRIMARY_DEFS = [W_AR, W_MP, W_DUR, W_SEED, "步数"];
const ADVANCED_DEFS = [
    "引导帧数", "CFG", "采样器", "调度器",
    "自动存档", "存档目录", "审片模式", "自动保存", "重跑起始段",
    "桥帧门控", "清晰度阈值", "回退上限", "接缝混合", "混合帧数", "锚定加噪",
    W_WIDTH, W_HEIGHT,
];
const PARAM_LABELS = { [W_DUR]: "每段时长(秒) · 新段默认" };

function paramsSig(node) {
    if (!node) return "n";
    return [...PRIMARY_DEFS, ...ADVANCED_DEFS].map((n) => {
        const w = (node.widgets || []).find((x) => x.name === n);
        return w ? String(w.value) : "-";
    }).join("|");
}

/** 单个节点控件的表单域（combo→select，数值→number 输入；种子带🎲） */
function renderWidgetField(node, name, labelOverride) {
    const w = (node.widgets || []).find((x) => x.name === name);
    const field = el("div", "h3d-param");
    field.append(el("label", "", escapeHtml(labelOverride || name)));
    if (!w) {
        field.append(el("span", "h3d-hint", "—"));
        return field;
    }
    const opts = w.options && Array.isArray(w.options.values) ? w.options.values : null;
    if (opts) {
        const sel = document.createElement("select");
        sel.className = "h3d-select";
        for (const v of opts) {
            const o = document.createElement("option");
            o.value = v;
            o.textContent = v;
            if (String(v) === String(w.value)) o.selected = true;
            sel.append(o);
        }
        sel.onchange = () => setWidgetValue(node, name, sel.value);
        field.append(sel);
    } else {
        const row = el("div", "h3d-seedrow");
        const inp = document.createElement("input");
        inp.type = "number";
        inp.value = w.value;
        const step = w.options && w.options.step;
        inp.step = step || (name === "CFG" || name === W_DUR ? 0.1 : 1);
        inp.onchange = () => setWidgetValue(node, name, Number(inp.value));
        row.append(inp);
        if (name === W_SEED) {
            const dice = el("button", "h3d-btn", "🎲");
            dice.title = "随机种子";
            dice.onclick = () => {
                const v = Math.floor(Math.random() * 2 ** 48);
                setWidgetValue(node, name, v);
                inp.value = v;
            };
            row.append(dice);
        }
        field.append(row);
    }
    return field;
}

function renderParamsZone(sec, data) {
    const { node } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>链参数</strong><small>常规五项直改；其余收进「高级设置」</small>"));
    if (!node) {
        sec.append(el("div", "h3d-empty", "画布上未找到节点，参数面板不可用"));
        return;
    }
    const grid = el("div", "h3d-params");
    for (const name of PRIMARY_DEFS) {
        grid.append(renderWidgetField(node, name, PARAM_LABELS[name]));
    }
    /* 换算徽章：宽高比+百万像素 -> 实际画布；自定义模式改为显示宽/高输入 */
    const ar = String(getWidgetValue(node, W_AR) ?? "");
    if (AR_RATIO[ar]) {
        const badgeTxt = canvasBadgeText(node);
        if (badgeTxt) {
            const b = el("div", "h3d-convbadge", `${escapeHtml(badgeTxt)} · 32倍数对齐`);
            b.title = "官方 Resolution Selector 同款换算：短边≤768、长边≤1344（H3 原生画布上限）";
            grid.append(b);
        }
    } else {
        grid.append(renderWidgetField(node, W_WIDTH));
        grid.append(renderWidgetField(node, W_HEIGHT));
    }
    /* 高级设置：其余全部参数（默认收起） */
    const adv = el("details", "h3d-adv");
    adv.innerHTML = "<summary>⚙ 高级设置（引导 / 采样 / 审片 / 存档 / 接缝 / 画布微调）</summary>";
    const agrid = el("div", "h3d-adv-grid");
    for (const name of ADVANCED_DEFS) {
        if (ar !== "自定义" && (name === W_WIDTH || name === W_HEIGHT)) continue;
        agrid.append(renderWidgetField(node, name));
    }
    adv.append(agrid);
    grid.append(adv);
    sec.append(grid);
    sec.append(el("div", "h3d-foot",
        "全部节点参数已收纳于此：改「高级设置」同样随工作流保存；「生成模式」由左侧模式条控制。"));
}

/* ---- 右栏 ---- */

function renderHistoryZone(sec, data) {
    const { history, prefix, state } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>成片历史</strong><small>播放 / 下载 / 删除（输出目录随前缀变化）</small>"));
    const box = el("div", "h3d-hist");
    const items = (history || []).slice(0, 10);

    if (!items.length) {
        box.append(el("div", "h3d-empty",
            `还没有成片：工作流跑完后，最终视频与分段会出现在 output/${escapeHtml(prefix)}/<br>（接 H3ChainSaver 节点或开启「自动保存」）`));
    } else {
        items.forEach((it, i) => {
            const url = viewUrl(prefix, it.file);
            const card = el("div", "h3d-result" + (i === 0 ? " current" : ""));
            const v = document.createElement("video");
            v.controls = true;
            v.preload = "metadata";
            v.src = url;
            card.append(v, el("div", "h3d-result-name", (i === 0 ? "当前成片 · " : "") + escapeHtml(it.file)));
            card.append(el("div", "h3d-result-info",
                escapeHtml(`${fmtTime(it.time)} · ${it.frames || "?"} 帧` +
                    (it.archive ? ` · 存档 ${it.archive}` : "") +
                    (it.segments ? ` · 分段×${it.segments}` : ""))));
            const meta = el("div", "h3d-result-meta");
            const acts = el("div", "h3d-result-acts");
            const open = el("a", "h3d-dl", "↗ 打开");
            open.href = url; open.target = "_blank";
            const dl = el("a", "h3d-dl", "⬇ 下载");
            dl.href = url; dl.download = it.file;
            acts.append(open, dl);
            if (deleteRouteOk) {
                const del = el("button", "h3d-btn h3d-btn-danger", "🗑 删除");
                del.title = "删除成片文件（二次确认）";
                del.onclick = async () => {
                    if (del.dataset.armed !== "1") {
                        del.dataset.armed = "1";
                        del.textContent = "确认删除？";
                        setTimeout(() => { del.dataset.armed = ""; del.textContent = "🗑 删除"; }, 2500);
                        return;
                    }
                    try {
                        const r = await fetch("/h3chain/delete", {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ file: `${prefix}/${it.file}` }),
                        });
                        if (r.ok) { refresh(); return; }
                        if (r.status === 404) { deleteRouteOk = false; alert("删除路由不可用：当前引擎分支未合并 /h3chain/delete，已隐藏删除按钮"); refresh(); return; }
                        alert("删除失败（文件可能已被移动）");
                    } catch (e) {
                        deleteRouteOk = false;
                        alert("删除路由不可用：已隐藏删除按钮");
                        refresh();
                    }
                };
                acts.append(del);
            }
            meta.append(acts);
            card.append(meta);
            box.append(card);
        });
        if ((history || []).length > 10) {
            box.append(el("div", "h3d-foot", `仅显示最近 10 条（共 ${history.length} 条），更早在 output/${escapeHtml(prefix)}/`));
        }
    }
    sec.append(box);
    if (state?.dir) {
        sec.append(el("div", "h3d-foot", `分段存档：output/checkpoints/${escapeHtml(state.dir)}`));
    }
}

/* ---- 页脚 ---- */

function renderFooter(z, data) {
    const { node, state, mf, plan } = data;
    const total = plan ? plan.length : 0;
    const done = state?.done ?? mf?.done ?? 0;

    const infos = [];
    infos.push(`当前 <b>${escapeHtml(state?.dir || (node ? getDirValue(node) || "未命名" : "无"))}</b>`);
    if (node) {
        const ps = paramsSummary(node, mf);
        const totalSec = total ? `共${chainSeconds(node, data.ds, plan).toFixed(1)}s` : "";
        const geoInfo = [ps.geo, totalSec].filter(Boolean).join(" · ");
        if (geoInfo) infos.push(`<b>${escapeHtml(geoInfo)}</b>`);
    }
    if (state?.review) infos.push("逐段审片：每次排队只生成下一段");
    if (!node) infos.push("只读模式：画布上未找到 H3 Seamless Chain 节点");
    if (node && !hasInserts(node)) infos.push("当前引擎未合并「插入视频」能力（合并插入分支后自动启用）");
    z.footInfo.innerHTML = infos.map((s) => `<span>${s}</span>`).join("");

    const run = z.run;
    run.onclick = queuePrompt;
    if (!total) {
        run.disabled = true;
        run.textContent = "▶ 开始生成（先配置段落）";
    } else if (done >= total) {
        run.disabled = true;
        run.textContent = `✓ 本链已完成（${total} 段）`;
    } else {
        run.disabled = false;
        run.textContent = done > 0
            ? `▶ 继续下一段（段 ${done + 1}）`
            : `▶ 开始生成（共 ${total} 段）`;
    }
}

/* ---- 新建项目模态 ---- */

function openNewProjectModal() {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点（只读模式）"); return; }
    if (document.querySelector(".h3d-overlay")) return;

    const t = new Date();
    const pad = (x) => String(x).padStart(2, "0");
    const def = `h3chain_${t.getFullYear()}${pad(t.getMonth() + 1)}${pad(t.getDate())}_${pad(t.getHours())}${pad(t.getMinutes())}`;

    const overlay = el("div", "h3d-overlay");
    const dialog = el("div", "h3d-dialog");
    dialog.innerHTML = `
        <h3>＋ 新建项目</h3>
        <p class="h3d-lead">新开一条视频链：换存档目录名即换链，旧链原样保留可随时切回。
        提示词沿用当前内容作底稿（可勾选下方清空）。</p>`;
    const input = document.createElement("input");
    input.type = "text";
    input.value = def;
    input.spellcheck = false;
    const err = el("div", "h3d-err", "");
    const check = el("label", "h3d-check");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    check.append(cb, document.createTextNode("创建后清空全部提示词（否则沿用当前底稿）"));
    const row = el("div", "h3d-dialog-row");
    const cancel = el("button", "h3d-btn", "取消");
    const ok = el("button", "h3d-btn h3d-btn-cta", "创建项目");
    row.append(cancel, ok);
    dialog.append(input, err, check, row);
    overlay.append(dialog);
    overlay.addEventListener("pointerdown", (e) => { if (e.target === overlay) overlay.remove(); });
    document.body.append(overlay);
    input.focus();
    input.select();

    const close = () => overlay.remove();
    cancel.onclick = close;
    overlay.addEventListener("keydown", (e) => { if (e.key === "Escape") close(); });
    const submit = () => {
        const name = input.value.trim();
        if (!name) { err.textContent = "项目名不能为空"; return; }
        if (!/^[0-9A-Za-z_\-一-龥]+$/.test(name)) {
            err.textContent = "仅允许中文、字母、数字、下划线、连字符";
            return;
        }
        if (!setDirValue(node, name)) { err.textContent = "节点上没有「存档目录/断点目录」控件"; return; }
        setWidgetValue(node, W_REROLL, 0);
        if (cb.checked) clearPrompts(node);
        rememberProject(name);
        close();
        scheduleRefresh(200);
    };
    ok.onclick = submit;
    input.addEventListener("keydown", (e) => { if (e.key === "Enter") submit(); });
}

/* ---------- 刷新 ---------- */

async function refresh() {
    try {
        const data = await collectData();
        renderMini(data);
        if (desk) updateDesk(data);
    } catch (e) {
        console.warn("[h3-director] refresh failed:", e);
    }
}

/* ---------- 挂载 ---------- */

const FAB_ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>`;

function mountSidebar(target) {
    miniBox = target;
    refresh();
}

function mountFallbackFab() {
    const btn = document.createElement("button");
    btn.className = "h3d-fab";
    btn.title = "长片导演台（H3 Seamless Chain）";
    btn.innerHTML = FAB_ICON;
    btn.onclick = openDesk;
    document.body.append(btn);
}

app.registerExtension({
    name: "H3SeamlessChain.DirectorDesk",
    setup() {
        injectStyles();

        /* 旧工作流迁移：widget 参数官方化后按位错位，载入画布前重排 widgets_values */
        if (typeof app.loadGraphData === "function") {
            const origLoadGraphData = app.loadGraphData.bind(app);
            app.loadGraphData = function (data, ...args) {
                return origLoadGraphData(migrateGraphWidgets(data), ...args);
            };
        }

        if (app.extensionManager && typeof app.extensionManager.registerSidebarTab === "function") {
            app.extensionManager.registerSidebarTab({
                id: "h3chain-director",
                // 图标必须是已加载图标库的类名（前端内置 PrimeVue），传 SVG 源码不会渲染
                icon: "pi pi-video",
                title: "长片导演台",
                tooltip: "H3 Seamless Chain：项目管理 + 段落流水线 + 成片历史（继续下一段 / 重摇 / 改词 / 插入视频）",
                type: "custom",
                render: (elTarget) => mountSidebar(elTarget),
            });
        } else {
            mountFallbackFab();
        }

        api.addEventListener("executing", ({ detail }) => {
            if (detail === null) { scheduleRefresh(); return; } // 队列清空
            const n = (app.graph?._nodes || []).find((x) => String(x.id) === String(detail));
            if (n && (n.type === NODE_TYPE || n.type === SAVER_TYPE)) {
                setLed("running", "生成中");
            }
        });
        api.addEventListener("execution_success", () => {
            if (pendingReset) {
                const node = findNode();
                if (node) setWidgetValue(node, W_REROLL, 0);
                pendingReset = false;
            }
            setLed("done", "生成完成");
            scheduleRefresh();
        });
        api.addEventListener("execution_error", () => {
            setLed("error", "运行出错");
            scheduleRefresh();
        });
    },
});
