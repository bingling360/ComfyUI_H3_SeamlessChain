/**
 * H3 Seamless Chain —— 「长片导演台」（全屏一体化控制台 + 侧栏迷你入口）
 *
 * 参照「H3 一体化总控导演台」状态驱动架构：所有输入（模式/提示词/首帧/参考素材）
 * 存入节点「导演台状态」JSON widget，节点 execute 读取 JSON 加载素材；
 * 画布节点只作镜像/兜底（配套工作流的常驻隐藏节点），不在画布上动态建线。
 *
 * 状态 JSON schema v2（存入「导演台状态」widget）：
 *   { mode:"文生视频"|"首帧视频"|"多参视频",
 *     prompts:["段1文本","段2文本",...],
 *     first_frame:"input目录下的文件名",
 *     ref_assets:[ { file:"文件名", kind:"image"|"video"|"audio", label:"角色1" }, ... ],
 *                                // 标签素材池（真源，三类混排；官方单段上限 图9/视3/音3）
 *     ref_images:["文件名1",...], // 兼容保留 = ref_assets 中图片类文件的序列
 *     segments:[
 *       { scene_prompt:"场景描述", character_prompt:"角色描述",
 *         seconds:6.5,        // 本段时长（秒），null=跟随节点「每段时长」默认
 *         refs:["角色1","场景1"] },  // 本段引用的素材标签（跨类别），[]=引用全部
 *       ...
 *     ] }
 *
 * 标签引用：提示词写 [[角色1]]，后端按段按类别压实重编号（图→<Picture k>、视→<Video k>、
 * 音→<Audio j>）；原生 token 写法继续兼容。v1 状态（只有 ref_images）自动迁移为图片类。
 * 段级注入：未勾选的素材完全不进该段 conditioning；参考视频的原声自动配对成 <Audio j>。
 *
 * 模式互斥在 JSON 层完成：切模式即清空不相容字段，后端复验不匹配报错。
 * 分段处理中心：segments 与 prompts 等长，每段的 scene_prompt/character_prompt
 * 由后端组合到该段主提示词前（scene → character → 主提示词），不影响核心采样。
 *
 * 配套工作流（web/h3_default_workflow.js）：画布无 H3 节点时可一键载入官方风格
 * 预置工作流（含提示词×3 镜像 + 分段输出链 CreateVideo→SaveVideo）；
 * 「素材池 · 自动管理」组的 LoadImage/LoadVideo/LoadAudio 常驻预连，导演台上传
 * 素材时点亮对应节点（mode=0），删除时隐藏（mode=2+折叠）——连线始终存在，只做显隐。
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
const MP_LIST = [...Array.from({ length: 20 }, (_v, i) => ((i + 1) / 10).toFixed(1)), "0.98"].sort((a, b) => a - b);   // 0.1–2.0 共 20 档 + 0.98（官方 1344×768 原生档，旧档迁移用）
const QUICK_LABELS = ["角色1", "角色2", "场景1", "场景2", "风格", "道具"];
/* 素材类别：与后端 REF_CAPS/_KIND_NAME 对齐（官方单段上限 图9/视3/音3） */
const KIND_LIST = ["image", "video", "audio"];
const KIND_NAME = { image: "图片", video: "视频", audio: "音频" };
const KIND_ICON = { image: "🖼", video: "🎞", audio: "🎵" };
const KIND_TOKEN = { image: "Picture", video: "Video", audio: "Audio" };
const KIND_CAPS = { image: 9, video: 3, audio: 3 };
const KIND_ACCEPT = { image: "image/*", video: "video/*", audio: "audio/*" };
/* 引用语模板库：插入到提示词光标处，[[标签]] 由后端按段压实为 <Picture k> */
const REF_TEMPLATES = [
    ["插入 [[标签]]", (l) => `[[${l}]]`],
    ["主角出场", (l) => `主角 [[${l}]] 全程出镜（主体身份、外观与服饰全程保持一致）`],
    ["配角出场", (l) => `画面中出现的 [[${l}]] 为次要角色，身份与外观保持一致`],
    ["场景还原", (l) => `场景以 [[${l}]] 为准，延续其环境、光照与空间布局`],
    ["风格参考", (l) => `整体画风、色调与质感参考 [[${l}]]`],
    ["镜头参考", (l) => `运镜方式参考 [[${l}]]（可用官方词汇：Push In / Pan Left / Truck Right / Tracking Shot，加 with small amplitude at slow speed 等修饰）`],
    ["说话人", () => `短发女主 (S1) 轻声说：「……」`],
];
const MODES = [
    ["文生视频", "文生", "纯文本，fl2va UNET，不接图片"],
    ["首帧视频", "首帧", "首帧图片起手，fl2va UNET"],
    ["多参视频", "多参", "参考图/视频/音频，ref2va UNET，[[标签]] 引用"],
];
const MODE_DEFAULT = "文生视频";
const MAX_SEG = 64;

let miniBox = null;
let desk = null;
let pendingReset = false;
let refreshTimer = null;
let ledPhase = "idle";
let ledText = "待命";
let apiErrorText = "";

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

/* 项目存档后端接口（/h3chain/*，与 /api/view 文件读取不同源） */
async function apiGet(path) {
    try {
        const r = await api.fetchApi(path);
        if (!r.ok) return { ok: false, status: r.status };
        return { ok: true, data: await r.json() };
    } catch (e) {
        return { ok: false, status: 0, error: String(e) };
    }
}

function setApiError(text) {
    apiErrorText = text || "";
    document.querySelectorAll(".h3d-banner").forEach((b) => {
        b.textContent = apiErrorText;
        b.style.display = apiErrorText ? "" : "none";
    });
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
    const total = parseFloat(mp) * 1024 * 1024;
    if (!isFinite(total) || total <= 0) return null;
    const fit = (x) => {
        const u = x / 32, f = Math.floor(u), d = u - f;
        const k = d === 0.5 ? (f % 2 === 0 ? f : f + 1) : Math.round(u);   // Python round：.5 取偶
        return Math.max(32, k * 32);
    };
    return [fit(Math.sqrt(total * r)), fit(Math.sqrt(total / r))];
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

/** 反推：宽高完全命中某 AR×MP 组合则返回 [ar, mp]，否则 null（旧工作流迁移用）。
 *  mp 返回数值（「百万像素」已改浮点控件，写字符串会留下类型混杂） */
function matchCanvasCombo(w, h) {
    for (const ar of Object.keys(AR_RATIO)) {
        for (const mp of MP_LIST) {
            const c = resolveCanvas(ar, mp);
            if (c && c[0] === w && c[1] === h) return [ar, Number(mp)];
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
        combo ? combo[1] : 0.5,              // 百万像素（浮点控件）
        Number(w), Number(h),                // 宽/高（自定义模式继续生效，存档指纹不变）
        secs, guide, Number(seed), ctrl, ...tail,
    ];
}

/* 导演台时代（25 值）→ 统一接缝后端（35 值）：尾部错位修正。
 * 25 值布局尾部 […, 重跑起始段, 生成模式, 导演台状态] 直接按位载入 35 控件会把
 * 生成模式/导演台状态错塞到精修强度/精修窗口上；此处把后端新增的 10 个接缝控件
 * 默认值插入「重跑起始段」之后，并把「接缝混合」旧值映射到「接缝处理」旧值档
 * （与后端 backfill 同规则），其余位置不变。 */
const V25_WIDGET_COUNT = 25;   // 24 控件 + 种子 control 占 1 位
const SEAM_LEGACY_MAP = { "smoothstep": "smoothstep像素混合", "潜空间精修": "潜空间精修", "关闭": "关闭" };
const V35_SEAM_DEFAULTS = [0.45, "39", "自动", 0.06, 1, "关闭", "关闭", 17, 48, "开启"];

function remapV25WidgetValues(wv) {
    const out = wv.slice(0, 23);             // 宽高比…重跑起始段（0..22 位布局不变）
    out[17] = SEAM_LEGACY_MAP[out[17]] || "标准";   // 接缝混合旧值 → 接缝处理
    out.push(...V35_SEAM_DEFAULTS);          // 新增 10 个接缝控件取后端默认值
    out.push(...wv.slice(23));               // 生成模式 / 导演台状态原样保留
    return out;
}

function migrateGraphWidgets(graphData) {
    if (!graphData || !Array.isArray(graphData.nodes)) return graphData;
    let migrated = 0;
    for (const n of graphData.nodes) {
        if (n.type !== NODE_TYPE || !Array.isArray(n.widgets_values) || !n.widgets_values.length) continue;
        try {
            if (typeof n.widgets_values[0] !== "string") {
                n.widgets_values = remapOldWidgetValues(n.widgets_values);
                migrated += 1;
            } else if (n.widgets_values.length === V25_WIDGET_COUNT) {
                n.widgets_values = remapV25WidgetValues(n.widgets_values);
                migrated += 1;
            }
        } catch (e) { /* 解析失败保持原样，交给兜底修正 */ }
    }
    if (migrated) console.log(`[h3-director] 已迁移 ${migrated} 个旧版 H3 节点的参数（旧布局 → 宽高比+百万像素+秒 / 统一接缝后端）`);
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
    return { mode: MODE_DEFAULT, prompts: [""], first_frame: "", last_frame: "", ref_images: [], ref_assets: [], segments: [] };
}

function defaultSegment() {
    return { scene_prompt: "", character_prompt: "", soundscape: "", music: "", seconds: null, refs: [] };
}

function getDs(node) {
    if (!node) return defaultDs();
    const w = (node.widgets || []).find((x) => x.name === W_DS);
    if (!w) return defaultDs();
    try {
        const raw = w.value ? JSON.parse(w.value) : {};
        const prompts = Array.isArray(raw.prompts) && raw.prompts.length ? raw.prompts.map(String) : [""];

        /* v2 标签素材池（含 kind 三类）；v1（只有 ref_images）自动迁移为图片 */
        let refAssets = Array.isArray(raw.ref_assets)
            ? raw.ref_assets.filter((a) => a && typeof a === "object" && a.file)
                .map((a) => ({
                    file: String(a.file),
                    kind: KIND_LIST.includes(a.kind) ? a.kind : "image",
                    label: cleanLabel(a.label) || "",
                }))
            : null;
        if (!refAssets) {
            const legacy = Array.isArray(raw.ref_images) ? raw.ref_images.filter(String).map(String) : [];
            refAssets = legacy.map((file) => ({ file, kind: "image", label: "" }));
        }
        const kindCount = { image: 0, video: 0, audio: 0 };
        refAssets.forEach((a) => {
            if (!a.label) { kindCount[a.kind] += 1; a.label = `${KIND_NAME[a.kind]}${kindCount[a.kind]}`; }
        });
        const taken = new Set();
        refAssets.forEach((a) => {
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
                soundscape: typeof s?.soundscape === "string" ? s.soundscape : "",
                music: typeof s?.music === "string" ? s.music : "",
                seconds: (isFinite(sec) && sec > 0) ? Math.min(15, Math.max(0.5, sec)) : null,
                refs: Array.isArray(s?.refs) ? s.refs.map(String).filter((l) => validLabels.has(l)) : [],
            };
        });
        return {
            mode: MODES.some(([m]) => m === raw.mode) ? raw.mode : MODE_DEFAULT,
            prompts,
            first_frame: typeof raw.first_frame === "string" ? raw.first_frame : "",
            last_frame: typeof raw.last_frame === "string" ? raw.last_frame : "",
            ref_images: refAssets.filter((a) => a.kind === "image").map((a) => a.file),
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
    /* ref_images 兼容字段 = ref_assets 图片类文件序列（旧版前端/后端仍可读） */
    if (Array.isArray(ds.ref_assets)) {
        ds.ref_assets = ds.ref_assets.filter((a) => a && a.file);
        ds.ref_images = ds.ref_assets.filter((a) => (a.kind || "image") === "image")
            .map((a) => String(a.file));
    }
    w.value = JSON.stringify(ds);
    if (typeof w.callback === "function") {
        try { w.callback(w.value); } catch (e) { /* callback 可选 */ }
    }
    node.setDirtyCanvas(true, true);
    node.graph?.change?.();
    syncMirrors(node, ds);   // 状态写入统一同步画布镜像（提示词/首帧/素材，幂等）
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

/* ---- 分段处理中心：场景/角色/声音提示词 + 每段时长 + 段级素材引用（状态驱动） ---- */

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

/** 勾选/取消本段引用的素材标签；全部取消 = 引用全部（与后端约定一致）。
 *  勾选时按类别校验官方单段上限（图9/视3/音3），超限拦截并提示。 */
function toggleSegmentRef(node, idx, label) {
    const ds = getDs(node);
    if (idx < 0 || idx >= (ds.segments || []).length) return;
    const seg = ds.segments[idx];
    if (!Array.isArray(seg.refs)) seg.refs = [];
    const pos = seg.refs.indexOf(label);
    if (pos >= 0) {
        seg.refs.splice(pos, 1);
    } else {
        const asset = (ds.ref_assets || []).find((a) => a.label === label);
        const kind = asset ? asset.kind : "image";
        const sameKind = seg.refs.filter((l) => {
            const a = (ds.ref_assets || []).find((x) => x.label === l);
            return a && a.kind === kind;
        }).length;
        if (sameKind >= KIND_CAPS[kind]) {
            alert(`本段引用${KIND_NAME[kind]}素材已达官方上限 ${KIND_CAPS[kind]} 个：请先取消一个再勾选「${label}」`);
            return;
        }
        seg.refs.push(label);
    }
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

function addAsset(node, kind, filename) {
    const k = KIND_LIST.includes(kind) ? kind : "image";
    const ds = getDs(node);
    const sameKind = ds.ref_assets.filter((a) => a.kind === k).length;
    if (sameKind >= KIND_CAPS[k]) { alert(`${KIND_NAME[k]}素材最多 ${KIND_CAPS[k]} 个（官方单段上限）`); return; }
    const taken = new Set(ds.ref_assets.map((a) => a.label));
    const label = uniqueLabelFrom(taken, `${KIND_NAME[k]}${sameKind + 1}`);
    ds.ref_assets.push({ file: filename, kind: k, label });
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

/* 尾帧锚定（身份锚点，任意模式可用）：与首帧同机制，但不随模式切换清空 */
function setLastFrame(node, filename) {
    const ds = getDs(node);
    ds.last_frame = filename || "";
    setDs(node, ds);
    syncMirrors(node, ds);
    scheduleRefresh(120);
}

function removeLastFrame(node) {
    setLastFrame(node, "");
}

/* ---- 配套工作流：素材/提示词节点镜像（连线常驻，仅切换 点亮/隐藏） ---- */

function mirrorNodeByTitle(title) {
    return (app.graph?._nodes || []).find((n) => n.getTitle?.() === title || n.title === title) || null;
}

/** 点亮/隐藏配套工作流节点并同步其值：有值=mode0+展开，无值=mode2+折叠（连线与配置保留）。
 *  widgetNames 为候选控件名列表（LoadImage=image / LoadVideo=file / LoadAudio=audio / Primitive=value），
 *  都找不到时退回第一个控件。 */
function setMirrorNode(title, value, widgetNames) {
    const m = mirrorNodeByTitle(title);
    if (!m) return false;
    let w = null;
    for (const n of (widgetNames || [])) {
        w = (m.widgets || []).find((x) => x.name === n);
        if (w) break;
    }
    w = w || (m.widgets || [])[0];
    if (w && value && String(w.value ?? "") !== String(value)) {
        w.value = value;
        if (typeof w.callback === "function") { try { w.callback(value); } catch (e) { /* 可选 */ } }
    }
    const want = value ? 0 : 2;
    if (m.mode !== want) m.mode = want;
    m.flags = m.flags || {};
    m.flags.collapsed = !value;
    m.setDirtyCanvas?.(true, true);
    return true;
}

/** 全量同步镜像：首帧图 + 参考图·1..9 + 参考视频·1..3（联动拆分视频）+ 参考音频·1..3 + 提示词·1..3。
 *  仅导演台素材/提示词操作时调用，不动手摆工作流（找不到同名节点=手摆，静默跳过）。 */
function syncMirrors(node, ds) {
    if (!node || !ds) return;
    const byKind = (k) => (Array.isArray(ds.ref_assets) ? ds.ref_assets.filter((a) => a.kind === k) : []);
    setMirrorNode("首帧图", ds.mode === "首帧视频" ? (ds.first_frame || "") : "", ["image"]);
    setMirrorNode("尾帧图", ds.last_frame || "", ["image"]);   // 尾帧锚定：任意模式可用
    const multi = ds.mode === "多参视频";
    const imgs = multi ? byKind("image") : [];
    for (let i = 0; i < 9; i++) {
        setMirrorNode(`参考图·${i + 1}`, imgs[i] ? String(imgs[i].file) : "", ["image"]);
    }
    const vids = multi ? byKind("video") : [];
    for (let i = 0; i < 3; i++) {
        const f = vids[i] ? String(vids[i].file) : "";
        setMirrorNode(`参考视频·${i + 1}`, f, ["file"]);
        setMirrorNode(`拆分视频·${i + 1}`, f);   // 与 LoadVideo 同步点亮/隐藏（hidden=断链）
    }
    const auds = multi ? byKind("audio") : [];
    for (let i = 0; i < 3; i++) {
        setMirrorNode(`参考音频·${i + 1}`, auds[i] ? String(auds[i].file) : "", ["audio"]);
    }
    const prompts = Array.isArray(ds.prompts) ? ds.prompts.map((p) => String(p || "")) : [];
    for (let i = 0; i < 3; i++) {
        const t = prompts[i] && prompts[i].trim() ? prompts[i] : "";
        setMirrorNode(`提示词·${i + 1}`, t, ["value"]);
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

/** 文件选择 + 上传：target="first"=首帧图；target="last"=尾帧锚定；kind=image/video/audio=入素材池。
 *  上传统一走 /upload/image（服务端按原样字节写入 input 目录，不限图片）。 */
async function pickAsset(target, kind) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = (target === "first" || target === "last") ? "image/*" : KIND_ACCEPT[kind] || "image/*";
    input.onchange = async () => {
        const f = input.files && input.files[0];
        if (!f) return;
        try {
            const name = await uploadToInput(f);
            if (target === "first") {
                setFirstFrame(node, name);
            } else if (target === "last") {
                setLastFrame(node, name);
            } else {
                addAsset(node, kind, name);
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

/* ---------- 项目动作（游戏式存读档） ---------- */

function doReroll(segNo) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const okReroll = setWidgetValue(node, W_REROLL, segNo);
    setWidgetValue(node, W_SEED, Math.floor(Math.random() * 2 ** 48));
    pendingReset = okReroll;
    queuePrompt();
}

/** 只跑某一段：审片模式（每次运行只生成一段即返回）+ 重跑起始段组合。
 *  segNo ≤ done：重跑起始段=segNo → 重做该段后暂停（其后段落存档被截断，继续时重新生成）；
 *  segNo = done+1：不设重跑 → 顺序生成下一段后暂停（无损）；
 *  segNo > done+1：链式续拍必须按顺序，拦截并说明。
 *  队列提交后立即还原控件（提示词快照已在服务端，用户界面不残留临时值）。 */
async function doRunOnly(segNo, done) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (!Number.isInteger(segNo) || segNo < 1) return;
    if (segNo > done + 1) {
        alert(`链式续拍需按顺序生成：段 ${done + 1} 尚未完成。\n`
            + `请先连续生成到段 ${done + 1}（普通「▶ 继续」每次一段），或对已完成段用「重摇此段」。`);
        return;
    }
    if (segNo <= done
        && !confirm(`重做段 ${segNo}？该段及其后的断点存档会被丢弃，确认满意后继续跑会逐段重新生成。`)) return;
    const prevReview = getWidgetValue(node, "审片模式") ?? "关闭";
    const prevReroll = getWidgetValue(node, W_REROLL) ?? 0;
    if (!setWidgetValue(node, "审片模式", "逐段确认")) {
        alert("节点上没有「审片模式」控件：请重新载入配套工作流");
        return;
    }
    setWidgetValue(node, W_REROLL, segNo <= done ? segNo : 0);
    const ds = getDs(node);
    setDs(node, ds);
    syncModeWidget(node, ds.mode);
    setLed("running", `只跑段 ${segNo} 已提交`);
    try {
        await app.queuePrompt();
    } catch (e) {
        console.warn("[h3-director] queue failed:", e);
    }
    setWidgetValue(node, "审片模式", prevReview);
    setWidgetValue(node, W_REROLL, prevReroll);
    scheduleRefresh();
}

/** 从段 idx+1 之后继续生成到底（游戏读档语义）：
 *  保留第 1..N 段，N 之后有旧存档时先截断（弹确认），然后连续生成到链尾。 */
function continueFromSegment(idx, done) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const n = idx + 1;
    if (n < done
        && !confirm(`从段 ${n + 1} 继续生成？\n段 ${n + 1} 之后的旧存档会被丢弃，从段 ${n + 1} 起逐段重新生成到链尾。`)) return;
    setWidgetValue(node, W_REROLL, n < done ? n + 1 : 0);
    setLed("running", `从段 ${Math.min(n + 1, done + 1)} 续拍已提交`);
    queuePrompt();
}

/** 删除项目 = 删除整个项目文件夹（分段视频/成片/提示词/latent 全删）。 */
async function deleteProject(dir) {
    if (!dir) return;
    if (!confirm(`删除项目「${dir}」？\n\n`
        + `将删除整个文件夹：output/h3_projects/${dir}\n`
        + `（全部分段视频、成片、提示词清单、续拍 latent 一并删除）\n\n不可恢复，继续？`)) return;
    try {
        const r = await api.fetchApi("/h3chain/delete_project", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ dir }),
        });
        if (r.status === 404 || r.status === 405) {
            setApiError(`项目接口未注册（HTTP ${r.status}）：请重启 ComfyUI 并检查控制台是否出现`
                + `「[ComfyUI_H3_SeamlessChain] 路由已注册」日志；若仍失败请把控制台报错反馈给开发。`);
            setLed("error", `删除路由未注册 (HTTP ${r.status})`);
            return;
        }
        if (!r.ok) {
            const j = await r.json().catch(() => ({}));
            alert(`删除失败：${j.error || `HTTP ${r.status}`}`);
            return;
        }
        setLed("idle", "项目已删除");
        scheduleRefresh(300);
    } catch (e) {
        alert(`删除失败：${e}`);
    }
}

/** 读档：切换「存档目录」+ 载入该项目的提示词进导演台状态（后端续跑校验共享参数）。 */
async function switchProject(dir) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点（只读模式）"); return; }
    if (!setDirValue(node, dir)) { alert("节点上没有「存档目录/断点目录」控件"); return; }
    setWidgetValue(node, W_REROLL, 0);
    const r = await apiGet(`/h3chain/project?dir=${encodeURIComponent(dir)}`);
    const mf = r.ok ? (r.data?.manifest || null) : null;
    if (mf) {
        const total = mf.total || (mf.prompts || []).length || 0;
        if (total) {
            const ds = getDs(node);
            ds.prompts = Array.from({ length: total }, (_v, i) => String((mf.prompts || [])[i] ?? ""));
            ds.segments = Array.from({ length: total }, (_v, i) => ds.segments?.[i] ?? defaultSegment());
            setDs(node, ds);
        }
        setLed("idle", `已读档「${mf.title || dir}」（${mf.total ? `${mf.done ?? 0}/${mf.total} 段` : "草稿，未配置段落"}）`);
    } else {
        setLed("idle", `已指向 ${dir}`);
    }
    scheduleRefresh(200);
}

/** 把项目 manifest 的共享参数映射回画布控件（画幅优先尝试 宽高比×MP combo 匹配）。 */
function applyParamsToCanvas(node, params) {
    if (!node || !params) return;
    const applied = [];
    const w = Number(params.width), h = Number(params.height);
    if (w && h) {
        const combo = matchCanvasCombo(w, h);
        if (combo && setWidgetValue(node, W_AR, combo[0]) && setWidgetValue(node, W_MP, combo[1])) {
            applied.push(`画幅 ${combo[0]}·${combo[1]}MP（${w}×${h}）`);
        } else if (setWidgetValue(node, W_WIDTH, w) && setWidgetValue(node, W_HEIGHT, h)) {
            applied.push(`宽×高 ${w}×${h}（自定义）`);
        }
    }
    if (params.length) {
        const sec = +(params.length / 24).toFixed(2);
        if (setWidgetValue(node, W_DUR, sec)) applied.push(`每段时长 ${sec}s`);
    }
    if (params.ctx != null) {
        const v = String(params.ctx);
        const wd = (node.widgets || []).find((x) => x.name === "引导帧数");
        if (!wd || (wd.options?.options || []).includes(v)) {
            if (setWidgetValue(node, "引导帧数", v)) applied.push(`引导帧数 ${v}`);
        }
    }
    if (params.steps && setWidgetValue(node, "步数", params.steps)) applied.push(`步数 ${params.steps}`);
    if (params.cfg != null && setWidgetValue(node, "CFG", params.cfg)) applied.push(`CFG ${params.cfg}`);
    for (const [name, key] of [["采样器", "sampler"], ["调度器", "scheduler"]]) {
        const v = params[key];
        if (!v) continue;
        const wd = (node.widgets || []).find((x) => x.name === name);
        if (wd && !(wd.options?.options || []).includes(v)) continue;
        if (setWidgetValue(node, name, v)) applied.push(`${name} ${v}`);
    }
    scheduleRefresh(300);
    alert(applied.length
        ? `已套用参数到画布：\n\n${applied.join("\n")}`
        : "该项目没有可套用的参数（或控件不匹配）");
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

    /* 诊断 + 项目列表（后端实时扫描磁盘，一个项目一个文件夹） */
    const ping = await apiGet("/h3chain/ping");
    let projects = [];
    if (ping.ok) {
        const r = await apiGet("/h3chain/projects");
        if (r.ok) projects = r.data?.projects || [];
    }
    setApiError(ping.ok ? "" :
        `项目存档接口未注册（HTTP ${ping.status || "??"}）：请重启 ComfyUI 并检查控制台是否出现`
        + `「[ComfyUI_H3_SeamlessChain] 路由已注册」日志；若仍失败请把控制台报错反馈给开发。`);

    /* 当前项目：节点「存档目录」指向优先（用户刚切换还没跑），回落 state 指针 */
    const stateRaw = await fetchJson("h3_projects", "h3chain_state.json");
    const nodeDir = node ? String(getDirValue(node) || "").trim() : "";
    const dir = nodeDir || stateRaw?.dir || "";
    const mf = dir ? await fetchJson(`h3_projects/${dir}`, "manifest.json") : null;
    const state = stateRaw || {};
    const sameChain = !!stateRaw && dir === stateRaw.dir;
    state.dir = dir;
    state.done = mf?.done ?? (sameChain ? stateRaw?.done : 0) ?? 0;
    state.total = mf?.total ?? (sameChain ? stateRaw?.total : 0) ?? 0;

    const prefix = saverPrefix();
    const history = await fetchJson(prefix, "h3saver_history.json");

    let plan = null;
    let drafts = [];
    let ds = defaultDs();
    if (node) {
        ({ plan, drafts, ds } = planFromDs(node));
    }
    if (!plan || !plan.length) {
        const total = state.total ?? (mf && mf.total) ?? 0;
        if (total && node) {
            // 采纳存档提示词进导演台状态：卡片可编辑（此前回退成只读历史段，导致无法输入）
            const dsAdopt = getDs(node);
            dsAdopt.prompts = Array.from({ length: total }, (_v, i) => String(((mf && mf.prompts) || [])[i] ?? ""));
            dsAdopt.segments = Array.from({ length: total }, (_v, i) => dsAdopt.segments?.[i] ?? defaultSegment());
            setDs(node, dsAdopt);
            ({ plan, drafts, ds } = planFromDs(node));
        } else if (total) {
            plan = Array.from({ length: total }, (_v, i) => {
                const ins = ((mf && mf.inserts) || []).find((x) => x.slot === i);
                return ins ? { kind: "insert", pos: i + 1, file: ins.file || "" }
                           : { kind: "prompt", text: ((mf && mf.prompts) || [])[i] || "" };
            });
        } else {
            plan = [];
        }
    }
    return { node, state, mf, plan, drafts, history, prefix, ds, projects, apiOk: ping.ok };
}

function statusLine(state, mf, plan) {
    const total = plan ? plan.length : (mf?.total ?? state?.total ?? 0);
    const done = mf?.done ?? state?.done ?? 0;
    if (!total) return { text: "尚未配置段落：在流水线卡片填写提示词，或点「＋ 添加一段」", next: false };
    if (done >= total) return { text: `本链已全部完成（${total}/${total} 段）✓ 可「＋ 新建项目」开下一条`, next: false };
    const anchored = done > 0 ? `，锚定段 ${done} 尾部` : "";
    return { text: `已载入段 1–${done}（存档回放）→ 本次将生成段 ${done + 1}${anchored}`, next: true };
}

function segMediaHtml(state, mf, idx) {
    const done = mf?.done ?? 0;
    if (idx >= done || !state?.dir) return null;
    const sub = `h3_projects/${state.dir}`;
    const thumbFile = (mf.thumbs || [])[idx];
    const videoFile = (mf.videos || [])[idx];
    const thumbSrc = thumbFile ? viewUrl(sub, thumbFile) : "";
    if (videoFile) {
        return `<video class="h3d-segvideo" controls preload="metadata"${thumbSrc ? ` poster="${thumbSrc}"` : ""} src="${viewUrl(sub, videoFile)}"></video>`;
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
    :root{--h3d-ink:#1c2128;--h3d-panel:#22272e;--h3d-panel2:#2d333b;--h3d-line:#444c56;--h3d-cyan:#6cb6ff;--h3d-copper:#daaa3f;--h3d-bone:#cdd9e1;--h3d-muted:#909dab;--h3d-ok:#57ab5a;--h3d-warn:#e0823d;--h3d-danger:#e5534b}
    .h3d-page *,.h3d-mini *,.h3d-dialog *{box-sizing:border-box}
    @keyframes h3d-blink{50%{opacity:.35}}
    .h3d-led{display:inline-flex;gap:7px;align-items:center;color:var(--h3d-muted);font-size:12px;white-space:nowrap}
    .h3d-led i{width:8px;height:8px;border-radius:50%;background:#636e7b;flex:none}
    .h3d-led.running i{background:var(--h3d-cyan);box-shadow:0 0 10px #6cb6ffcc;animation:h3d-blink 1s infinite}
    .h3d-led.done i{background:var(--h3d-ok);box-shadow:0 0 12px #57ab5a88}
    .h3d-led.error i{background:var(--h3d-warn);box-shadow:0 0 12px #e0823d88}
    .h3d-btn{cursor:pointer;border:1px solid #444c56;border-radius:6px;background:#2d333b;color:var(--h3d-bone);padding:6px 11px;font-size:12px;font-family:inherit;transition:border-color .12s,filter .12s}
    .h3d-btn:hover{filter:brightness(1.18)}
    .h3d-btn:disabled{opacity:.5;cursor:not-allowed;filter:none}
    .h3d-btn-cyan{border-color:#316dca;background:#1f2f45;color:#9ecbff}
    .h3d-btn-danger{border-color:#9a4144;background:#3a2225;color:#f0a0a4}
    .h3d-btn-cta{border:0;background:linear-gradient(135deg,#f0c274,#e6b566 55%,#d99e4a);color:#1a1408;font-weight:700;box-shadow:0 2px 14px #e6b56633}
    .h3d-btn-cta:hover{filter:brightness(1.08);box-shadow:0 3px 18px #e6b56644}
    .h3d-chip{font-size:10.5px;padding:2px 8px;border-radius:10px;border:1px solid #444c56;background:#2d333b;color:#adbac7;white-space:nowrap}
    .h3d-chip.ok{border-color:#2ea04366;background:#12261e;color:#7ee2a8}
    .h3d-chip.media{border-color:#7a5f36;background:#352a19;color:#e9c07a}
    .h3d-chip.cyan{border-color:#316dca;background:#1f2f45;color:#9ecbff}
    .h3d-chip.warn{border-color:#9a4144;background:#402227;color:#f0a0a4}

    /* ---- 侧栏迷你入口卡 ---- */
    .h3d-mini{display:flex;flex-direction:column;gap:9px;padding:10px;font-size:12px;color:var(--h3d-bone);background:linear-gradient(150deg,#22272e,#2d333b);border:1px solid #444c56;border-radius:10px}
    .h3d-mini-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
    .h3d-mini-brand{font-weight:700;letter-spacing:.04em;min-width:0}
    .h3d-mini-brand small{display:block;margin-top:3px;color:var(--h3d-cyan);font-weight:500;font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-rail{display:flex;gap:3px}
    .h3d-mini-rail span{flex:1;height:4px;border-radius:4px;background:#444c56}
    .h3d-mini-rail span.done{background:var(--h3d-cyan)}
    .h3d-mini-rail span.next{background:#6e7b8c;animation:h3d-blink 1.2s infinite}
    .h3d-mini-cards{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}
    .h3d-mini-card{min-height:62px;padding:9px;border:1px solid #3f4854;border-radius:8px;background:#1b2027;font-size:11px;color:var(--h3d-muted);line-height:1.65;overflow:hidden}
    .h3d-mini-count{display:grid;place-items:center;padding:9px 10px;min-width:66px;border:1px solid #3f4854;border-radius:8px;background:#1b2027;font:700 20px/1.15 ui-monospace,Consolas;color:var(--h3d-cyan);text-align:center}
    .h3d-mini-count small{font-size:10px;color:var(--h3d-muted);font-weight:400}
    .h3d-mini-foot{display:flex;flex-direction:column;gap:8px}
    .h3d-mini-params{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-open{width:100%;padding:9px}

    /* ---- 全屏导演台 ---- */
    .h3d-page{position:fixed;inset:0;z-index:1000000;background:var(--h3d-ink);color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif;display:grid;grid-template-rows:58px 1fr 58px}
    .h3d-topbar{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--h3d-line);background:linear-gradient(90deg,#22272e,#262c36)}
    .h3d-top-left{min-width:0;display:flex;gap:14px;align-items:center}
    .h3d-kicker{color:var(--h3d-copper);font:700 11px/1 ui-monospace,Consolas;letter-spacing:.18em;white-space:nowrap}
    .h3d-title{font-size:17px;font-weight:700;white-space:nowrap}
    .h3d-sub{min-width:0;color:var(--h3d-muted);font-family:ui-monospace,Consolas;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-top-right{display:flex;gap:14px;align-items:center;justify-self:end}
    .h3d-close{width:36px;height:36px;border:1px solid var(--h3d-line);border-radius:7px;background:#2d333b;color:var(--h3d-bone);cursor:pointer;font-size:15px}
    .h3d-close:hover{filter:brightness(1.2)}
    .h3d-banner{display:none;padding:9px 18px;border-bottom:1px solid #8a4a3f;background:#3a2620;color:#ffc9b8;font-size:12px;line-height:1.6}
    .h3d-stage{min-height:0;display:grid;grid-template-columns:minmax(255px,300px) minmax(400px,1fr) minmax(255px,300px);gap:1px;background:var(--h3d-line)}
    .h3d-col{min-width:0;min-height:0;background:var(--h3d-panel);overflow:auto}
    .h3d-sechead{position:sticky;top:0;z-index:3;padding:14px 16px 10px;background:#22272eee;backdrop-filter:blur(8px);border-bottom:1px solid #3f4854}
    .h3d-sechead strong{display:block}
    .h3d-sechead small{color:var(--h3d-muted)}

    .h3d-projlist{display:grid;gap:7px;padding:12px}
    .h3d-projrow{display:flex;gap:6px;align-items:stretch}
    .h3d-projrow .h3d-proj{flex:1;min-width:0}
    .h3d-proj-del{flex:none;width:30px;border:1px solid #4c3a3d;border-radius:8px;background:#2a2225;color:#b08a8e;cursor:pointer;font-size:13px;line-height:1;transition:.15s}
    .h3d-proj-del:hover{border-color:#c2565f;background:#3a2429;color:#ffb3b8}
    .h3d-proj{display:grid;grid-template-columns:44px 1fr;grid-template-rows:auto auto;gap:2px 10px;padding:8px 10px 9px 13px;border:1px solid #3f4854;border-radius:8px;background:#262c36;cursor:pointer;box-shadow:inset 3px 0 0 #444c56;text-align:left;font-family:inherit;color:inherit}
    .h3d-proj:hover{background:#2a313b}
    .h3d-proj.active{box-shadow:inset 3px 0 0 var(--h3d-cyan);border-color:#316dca}
    .h3d-proj-cover{grid-row:1/3;width:44px;height:33px;object-fit:cover;border-radius:5px;border:1px solid #3f4854;background:#1c2128}
    .h3d-proj.nocover{grid-template-columns:1fr;padding-left:13px}
    .h3d-proj-name{font-weight:700;word-break:break-all;font-size:12.5px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;color:var(--h3d-bone)}
    .h3d-proj-meta{color:var(--h3d-muted);font-size:11px;font-family:ui-monospace,Consolas}
    .h3d-newrow{display:flex;gap:8px;padding:0 12px 12px;flex-wrap:wrap}
    .h3d-meter{margin:0 12px 12px;padding:11px;border:1px solid #444c56;border-radius:8px;background:#262c36}
    .h3d-meter strong{display:block;color:var(--h3d-cyan);font:700 16px/1.3 ui-monospace,Consolas}
    .h3d-meter p{margin:4px 0 0;color:var(--h3d-muted);font-size:11px}
    .h3d-drafts{margin:0 12px 12px;color:var(--h3d-muted);font-size:11px;line-height:1.7}

    .h3d-center-pad{padding:14px 18px 20px}
    .h3d-statusbar{display:flex;align-items:center;gap:10px;padding:10px 13px;border:1px solid #316dca;border-radius:8px;background:linear-gradient(90deg,#1f2f45,#22272e);line-height:1.6}
    .h3d-statusbar .h3d-st-text{flex:1;min-width:0}
    .h3d-statusbar .h3d-chip{align-self:center}
    .h3d-refresh{padding:4px 10px;flex:none}
    .h3d-rail{display:flex;gap:4px;margin:12px 0 14px}
    .h3d-rail span{flex:1;height:5px;border-radius:4px;background:#444c56}
    .h3d-rail span.done{background:var(--h3d-cyan)}
    .h3d-rail span.next{background:#6e7b8c;animation:h3d-blink 1.2s infinite}
    .h3d-cards{display:grid;gap:10px}
    .h3d-card{display:grid;grid-template-columns:150px minmax(0,1fr);gap:11px;padding:10px;border:1px solid #3f4854;border-radius:9px;background:#262c36}
    .h3d-card.todo{opacity:.72}
    .h3d-card.todo:hover{opacity:1}
    .h3d-thumb{width:150px;aspect-ratio:16/9;border-radius:6px;overflow:hidden;background:#10151c;display:grid;place-items:center;color:#636e7b;font:700 11px ui-monospace,Consolas}
    .h3d-thumb video,.h3d-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-thumb video.h3d-segvideo{object-fit:contain;background:#000}
    .h3d-cbody{min-width:0;display:flex;flex-direction:column;gap:5px}
    .h3d-ctitle{display:flex;gap:7px;align-items:center;flex-wrap:wrap;font-weight:700}
    .h3d-cmeta{color:var(--h3d-muted);font:11px ui-monospace,Consolas;word-break:break-all}
    .h3d-cprompt{color:#b6c2cf;font-size:12px;line-height:1.65;word-break:break-word}
    .h3d-ta{width:100%;min-height:84px;resize:vertical;border:1px solid #444c56;border-radius:6px;background:#2d333b;color:var(--h3d-bone);padding:8px 9px;font:12.5px/1.65 "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none;box-sizing:border-box}
    .h3d-ta:focus{border-color:#6cb6ff;box-shadow:0 0 0 2px #6cb6ff33}
    .h3d-ta:disabled{color:#636e7b;cursor:not-allowed;background:#22272e}
    .h3d-ta-row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:4px}
    .h3d-ta-hint{color:var(--h3d-muted);font-size:10.5px}
    .h3d-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:2px}
    .h3d-hint{color:var(--h3d-muted);font-size:11px;align-self:center}
    .h3d-editor{display:grid;gap:7px;margin-top:4px;padding:10px;border:1px solid #444c56;border-radius:8px;background:#20262e}
    .h3d-editor textarea{width:100%;min-height:96px;resize:vertical;border:1px solid #444c56;border-radius:6px;background:#2d333b;color:var(--h3d-bone);padding:9px;font:13px/1.7 "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none}
    .h3d-editor textarea:focus{border-color:#6cb6ff;box-shadow:0 0 0 2px #6cb6ff33}
    .h3d-editor-row{display:flex;gap:7px;flex-wrap:wrap}
    .h3d-addrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .h3d-drawer{margin-top:14px;border:1px solid #3f4854;border-radius:8px;background:#262c36}
    .h3d-drawer summary{padding:10px 12px;cursor:pointer;color:var(--h3d-cyan)}
    .h3d-drawer pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;font-size:11.5px;max-height:300px;overflow:auto;color:#b6c2cf}

    .h3d-hist{padding:12px;display:grid;gap:10px;align-content:start}
    .h3d-hist-head{color:var(--h3d-copper);font:700 10.5px/1 ui-monospace,Consolas;letter-spacing:.08em;margin:2px 0 -2px;word-break:break-all}
    .h3d-empty{padding:16px 10px;border:1px dashed #444c56;border-radius:8px;color:var(--h3d-muted);text-align:center;background:#262c36;line-height:1.8}
    .h3d-result{padding:8px;border:1px solid #3f4854;border-radius:9px;background:#20262e}
    .h3d-result.current{border-color:#316dca}
    .h3d-result video{display:block;width:100%;max-height:230px;border-radius:6px;background:#0a0d12}
    .h3d-result-meta{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:7px}
    .h3d-result-name{min-width:0;color:#b6c2cf;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-result-info{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-result-acts{display:flex;gap:6px;flex:none}
    .h3d-dl{padding:4px 8px;border:1px solid #316dca;border-radius:6px;color:#9ecbff;text-decoration:none;background:#1f2f45;font-size:11px}
    .h3d-dl:hover{filter:brightness(1.15)}
    .h3d-foot{padding:2px 16px 14px;color:var(--h3d-muted);font-size:10.5px;word-break:break-all;line-height:1.8}

    /* ---- 分段处理中心面板 ---- */
    .h3d-seg-panel{margin-top:6px;border:1px solid #37332b;border-radius:7px;background:#181712;overflow:hidden}
    .h3d-seg-panel summary{padding:7px 10px;cursor:pointer;color:var(--h3d-cyan);font-size:11.5px;font-weight:600;user-select:none;display:flex;align-items:center;gap:6px;flex-wrap:wrap}
    .h3d-seg-panel summary::-webkit-details-marker{display:none}
    .h3d-seg-panel summary::before{content:"▸";font-size:10px;transition:transform .15s}
    .h3d-seg-panel[open] summary::before{transform:rotate(90deg)}
    .h3d-seg-panel.has-content{border-color:#46604f}
    .h3d-seg-body{display:grid;gap:7px;padding:0 10px 10px}
    .h3d-seg-label{display:block;margin-bottom:2px;color:var(--h3d-muted);font-size:10.5px;font-weight:600}
    .h3d-seg-ta{min-height:48px !important;font-size:12px !important;line-height:1.55 !important}

    /* ---- 每段时长 + 段级引用素材 ---- */
    .h3d-secs{width:58px;border:1px solid #3a352c;border-radius:5px;background:#211f1a;color:var(--h3d-bone);padding:2px 4px;font:11px ui-monospace,Consolas;text-align:right;outline:none}
    .h3d-secs:focus{border-color:#a8d8bd}
    .h3d-secs-hint{color:var(--h3d-muted);font:10px ui-monospace,Consolas;white-space:nowrap}
    .h3d-refrow{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin-top:4px;padding:7px 8px;border:1px solid #37332b;border-radius:7px;background:#181712}
    .h3d-refrow>label{color:var(--h3d-muted);font-size:10.5px;font-weight:600;flex:none}
    .h3d-refchip{display:inline-flex;gap:5px;align-items:center;padding:3px 8px 3px 3px;border:1px solid #3a352c;border-radius:12px;background:#25221c;color:#a8a294;cursor:pointer;font-size:11px;font-family:inherit}
    .h3d-refchip:hover{border-color:#46604f}
    .h3d-refchip.on{border-color:#2f6e57;background:#12291f;color:#7fe0b0}
    .h3d-refchip img{width:20px;height:20px;border-radius:9px;object-fit:cover;background:#0d0c0a}
    .h3d-reftpl{max-width:118px;border:1px solid #3a352c;border-radius:6px;background:#211f1a;color:var(--h3d-bone);padding:3px 4px;font-size:11px;outline:none;margin-left:auto}
    .h3d-reftpl:focus{border-color:#a8d8bd}

    /* ---- 素材标签编辑 ---- */
    .h3d-labelinp{width:100%;border:1px solid #3a352c;border-radius:5px;background:#211f1a;color:var(--h3d-bone);padding:4px 6px;font:600 12px "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none}
    .h3d-labelinp:focus{border-color:#a8d8bd;box-shadow:0 0 0 2px #8cc9a833}
    .h3d-quicklbl{display:flex;gap:4px;flex-wrap:wrap;margin-top:5px}
    .h3d-quicklbl button{padding:2px 7px;border:1px solid #3a352c;border-radius:10px;background:#25221c;color:#a39d90;cursor:pointer;font-size:10px;font-family:inherit}
    .h3d-quicklbl button:hover{border-color:#46604f;color:#d9d4c9}
    .h3d-asset-usage{margin-top:5px;display:flex;gap:4px;flex-wrap:wrap}

    /* ---- 链参数：换算徽章 + 高级设置折叠 ---- */
    .h3d-convbadge{grid-column:1/-1;margin:-4px 0 0;padding:8px 10px;border:1px dashed #46604f;border-radius:7px;background:#1f2a23;color:#c2e0cd;font:11.5px ui-monospace,Consolas;word-break:break-all}
    .h3d-adv{grid-column:1/-1;border:1px solid #37332b;border-radius:8px;background:#181712;overflow:hidden}
    .h3d-adv summary{padding:8px 10px;cursor:pointer;color:var(--h3d-muted);font-size:11.5px;font-weight:600;user-select:none}
    .h3d-adv summary:hover{color:#d9d4c9}
    .h3d-adv summary::-webkit-details-marker{display:none}
    .h3d-adv summary::before{content:"⚙ ";}
    .h3d-adv[open] summary{border-bottom:1px solid #302c25;color:var(--h3d-cyan)}
    .h3d-adv-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:10px}
    .h3d-adv .h3d-param{margin:0}
    .h3d-param{margin:0}
    .h3d-param .h3d-hint{display:block;margin-top:4px}

    .h3d-loadwf{display:flex;flex-direction:column;gap:10px;align-items:center;padding:22px 14px;border:1px dashed #46604f;border-radius:10px;background:#1f2a23;text-align:center;line-height:1.9}
    .h3d-loadwf p{margin:0;color:var(--h3d-muted);font-size:11.5px}

    .h3d-footer{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 20px;border-top:1px solid var(--h3d-line);background:#1b1a16}
    .h3d-footinfo{display:flex;gap:16px;color:var(--h3d-muted);flex-wrap:wrap;min-width:0;font-size:11.5px;align-items:center}
    .h3d-footinfo b{color:var(--h3d-bone)}
    .h3d-run{min-width:150px;padding:11px 18px}

    /* ---- 素材与参考 / 链参数 ---- */
    .h3d-lsec,.h3d-rsec{display:flex;flex-direction:column;min-height:0}
    .h3d-col.left .h3d-sechead,.h3d-col.right .h3d-sechead{position:static;backdrop-filter:none}
    .h3d-modebar{display:flex;gap:6px;padding:10px 12px 6px;border-bottom:1px solid var(--h3d-line);background:#171612}
    .h3d-mode{flex:1;min-width:0;padding:8px 4px;border:1px solid #332f27;border-radius:7px;background:#1b1a16;color:#a8a294;cursor:pointer;font:700 12px "Microsoft YaHei UI","Segoe UI",sans-serif;transition:all .12s}
    .h3d-mode:hover{border-color:#46604f;color:#d9d4c9}
    .h3d-mode.active{color:#12241c;background:var(--h3d-cyan);border-color:var(--h3d-cyan);box-shadow:0 0 0 1px var(--h3d-cyan) inset,0 0 14px #8cc9a844}
    .h3d-mode small{display:block;font-weight:400;font-size:9.5px;opacity:.72;margin-top:2px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-mode.active small{opacity:.85}
    .h3d-assets{display:grid;gap:8px;padding:12px}
    .h3d-asset{display:grid;grid-template-columns:64px minmax(0,1fr) auto;gap:9px;align-items:center;padding:8px;border:1px solid #332f27;border-radius:8px;background:#1b1a16;box-shadow:inset 3px 0 0 #3a352c}
    .h3d-asset.on{box-shadow:inset 3px 0 0 var(--h3d-cyan)}
    .h3d-asset-thumb{width:64px;height:52px;border-radius:5px;background:#262319;display:grid;place-items:center;overflow:hidden;color:#77705f;font:700 10px ui-monospace,Consolas}
    .h3d-asset-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-asset-copy{min-width:0}
    .h3d-asset-copy strong{display:block;font-size:12px}
    .h3d-asset-copy small{display:block;margin-top:3px;color:var(--h3d-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-asset-acts{display:flex;gap:5px}
    .h3d-asset-acts .h3d-btn{padding:4px 8px;font-size:11px}
    .h3d-addasset{justify-self:start;padding:6px 12px}
    .h3d-upload-row{display:flex;gap:7px;flex-wrap:wrap}
    .h3d-upload-row .h3d-btn{flex:1;min-width:84px}
    .h3d-upload-row .h3d-btn:disabled{opacity:.38;cursor:not-allowed}
    .h3d-kindmark{font-size:12px;line-height:1}
    .h3d-kindmark.big{font-size:22px}
    .h3d-params{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px}
    .h3d-param label{display:block;margin-bottom:4px;color:var(--h3d-muted);font-size:11px}
    .h3d-select,.h3d-seedrow input{width:100%;border:1px solid #3a352c;border-radius:6px;background:#211f1a;color:var(--h3d-bone);padding:6px 7px;font-size:12px;outline:none;font-family:inherit}
    .h3d-select:focus,.h3d-seedrow input:focus{border-color:#a8d8bd}
    .h3d-seedrow{display:flex;gap:5px}
    .h3d-seedrow input{flex:1;min-width:0}
    .h3d-seedrow input::-webkit-outer-spin-button,.h3d-seedrow input::-webkit-inner-spin-button{-webkit-appearance:none;margin:0}
    .h3d-seedrow input[type=number]{-moz-appearance:textfield;appearance:textfield}
    .h3d-seedrow .h3d-btn{padding:4px 8px;flex:none}
    .h3d-psec .h3d-foot,.h3d-asec .h3d-foot,.h3d-projsec .h3d-foot{padding:0 16px 14px}

    /* ---- 新建项目模态 ---- */
    .h3d-overlay{position:fixed;z-index:1000003;inset:0;display:grid;place-items:center;padding:24px;background:#0a0906d0;backdrop-filter:blur(10px)}
    .h3d-dialog{width:min(470px,calc(100vw - 40px));border:1px solid #464033;border-radius:14px;background:linear-gradient(160deg,#242019,#191712 62%);box-shadow:0 22px 64px #000b;padding:20px;color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif}
    .h3d-dialog h3{margin:0 0 6px;font-size:16px}
    .h3d-dialog .h3d-lead{color:var(--h3d-muted);margin:0 0 14px;line-height:1.75;font-size:12px}
    .h3d-dialog input[type=text]{width:100%;border:1px solid #3a352c;border-radius:6px;background:#211f1a;color:var(--h3d-bone);padding:9px 10px;font:13px ui-monospace,Consolas;outline:none;margin-bottom:6px}
    .h3d-dialog input[type=text]:focus{border-color:#a8d8bd}
    .h3d-err{color:#e89090;font-size:11px;min-height:16px;margin-bottom:6px}
    .h3d-check{display:flex;gap:8px;align-items:center;color:var(--h3d-muted);font-size:12px;margin-bottom:14px;cursor:pointer}
    .h3d-dialog-row{display:flex;gap:8px;justify-content:flex-end}

    .h3d-fab{position:fixed;right:16px;top:120px;z-index:80;width:44px;height:44px;border-radius:50%;border:1px solid #46604f;background:#1f2a23;color:#c2e0cd;cursor:pointer;font-size:17px}
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
    const done = mf?.done ?? state?.done ?? 0;

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
    const count = el("div", "h3d-mini-count", total
        ? `${done}<small>/${total} 段</small>`
        : `<small>0 段<br>待添加</small>`);
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

    /* 诊断横幅：项目存档接口未注册时显示（/h3chain/ping 探测失败） */
    const banner = el("div", "h3d-banner");
    banner.textContent = apiErrorText;
    banner.style.display = apiErrorText ? "" : "none";

    /* 主舞台三栏（侧栏各含两个固定分区，渲染器只填充分区内容） */
    const stage = el("div", "h3d-stage");
    const colL = el("aside", "h3d-col left");
    const lProj = el("section", "h3d-lsec h3d-projsec");
    const lAssets = el("section", "h3d-lsec h3d-asec");
    colL.append(lProj, lAssets);
    const colC = el("section", "h3d-col center");
    colC.append(el("div", "h3d-sechead",
        "<strong>段落流水线</strong><small>卡片 = 生成顺序（1–64 段不限，「＋ 添加一段」即可加段；「提示词·1..3」只是画布镜像）；✏ 改词 · 🎲 重摇 · 📎 插视频 · 🎬 分段处理（场景/角色）</small>"));
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

    page.append(topbar, banner, stage, footer);
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
    const { node, state, mf, plan, drafts, history, prefix } = data;
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
    const histSig = prefix + "|" + (history ? history.length + ":" + (history[0]?.file ?? "") : "-")
        + "|" + (state?.dir ?? "") + "|" + (mf?.finals || []).join(",");
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
    const { node, state, mf } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>项目存档</strong><small>一个项目一个文件夹 · 点击读档继续拍</small>"));

    const list = el("div", "h3d-projlist");
    const projects = mergeProjects(data.projects, state);
    if (!data.apiOk) {
        list.append(el("div", "h3d-empty",
            "项目接口未注册：列表暂不可用（生成不受影响）。请重启 ComfyUI 后刷新浏览器。"));
    } else if (!projects.length) {
        list.append(el("div", "h3d-empty",
            "暂无项目存档：点下方「＋ 新建项目」立即在 output/h3_projects/ 建好文件夹；"
            + "开自动存档/审片跑一次也会自动生成项目。"));
    }
    for (const p of projects) {
        const active = state?.dir && p.dir === state.dir;
        const row = el("div", "h3d-projrow");
        const coverUrl = p.cover ? viewUrl(`h3_projects/${p.dir}`, p.cover) : "";
        const btn = el("button", "h3d-proj" + (active ? " active" : "") + (coverUrl ? "" : " nocover"));
        btn.innerHTML = `
            <span class="h3d-proj-name">${escapeHtml(p.title || p.dir)}
                ${p.finals && p.finals.length ? badge(`成片×${p.finals.length}`, "media") : ""}
                ${!p.total ? badge("草稿", "") : ""}
            </span>
            <span class="h3d-proj-meta">${p.total
                ? `${p.done ?? 0}/${p.total} 段`
                : "未配置段落"}${p.updated_at ? " · " + escapeHtml(fmtTime(p.updated_at)) : ""}</span>`;
        if (coverUrl) {
            const img = document.createElement("img");
            img.className = "h3d-proj-cover";
            img.loading = "lazy";
            img.src = coverUrl;
            img.alt = "";
            img.onerror = () => img.remove();
            btn.prepend(img);
        }
        btn.title = "点击读档：切换「存档目录」并载入该项目提示词（共享参数需与其一致，否则后端会提示换链）";
        btn.onclick = () => switchProject(p.dir);
        row.append(btn);
        const del = el("button", "h3d-proj-del", "🗑");
        del.type = "button";
        del.title = `删除项目（整个文件夹 output/h3_projects/${p.dir}，视频与提示词全删，不可恢复）`;
        del.onclick = (e) => { e.stopPropagation(); deleteProject(p.dir); };
        row.append(del);
        list.append(row);
    }
    sec.append(list);

    const newrow = el("div", "h3d-newrow");
    const newBtn = el("button", "h3d-btn h3d-btn-cyan", "＋ 新建项目");
    newBtn.title = "新开一条视频链：换存档目录名，提示词沿用当前内容作底稿";
    newBtn.onclick = openNewProjectModal;
    newrow.append(newBtn);
    if (node && mf?.params && Object.keys(mf.params).length) {
        const ap = el("button", "h3d-btn", "⚙ 套用参数到画布");
        ap.title = "把当前项目的共享参数（画幅/每段时长/引导帧数/步数/采样器等）写回节点控件";
        ap.onclick = () => applyParamsToCanvas(node, mf.params);
        newrow.append(ap);
    }
    sec.append(newrow);

    if (state?.dir) {
        const done = mf?.done ?? state.done ?? 0;
        const total = state.total ?? 0;
        const ps = paramsSummary(node, data.mf);
        const totalSec = data.plan?.length
            ? `共${chainSeconds(node, data.ds, data.plan).toFixed(1)}s` : "";
        const meter = el("div", "h3d-meter");
        meter.innerHTML = `<strong>${total ? `${done}/${total} 段` : "未配置段落（提示词每行一段）"}${state.review ? " · 审片中" : ""}</strong>
            <p>${escapeHtml([ps.geo, ps.len, ps.ctx, totalSec].filter(Boolean).join(" · ") || "参数待首次运行后显示")}</p>`;
        sec.append(meter);
        sec.append(el("div", "h3d-foot", `项目文件夹：output/h3_projects/${escapeHtml(state.dir)}`));
    }
}

function mergeProjects(projects, state) {
    const map = new Map();
    for (const p of projects || []) {
        if (p?.dir) map.set(p.dir, p);
    }
    if (state?.dir && !map.has(state.dir)) {
        map.set(state.dir, { dir: state.dir, done: state.done, total: state.total, updated_at: state.updated_at });
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
    const done = mf?.done ?? state?.done ?? 0;

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
        addSeg.title = "新增一段提示词到导演台状态（最多 64 段；「提示词·1..3」画布镜像只显前 3 段，超过 3 段全在导演台管理）";
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
            ta.placeholder = `第 ${idx + 1} 段画面与动作时间线：顺着上一段结尾继续；`
                + `对白写「…」自动转官方 <d>[中文] 格式，说话人标 (S1)；`
                + `运镜可写 The camera pushes in with small amplitude at slow speed；${poolHint}`;
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

            /* 引用素材：勾选本段用哪些（图/视/音混排，后端按类别独立编号压实 token） */
            if (node && it.idx !== undefined && data.ds.mode === "多参视频" && pool.length) {
                const seg = data.ds.segments[it.idx] || defaultSegment();
                const row = el("div", "h3d-refrow");
                row.append(el("label", "", "引用素材"));
                const pickedByKind = { image: 0, video: 0, audio: 0 };
                pool.forEach((a) => {
                    const k = a.kind || "image";
                    const on = (seg.refs || []).includes(a.label);
                    if (on) pickedByKind[k] += 1;
                    const chip = el("button", "h3d-refchip" + (on ? " on" : ""));
                    chip.type = "button";
                    chip.title = `${KIND_NAME[k]}素材：勾选后本段引用它（同类按勾选顺序编号 <${KIND_TOKEN[k]} k>）；`
                        + `提示词写 [[${a.label}]] 引用；单段上限 图${KIND_CAPS.image}/视${KIND_CAPS.video}/音${KIND_CAPS.audio}`;
                    if (k === "image") {
                        const im = document.createElement("img");
                        im.loading = "lazy";
                        im.src = inputViewUrl(a.file);
                        im.onerror = () => im.remove();
                        chip.append(im);
                    } else {
                        chip.insertAdjacentHTML("afterbegin", `<span class="h3d-kindmark">${KIND_ICON[k]}</span>`);
                    }
                    chip.append(document.createTextNode(a.label));
                    chip.onclick = () => { toggleSegmentRef(node, it.idx, a.label); scheduleRefresh(80); };
                    row.append(chip);
                });
                if ((seg.refs || []).length) {
                    row.insertAdjacentHTML("beforeend",
                        `<span class="h3d-secs-hint">图${pickedByKind.image}/${KIND_CAPS.image}`
                        + ` 视${pickedByKind.video}/${KIND_CAPS.video} 音${pickedByKind.audio}/${KIND_CAPS.audio}</span>`);
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
                const hasSeg = !!(seg.scene_prompt || seg.character_prompt || seg.soundscape || seg.music);
                const det = el("details", "h3d-seg-panel" + (hasSeg ? " has-content" : ""));
                det.open = hasSeg;
                det.innerHTML = `<summary>分段处理 · 场景 / 角色 / 声音${hasSeg ? ' <span class="h3d-chip cyan">已填写</span>' : ""}</summary>`;
                const segBody = el("div", "h3d-seg-body");

                const segFields = [
                    ["scene_prompt", "场景提示词",
                        "本段场景（风格+环境+光照），如：实景电影感，雨夜的东京小巷，霓虹灯反射在湿柏油路上。留空省略"],
                    ["character_prompt", "角色提示词",
                        "本段角色（外观+服饰+位置），如：米色风衣短发女主，坐在便利店门口台阶上。留空省略"],
                    ["soundscape", "环境音",
                        "环境音+动作音（1-4句，官方 overall_soundscape），如：雨点敲打柏油路，远处车流，伞面撑开的轻响。留空=不约束"],
                    ["music", "配乐",
                        "背景配乐（乐器+节奏+动态，角色听不到的；官方 non_diegetic_music），如：慢速钢琴单音，弦乐渐强后淡出。留空=不约束"],
                ];
                for (const [field, label, ph] of segFields) {
                    const lab = el("label", "h3d-seg-label", label);
                    const ta = document.createElement("textarea");
                    ta.className = "h3d-ta h3d-seg-ta";
                    ta.rows = 2;
                    ta.value = seg[field] || "";
                    ta.placeholder = ph;
                    ta.addEventListener("input", () => debounceSegmentWrite(node, it.idx, field, ta.value));
                    ta.addEventListener("blur", () => {
                        const key = `s${it.idx}_${field}`;
                        const t = _taTimers.get(key);
                        if (t) { clearTimeout(t); _taTimers.delete(key); }
                        setSegmentField(node, it.idx, field, ta.value);
                        scheduleRefresh(200);
                    });
                    segBody.append(lab, ta);
                }

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
                const cont = el("button", "h3d-btn h3d-btn-cta", "▶ 从这段继续");
                cont.title = `保留段 1–${idx + 1}，从段 ${idx + 2} 起连续生成到链尾`
                    + (idx + 1 < done ? `（段 ${idx + 2} 之后的旧存档会被丢弃）` : "");
                cont.onclick = () => continueFromSegment(idx, done);
                actions.append(cont);
            }
            const runOne = el("button", "h3d-btn h3d-btn-cyan",
                isDone ? "▶ 重跑这段" : "▶ 只跑这段");
            runOne.title = isDone
                ? `重做段 ${idx + 1} 后暂停（审片模式+重跑起始段=${idx + 1}），满意再继续`
                : `只生成段 ${idx + 1} 后暂停（审片模式）：预览单段效果，满意再继续整链`;
            runOne.onclick = () => doRunOnly(idx + 1, done);
            actions.append(runOne);
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

/** 多参模式：带标签编辑的素材卡（重命名同步所有段级引用；图/视/音三类混排） */
function labeledAssetCard(node, ds, idx) {
    const a = ds.ref_assets[idx];
    const file = a.file;
    const kind = a.kind || "image";
    const card = el("div", "h3d-asset on");
    const thumb = el("div", "h3d-asset-thumb");
    if (kind === "image") {
        const im = document.createElement("img");
        im.loading = "lazy";
        im.src = inputViewUrl(file);
        im.alt = a.label;
        im.onerror = () => { thumb.replaceChildren(el("span", "", "⚠")); };
        thumb.append(im);
    } else {
        thumb.innerHTML = `<span class="h3d-kindmark big">${KIND_ICON[kind]}</span>`;
        thumb.title = `${KIND_NAME[kind]}素材：${file}`;
    }

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
    const sameKind = ds.ref_assets.filter((x) => (x.kind || "image") === kind).length;
    usage.innerHTML = `<span class="h3d-chip">${KIND_ICON[kind]}${KIND_NAME[kind]}·${sameKind}/${KIND_CAPS[kind]}</span>`
        + (usedBy ? `<span class="h3d-chip ok">${usedBy} 段指定</span>` : "")
        + (defaultAll && (ds.segments || []).length ? '<span class="h3d-chip cyan">默认全段引用</span>' : "");
    copy.append(labelInp, quick, usage,
        el("small", "", `${escapeHtml(file)} · 提示词写 [[${escapeHtml(a.label)}]] → &lt;${KIND_TOKEN[kind]} k&gt;`));

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
        up.onclick = () => pickAsset("first");
        acts.insertBefore(up, acts.firstChild);
        if (!card.querySelector(".h3d-asset-acts")) card.append(acts);
        box.append(card);
    } else {
        /* 多参：标签素材池（图/视/音三类，状态驱动 + 画布镜像） */
        ds.ref_assets.forEach((_a, i) => {
            box.append(labeledAssetCard(node, ds, i));
        });
        const upRow = el("div", "h3d-upload-row");
        for (const k of KIND_LIST) {
            const sameKind = ds.ref_assets.filter((a) => (a.kind || "image") === k).length;
            const full = sameKind >= KIND_CAPS[k];
            const b = el("button", "h3d-btn h3d-addasset", `＋ ${KIND_NAME[k]}`);
            b.disabled = full;
            b.title = full
                ? `${KIND_NAME[k]}素材已达上限 ${KIND_CAPS[k]} 个（官方单段上限）`
                : `上传${KIND_NAME[k]} → 存入标签素材池（提示词写 [[标签]] → &lt;${KIND_TOKEN[k]} k&gt;；`
                    + `配套工作流对应节点自动点亮，手摆工作流不受影响）`;
            b.onclick = () => pickAsset("pool", k);
            upRow.append(b);
        }
        box.append(upRow);
    }

    /* 尾帧锚定（身份锚点，任意模式可用）：每段末尾注入参考帧，与段首引导桥形成双锚约束防漂移 */
    {
        const lf = ds.last_frame;
        const card = assetCard("尾帧锚定", lf, lf ? () => { removeLastFrame(node); } : null);
        card.querySelector("small").textContent = lf || "未设置（可选，建议角色正面清晰帧）";
        const acts = card.querySelector(".h3d-asset-acts") || el("div", "h3d-asset-acts");
        const up = el("button", "h3d-btn h3d-btn-cyan", lf ? "替换" : "上传");
        up.title = "尾帧锚定（任意模式可用）：注入每段末尾作为人物/场景参考，"
            + "与段首引导桥形成「隧道」双锚约束，防止过了桥窗口后主体漂移。画布「尾帧图」节点同步点亮";
        up.onclick = () => pickAsset("last");
        acts.insertBefore(up, acts.firstChild);
        if (!card.querySelector(".h3d-asset-acts")) card.append(acts);
        box.append(card);
    }
    sec.append(box);
    sec.append(el("div", "h3d-foot",
        "标签素材池（图 ≤9 / 视 ≤3 / 音 ≤3，官方单段上限）：每类素材打标签（角色1 / 场景1…），"
        + "段落卡片里勾选本段引用哪些——未勾选的素材不进该段（后端按段压实重编号），提示词写 [[标签]] 引用"
        + "（图→&lt;Picture k&gt;、视→&lt;Video k&gt;、音→&lt;Audio j&gt;）。参考视频的原声自动配对，无需单独传音轨。<br>"
        + "尾帧锚定不限模式：不想锚定就不上传。<br>"
        + "配套工作流下素材节点由导演台点亮/隐藏——连线常驻，只是隐藏，请勿删除。"));
}

/* ---- 链参数（面板直写画布控件）：常规五项 + 高级设置收纳其余全部 ---- */

const PRIMARY_DEFS = [W_AR, W_MP, W_DUR, W_SEED, "步数"];
const ADVANCED_DEFS = [
    "引导帧数", "CFG", "采样器", "调度器",
    "自动存档", "存档目录", "审片模式", "自动保存", "重跑起始段",
    "桥帧门控", "清晰度阈值", "回退上限", "接缝处理", "混合帧数", "锚定加噪",
    "精修强度", "精修窗口", "接缝重摇", "重摇阈值", "重摇上限",
    "智能切镜", "递减锚定", "切镜最多丢帧", "全链丢弃预算", "自适应精修",
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
        inp.step = step || (name === W_MP || name === "CFG" || name === W_DUR ? 0.1 : 1);
        if (w.options && Number.isFinite(w.options.min)) inp.min = w.options.min;
        if (w.options && Number.isFinite(w.options.max)) inp.max = w.options.max;
        inp.onchange = () => setWidgetValue(node, name, Number(inp.value));
        inp.addEventListener("wheel", (e) => e.preventDefault(), { passive: false });   // 滚轮滚动面板时不改数值
        row.append(inp);
        if (name === W_MP) {
            row.append(el("span", "h3d-secs-hint", "MP"));
            inp.title = "目标总像素 0.1–2.0MP，直接输入数字（官方口径 1MP=1024×1024）："
                + "0.2 草稿(608×352) / 0.5 预览(960×544) / 0.98 H3原生(1344×768) / 1.0(1376×768) / 2.0(1920×1088)";
            inp.min = inp.min || 0.1;   // 控件 options 未下发时兜底
            inp.max = inp.max || 2.0;
        }
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
            b.title = "官方 Resolution Selector 同款换算：1MP=1024×1024，两侧各自 round 对齐 32 倍数";
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
    const { history, prefix, state, mf } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>成片</strong><small>项目文件夹内的成片 + 保存节点历史</small>"));
    const box = el("div", "h3d-hist");

    /* 第一区块：项目文件夹内的成片（manifest.finals，最新在前） */
    const dir = state?.dir;
    const finals = (mf?.finals || []).filter(Boolean).slice(-10).reverse();
    if (dir && finals.length) {
        const head = el("div", "h3d-hist-head", `项目成片（output/h3_projects/${escapeHtml(dir)}/）`);
        box.append(head);
        finals.forEach((file, i) => {
            const sub = `h3_projects/${dir}`;
            const url = viewUrl(sub, file);
            const card = el("div", "h3d-result" + (i === 0 ? " current" : ""));
            const v = document.createElement("video");
            v.controls = true;
            v.preload = "metadata";
            v.src = url;
            card.append(v, el("div", "h3d-result-name", (i === 0 ? "最新成片 · " : "") + escapeHtml(file)));
            const meta = el("div", "h3d-result-meta");
            const acts = el("div", "h3d-result-acts");
            const open = el("a", "h3d-dl", "↗ 打开");
            open.href = url; open.target = "_blank";
            const dl = el("a", "h3d-dl", "⬇ 下载");
            dl.href = url; dl.download = file;
            acts.append(open, dl);
            const del = el("button", "h3d-btn h3d-btn-danger", "🗑 删除");
            del.title = "删除项目内该成片文件（二次确认，项目其余内容不动）";
            del.onclick = async () => {
                if (del.dataset.armed !== "1") {
                    del.dataset.armed = "1";
                    del.textContent = "确认删除？";
                    setTimeout(() => { del.dataset.armed = ""; del.textContent = "🗑 删除"; }, 2500);
                    return;
                }
                try {
                    const r = await api.fetchApi("/h3chain/delete_file", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ path: `h3_projects/${dir}/${file}` }),
                    });
                    if (r.ok) { refresh(); return; }
                    if (r.status === 404 || r.status === 405) {
                        setApiError(`删除接口未注册（HTTP ${r.status}）：请重启 ComfyUI 并检查控制台「路由已注册」日志。`);
                        return;
                    }
                    const j = await r.json().catch(() => ({}));
                    alert(`删除失败：${j.error || `HTTP ${r.status}`}`);
                } catch (e) {
                    alert("删除请求失败：" + e);
                }
            };
            acts.append(del);
            meta.append(acts);
            card.append(meta);
            box.append(card);
        });
    }

    /* 第二区块：成片保存节点（H3ChainSaver）输出历史 */
    const items = (history || []).slice(0, 10);
    if (items.length) {
        if (finals.length) box.append(el("div", "h3d-hist-head", `保存节点历史（output/${escapeHtml(prefix)}/）`));
        items.forEach((it, i) => {
            const url = viewUrl(prefix, it.file);
            const card = el("div", "h3d-result" + (!finals.length && i === 0 ? " current" : ""));
            const v = document.createElement("video");
            v.controls = true;
            v.preload = "metadata";
            v.src = url;
            card.append(v, el("div", "h3d-result-name", (!finals.length && i === 0 ? "当前成片 · " : "") + escapeHtml(it.file)));
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
                    const r = await api.fetchApi("/h3chain/delete", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({ file: `${prefix}/${it.file}` }),
                    });
                    if (r.ok) { refresh(); return; }
                    if (r.status === 404 || r.status === 405) {
                        setApiError(`删除接口未注册（HTTP ${r.status}）：请重启 ComfyUI 并检查控制台「路由已注册」日志。`);
                        return;
                    }
                    alert("删除失败（文件可能已被移动）");
                } catch (e) {
                    alert("删除请求失败：" + e);
                }
            };
            acts.append(del);
            meta.append(acts);
            card.append(meta);
            box.append(card);
        });
        if ((history || []).length > 10) {
            box.append(el("div", "h3d-foot", `仅显示最近 10 条（共 ${history.length} 条），更早在 output/${escapeHtml(prefix)}/`));
        }
    } else if (!finals.length) {
        box.append(el("div", "h3d-empty",
            "还没有成片：开启「自动保存」后成片直接落在项目文件夹；<br>"
            + `接 H3ChainSaver 节点的输出历史会出现在 output/${escapeHtml(prefix)}/`));
    }
    sec.append(box);
    if (state?.dir) {
        sec.append(el("div", "h3d-foot", `项目文件夹：output/h3_projects/${escapeHtml(state.dir)}（分段视频也在其中）`));
    }
}

/* ---- 页脚 ---- */

function renderFooter(z, data) {
    const { node, state, mf, plan } = data;
    const total = plan ? plan.length : 0;
    const done = mf?.done ?? state?.done ?? 0;

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
        点「创建项目」会在 output/h3_projects/ 下立即建好文件夹（游戏存档槽：创建即可见，0 段起跑）。
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
    const submit = async () => {
        const name = input.value.trim();
        if (!name) { err.textContent = "项目名不能为空"; return; }
        if (!/^[0-9A-Za-z_\-一-龥]+$/.test(name)) {
            err.textContent = "仅允许中文、字母、数字、下划线、连字符";
            return;
        }
        /* 立即落盘建项目文件夹（游戏存档槽语义：创建即可见）。
           接口 404/405（未注册）时降级为旧的惰性行为——首次运行仍会建目录，不阻断。 */
        let diskOk = false;
        try {
            const r = await api.fetchApi("/h3chain/create_project", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ dir: name }),
            });
            if (r.ok) {
                diskOk = true;
            } else if (r.status === 400) {
                const j = await r.json().catch(() => ({}));
                err.textContent = `创建失败：${j.error || "项目名非法"}`;
                return;
            } else if (r.status === 404 || r.status === 405) {
                setApiError(`项目接口未注册（HTTP ${r.status}）：请重启 ComfyUI 并检查控制台「路由已注册」日志。`
                    + `本次新建未落盘，首次运行时会自动建目录。`);
            } else {
                const j = await r.json().catch(() => ({}));
                err.textContent = `创建失败：${j.error || `HTTP ${r.status}`}`;
                return;
            }
        } catch (e) {
            console.warn("[h3-director] create_project failed:", e);
        }
        if (!setDirValue(node, name)) { err.textContent = "节点上没有「存档目录/断点目录」控件"; return; }
        setWidgetValue(node, W_REROLL, 0);
        if (cb.checked) clearPrompts(node);
        setLed("idle", `新项目「${name}」已就绪${diskOk ? "（文件夹已建）" : ""}`);
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
