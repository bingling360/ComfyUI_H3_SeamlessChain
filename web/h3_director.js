/**
 * H3 Seamless Chain —— 「长片导演台」（全屏一体化控制台 + 侧栏迷你入口）
 *
 * 视觉参照「H3 一体化总控导演台」（深青蓝工业风暗色主题），功能为原
 * 「长片审片」侧栏面板（feature 分支）的全面升级，零 Python 内核改动。
 *
 * 数据契约（全部经 ComfyUI 自带端点，零自建路由）：
 *   - checkpoints/h3chain_state.json ：当前链指针 + 上次报告（GET /api/view）
 *   - checkpoints/h3chain_index.json  ：项目索引（feature 引擎写的；404 时用
 *     localStorage 记忆 + state.dir 兜底，main 引擎也能列项目）
 *   - checkpoints/<链>/manifest.json  ：逐段种子/哈希/接缝/桥分/缩略图/提示词
 *   - output/<前缀>/h3saver_history.json ：成片历史（前缀读画布 H3ChainSaver）
 *   - 画布上 H3SeamlessChainSampler 节点：跑前读提示词组/插入视频控件，
 *     与后端 inserts.build_plan 同构地拼出统一段落计划 -> 跑前即有空卡片
 *
 * 双分支控件名自适应（合并技术改进前后都能用）：
 *   存档目录（main）/ 断点目录（feature）；「插入视频」控件缺失时隐藏插入功能
 *
 * 操作（改节点控件，再排队运行）：
 *   ▶ 开始生成/继续下一段：Queue（存档自动续接）
 *   ＋ 新建项目：新存档目录名，提示词沿用当前内容作底稿（可勾选清空）
 *   ⇄ 项目列表：点条目把「存档目录」切到该项目
 *   ＋ 添加一段：动态加 提示词_N 输入（优先连 PrimitiveStringMultiline）
 *   📎 插入视频：上传到 input 目录，写入「插入视频」控件 位置|文件名
 *   🎲 重摇此段：设「重跑起始段」=该段全局段号 + 随机种子 → Queue；成功后自动复位
 *   ✏ 改词重跑：台内编辑提示词写回图上输入 → Queue（哈希变化自动从该段重做）
 *   成片历史：播放 / 下载 / 删除（POST /h3chain/delete，路由缺失自动隐藏删除）
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3SeamlessChainSampler";
const SAVER_TYPE = "H3ChainSaver";
const W_REROLL = "重跑起始段";
const W_SEED = "种子";
const W_INSERTS = "插入视频";
const W_DIR_NAMES = ["存档目录", "断点目录"]; // main / feature 双分支控件名
const MAX_SEG = 64;
const LS_PROJECTS = "h3d_projects";

let miniBox = null;        // 侧栏迷你卡容器
let desk = null;           // 全屏导演台 { page, zones, close }
let pendingReset = false;  // 重摇后等 execution_success 自动复位重跑起始段
let refreshTimer = null;
let deleteRouteOk = true;  // /h3chain/delete 路由可用性（首次失败后永久隐藏删除钮）
let ledPhase = "idle";     // idle | running | done | error
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

/* ---------- 画布节点操作（移植自审片面板 + 双分支兼容） ---------- */

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

/** autogrow 提示词条目：{序号 -> {name, linked}}，输入槽与节点内输入框都认。 */
function promptEntries(node) {
    const pat = /^(?:提示词组\.)?提示词_(\d+)$/;
    const found = new Map();
    for (const inp of node.inputs || []) {
        const m = pat.exec(inp.name);
        if (m) found.set(Number(m[1]), { name: inp.name, linked: inp.link != null });
    }
    for (const w of node.widgets || []) {
        const m = pat.exec(w.name);
        if (m && !found.has(Number(m[1]))) found.set(Number(m[1]), { name: w.name, linked: false });
    }
    return [...found.entries()].sort((a, b) => a[0] - b[0]);
}

function readPromptText(node, entry) {
    const inp = (node.inputs || []).find((i) => i.name === entry.name);
    if (inp && inp.link != null) {
        const src = node.graph.getNodeById(inp.link.origin_id);
        const w = src && src.widgets && src.widgets[0];
        return w ? String(w.value ?? "") : "";
    }
    const w = (node.widgets || []).find((w) => w.name === entry.name);
    return w ? String(w.value ?? "") : "";
}

function writePromptText(node, entry, text) {
    const inp = (node.inputs || []).find((i) => i.name === entry.name);
    if (inp && inp.link != null) {
        const src = node.graph.getNodeById(inp.link.origin_id);
        const w = src && src.widgets && src.widgets[0];
        if (!w) return false;
        w.value = text;
        src.setDirtyCanvas(true, true);
    } else {
        const w = (node.widgets || []).find((w) => w.name === entry.name);
        if (!w) return false;
        w.value = text;
        node.setDirtyCanvas(true, true);
    }
    node.graph.change();
    return true;
}

function clearPrompts(node) {
    for (const [, entry] of promptEntries(node)) writePromptText(node, entry, "");
}

/** 「插入视频」控件文本 -> {items:[[pos,file]], bad:[坏行]}（容错版后端 parse_inserts）。 */
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

/**
 * 客户端段落计划（与后端 inserts.build_plan 同构）：
 * 空提示词段被后端丢弃，此处同样不排段。返回 {plan, drafts}。
 */
function planFromNode(node) {
    const entries = promptEntries(node);
    const filled = [], drafts = [];
    for (const [n, entry] of entries) {
        const text = readPromptText(node, entry).trim();
        if (text) filled.push({ n, entry, text });
        else drafts.push(entry.name);
    }
    const { items } = parseInsertsSpec(getWidgetValue(node, W_INSERTS));
    const plan = [];
    let pi = 0;
    for (const [pos, file] of items) {
        while (plan.length + 1 < pos) { plan.push({ kind: "prompt", ...filled[pi++] }); }
        plan.push({ kind: "insert", pos, file });
    }
    while (pi < filled.length) plan.push({ kind: "prompt", ...filled[pi++] });
    return { plan, drafts };
}

/** 添加一段：优先创建 PrimitiveStringMultiline 并连到新输入槽（与示例工作流同形态）。 */
function addPromptSegment(node) {
    const nums = promptEntries(node).map(([n]) => n);
    if (nums.length >= MAX_SEG) { alert(`最多 ${MAX_SEG} 段提示词`); return; }
    const n = (nums.length ? Math.max(...nums) : 0) + 1;
    const iname = `提示词组.提示词_${n}`;
    if ((node.inputs || []).some((i) => i.name === iname)) return;
    let src = null;
    try {
        src = LiteGraph.createNode("PrimitiveStringMultiline");
    } catch (e) { /* 老前端无此类型 -> 退化为节点内输入框 */ }
    node.addInput(iname, "STRING");
    if (src) {
        src.title = `提示词_${n}`;
        src.pos = [node.pos[0] - 340, node.pos[1] + 40 + n * 26];
        node.graph.add(src);
        src.connect(0, node, node.inputs.findIndex((i) => i.name === iname));
    } else {
        node.addWidget("text", iname, "", () => {});
    }
    node.setDirtyCanvas(true, true);
    node.graph.change();
    scheduleRefresh();
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
        const fd = new FormData();
        fd.append("image", f);
        fd.append("type", "input");
        fd.append("overwrite", "true");
        try {
            const r = await fetch("/upload/image", { method: "POST", body: fd });
            if (!r.ok) throw new Error(`HTTP ${r.status}`);
            const j = await r.json();
            const name = j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
            appendInsert(pos, name);
        } catch (e) {
            alert(`上传失败：${e}\n（也可手动把视频放进 input 目录，再在「插入视频」里写 位置|文件名）`);
        }
    };
    input.click();
}

function queuePrompt() {
    setLed("running", "已提交队列");
    app.queuePrompt();
    scheduleRefresh();
}

function scheduleRefresh(delay = 900) {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, delay);
}

/* ---------- 项目记忆（localStorage，索引文件缺失时的兜底） ---------- */

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
    const w = p.width || gw("宽度");
    const h = p.height || gw("高度");
    const len = p.length || gw("每段帧数");
    const ctx = p.ctx || gw("引导帧数");
    return {
        geo: w && h ? `${w}×${h}` : "—",
        len: len ? `${len}f/段` : "",
        ctx: ctx ? `引导${ctx}帧` : "",
    };
}

async function collectData() {
    const node = findNode();
    const state = await fetchJson("checkpoints", "h3chain_state.json");
    const idx = await fetchJson("checkpoints", "h3chain_index.json");
    let mf = null;
    if (state && state.dir) mf = await fetchJson(`checkpoints/${state.dir}`, "manifest.json");
    const prefix = saverPrefix();
    const history = await fetchJson(prefix, "h3saver_history.json");

    let plan = null;
    let drafts = [];
    if (node) ({ plan, drafts } = planFromNode(node));
    if (!plan) {
        // 无节点（或被删）：退化为 manifest 驱动的只读卡片
        const total = (state && state.total) ?? (mf && mf.total) ?? 0;
        plan = Array.from({ length: total }, (_v, i) => {
            const ins = ((mf && mf.inserts) || []).find((x) => x.slot === i);
            return ins ? { kind: "insert", pos: i + 1, file: ins.file || "" }
                       : { kind: "prompt", text: ((mf && mf.prompts) || [])[i] || "", entry: null };
        });
    }
    return { node, state, idx, mf, plan, drafts, history, prefix };
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
    :root{--h3d-ink:#0d1319;--h3d-panel:#151e27;--h3d-panel2:#1b2732;--h3d-line:#2e4252;--h3d-cyan:#65c8d8;--h3d-copper:#d18a4a;--h3d-bone:#edf2f4;--h3d-muted:#91a2ad;--h3d-ok:#6fca9a;--h3d-warn:#e9b96f;--h3d-danger:#e26d6d}
    .h3d-page *,.h3d-mini *,.h3d-dialog *{box-sizing:border-box}
    @keyframes h3d-blink{50%{opacity:.35}}
    .h3d-led{display:inline-flex;gap:7px;align-items:center;color:var(--h3d-muted);font-size:12px;white-space:nowrap}
    .h3d-led i{width:8px;height:8px;border-radius:50%;background:#788995;flex:none}
    .h3d-led.running i{background:var(--h3d-cyan);box-shadow:0 0 10px #65c8d8cc;animation:h3d-blink 1s infinite}
    .h3d-led.done i{background:var(--h3d-ok);box-shadow:0 0 12px #6fca9a88}
    .h3d-led.error i{background:var(--h3d-warn);box-shadow:0 0 12px #e9b96f88}
    .h3d-btn{cursor:pointer;border:1px solid #334957;border-radius:6px;background:#16222b;color:var(--h3d-bone);padding:6px 11px;font-size:12px;font-family:inherit}
    .h3d-btn:hover{filter:brightness(1.16)}
    .h3d-btn:disabled{opacity:.5;cursor:not-allowed;filter:none}
    .h3d-btn-cyan{border-color:#3f7385;background:#12313c;color:#7ce9f8}
    .h3d-btn-danger{border-color:#6d3f43;background:#2c1c22;color:#ff9a9a}
    .h3d-btn-cta{border:0;background:var(--h3d-copper);color:#101820;font-weight:700}
    .h3d-chip{font-size:10.5px;padding:2px 8px;border-radius:10px;border:1px solid #405362;background:#1a2730;color:#aebdc5;white-space:nowrap}
    .h3d-chip.ok{border-color:#3f8062;background:#18382a;color:#8ce1ad}
    .h3d-chip.media{border-color:#765f3d;background:#352b1d;color:#e9b96f}
    .h3d-chip.cyan{border-color:#3f7385;background:#12313c;color:#7ce9f8}
    .h3d-chip.warn{border-color:#6d3f43;background:#4a2026;color:#ff8585}

    /* ---- 侧栏迷你入口卡 ---- */
    .h3d-mini{display:flex;flex-direction:column;gap:9px;padding:10px;font-size:12px;color:var(--h3d-bone);background:linear-gradient(150deg,#121b23,#182630);border:1px solid #314756;border-radius:10px}
    .h3d-mini-head{display:flex;justify-content:space-between;align-items:flex-start;gap:8px}
    .h3d-mini-brand{font-weight:700;letter-spacing:.04em;min-width:0}
    .h3d-mini-brand small{display:block;margin-top:3px;color:var(--h3d-cyan);font-weight:500;font-size:10.5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-rail{display:flex;gap:3px}
    .h3d-mini-rail span{flex:1;height:4px;border-radius:4px;background:#304452}
    .h3d-mini-rail span.done{background:var(--h3d-cyan)}
    .h3d-mini-rail span.next{background:#3d6b7a;animation:h3d-blink 1.2s infinite}
    .h3d-mini-cards{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:8px}
    .h3d-mini-card{min-height:62px;padding:9px;border:1px solid #2d414f;border-radius:8px;background:#111a21;font-size:11px;color:var(--h3d-muted);line-height:1.65;overflow:hidden}
    .h3d-mini-count{display:grid;place-items:center;padding:9px 10px;min-width:66px;border:1px solid #2d414f;border-radius:8px;background:#111a21;font:700 20px/1.15 ui-monospace,Consolas;color:var(--h3d-cyan);text-align:center}
    .h3d-mini-count small{font-size:10px;color:var(--h3d-muted);font-weight:400}
    .h3d-mini-foot{display:flex;flex-direction:column;gap:8px}
    .h3d-mini-params{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-mini-open{width:100%;padding:9px}

    /* ---- 全屏导演台 ---- */
    .h3d-page{position:fixed;inset:0;z-index:1000000;background:var(--h3d-ink);color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif;display:grid;grid-template-rows:58px 1fr 58px}
    .h3d-topbar{display:grid;grid-template-columns:minmax(0,1fr) auto;align-items:center;gap:16px;padding:0 20px;border-bottom:1px solid var(--h3d-line);background:#0f171e}
    .h3d-top-left{min-width:0;display:flex;gap:14px;align-items:center}
    .h3d-kicker{color:var(--h3d-copper);font:700 11px/1 ui-monospace,Consolas;letter-spacing:.18em;white-space:nowrap}
    .h3d-title{font-size:17px;font-weight:700;white-space:nowrap}
    .h3d-sub{min-width:0;color:var(--h3d-muted);font-family:ui-monospace,Consolas;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-top-right{display:flex;gap:14px;align-items:center;justify-self:end}
    .h3d-close{width:36px;height:36px;border:1px solid var(--h3d-line);border-radius:7px;background:#18232c;color:var(--h3d-bone);cursor:pointer;font-size:15px}
    .h3d-close:hover{filter:brightness(1.2)}
    .h3d-stage{min-height:0;display:grid;grid-template-columns:minmax(255px,300px) minmax(400px,1fr) minmax(255px,300px);gap:1px;background:var(--h3d-line)}
    .h3d-col{min-width:0;min-height:0;background:var(--h3d-panel);overflow:auto}
    .h3d-sechead{position:sticky;top:0;z-index:3;padding:14px 16px 10px;background:#151e27ed;backdrop-filter:blur(8px);border-bottom:1px solid #283b48}
    .h3d-sechead strong{display:block}
    .h3d-sechead small{color:var(--h3d-muted)}

    .h3d-projlist{display:grid;gap:7px;padding:12px}
    .h3d-proj{display:grid;gap:3px;padding:9px 10px 9px 13px;border:1px solid #2b414f;border-radius:8px;background:#111a21;cursor:pointer;box-shadow:inset 3px 0 0 #293e4a;text-align:left;font-family:inherit;color:inherit}
    .h3d-proj:hover{background:#16222b}
    .h3d-proj.active{box-shadow:inset 3px 0 0 var(--h3d-cyan);border-color:#3f7385}
    .h3d-proj-name{font-weight:700;word-break:break-all;font-size:12.5px;display:flex;gap:6px;align-items:center;flex-wrap:wrap;color:var(--h3d-bone)}
    .h3d-proj-meta{color:var(--h3d-muted);font-size:11px;font-family:ui-monospace,Consolas}
    .h3d-newrow{display:flex;gap:8px;padding:0 12px 12px;flex-wrap:wrap}
    .h3d-meter{margin:0 12px 12px;padding:11px;border:1px solid #324957;border-radius:8px;background:#111a21}
    .h3d-meter strong{display:block;color:var(--h3d-cyan);font:700 16px/1.3 ui-monospace,Consolas}
    .h3d-meter p{margin:4px 0 0;color:var(--h3d-muted);font-size:11px}
    .h3d-drafts{margin:0 12px 12px;color:var(--h3d-muted);font-size:11px;line-height:1.7}

    .h3d-center-pad{padding:14px 18px 20px}
    .h3d-statusbar{display:flex;align-items:center;gap:10px;padding:10px 13px;border:1px solid #3f7385;border-radius:8px;background:#12313c;line-height:1.6}
    .h3d-statusbar .h3d-st-text{flex:1;min-width:0}
    .h3d-statusbar .h3d-chip{align-self:center}
    .h3d-refresh{padding:4px 10px;flex:none}
    .h3d-rail{display:flex;gap:4px;margin:12px 0 14px}
    .h3d-rail span{flex:1;height:5px;border-radius:4px;background:#304452}
    .h3d-rail span.done{background:var(--h3d-cyan)}
    .h3d-rail span.next{background:#3d6b7a;animation:h3d-blink 1.2s infinite}
    .h3d-cards{display:grid;gap:10px}
    .h3d-card{display:grid;grid-template-columns:150px minmax(0,1fr);gap:11px;padding:10px;border:1px solid #2b414f;border-radius:9px;background:#111a21}
    .h3d-card.todo{opacity:.72}
    .h3d-card.todo:hover{opacity:1}
    .h3d-thumb{width:150px;aspect-ratio:16/9;border-radius:6px;overflow:hidden;background:#0a1014;display:grid;place-items:center;color:#5d707c;font:700 11px ui-monospace,Consolas}
    .h3d-thumb video,.h3d-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-thumb video.h3d-segvideo{object-fit:contain;background:#000}
    .h3d-cbody{min-width:0;display:flex;flex-direction:column;gap:5px}
    .h3d-ctitle{display:flex;gap:7px;align-items:center;flex-wrap:wrap;font-weight:700}
    .h3d-cmeta{color:var(--h3d-muted);font:11px ui-monospace,Consolas;word-break:break-all}
    .h3d-cprompt{color:#c3d0d6;font-size:12px;line-height:1.65;word-break:break-word}
    .h3d-actions{display:flex;gap:7px;flex-wrap:wrap;margin-top:2px}
    .h3d-hint{color:var(--h3d-muted);font-size:11px;align-self:center}
    .h3d-editor{display:grid;gap:7px;margin-top:4px;padding:10px;border:1px solid #385162;border-radius:8px;background:#0f171d}
    .h3d-editor textarea{width:100%;min-height:96px;resize:vertical;border:1px solid #314754;border-radius:6px;background:#121c24;color:var(--h3d-bone);padding:9px;font:13px/1.7 "Microsoft YaHei UI","Segoe UI",sans-serif;outline:none}
    .h3d-editor textarea:focus{border-color:#79e9ff;box-shadow:0 0 0 2px #65c8d833}
    .h3d-editor-row{display:flex;gap:7px;flex-wrap:wrap}
    .h3d-addrow{display:flex;gap:8px;flex-wrap:wrap;margin-top:12px}
    .h3d-drawer{margin-top:14px;border:1px solid #2d424f;border-radius:8px;background:#111a21}
    .h3d-drawer summary{padding:10px 12px;cursor:pointer;color:var(--h3d-cyan)}
    .h3d-drawer pre{margin:0;padding:0 12px 12px;white-space:pre-wrap;font-size:11.5px;max-height:300px;overflow:auto;color:#b9c8cf}

    .h3d-hist{padding:12px;display:grid;gap:10px;align-content:start}
    .h3d-empty{padding:16px 10px;border:1px dashed #3a5060;border-radius:8px;color:var(--h3d-muted);text-align:center;background:#101820;line-height:1.8}
    .h3d-result{padding:8px;border:1px solid #365161;border-radius:9px;background:#0e171d}
    .h3d-result.current{border-color:#3f7385}
    .h3d-result video{display:block;width:100%;max-height:230px;border-radius:6px;background:#05090c}
    .h3d-result-meta{display:flex;gap:8px;align-items:center;justify-content:space-between;margin-top:7px}
    .h3d-result-name{min-width:0;color:#b9c8cf;font-size:11.5px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-result-info{color:var(--h3d-muted);font:10.5px ui-monospace,Consolas;margin-top:5px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}
    .h3d-result-acts{display:flex;gap:6px;flex:none}
    .h3d-dl{padding:4px 8px;border:1px solid #437386;border-radius:6px;color:#7ce9f8;text-decoration:none;background:#18303a;font-size:11px}
    .h3d-dl:hover{filter:brightness(1.15)}
    .h3d-foot{padding:2px 16px 14px;color:var(--h3d-muted);font-size:10.5px;word-break:break-all;line-height:1.8}

    .h3d-footer{display:flex;align-items:center;justify-content:space-between;gap:16px;padding:0 20px;border-top:1px solid var(--h3d-line);background:#0f171e}
    .h3d-footinfo{display:flex;gap:16px;color:var(--h3d-muted);flex-wrap:wrap;min-width:0;font-size:11.5px;align-items:center}
    .h3d-footinfo b{color:var(--h3d-bone)}
    .h3d-run{min-width:150px;padding:11px 18px}

    /* ---- 素材与参考 / 链参数 ---- */
    .h3d-lsec,.h3d-rsec{display:flex;flex-direction:column;min-height:0}
    .h3d-col.left .h3d-sechead,.h3d-col.right .h3d-sechead{position:static;backdrop-filter:none}
    .h3d-assets{display:grid;gap:8px;padding:12px}
    .h3d-asset{display:grid;grid-template-columns:64px minmax(0,1fr) auto;gap:9px;align-items:center;padding:8px;border:1px solid #2b414f;border-radius:8px;background:#111a21;box-shadow:inset 3px 0 0 #293e4a}
    .h3d-asset.on{box-shadow:inset 3px 0 0 var(--h3d-cyan)}
    .h3d-asset-thumb{width:64px;height:52px;border-radius:5px;background:#202c35;display:grid;place-items:center;overflow:hidden;color:#6f818d;font:700 10px ui-monospace,Consolas}
    .h3d-asset-thumb img{width:100%;height:100%;object-fit:cover}
    .h3d-asset-copy{min-width:0}
    .h3d-asset-copy strong{display:block;font-size:12px}
    .h3d-asset-copy small{display:block;margin-top:3px;color:var(--h3d-muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .h3d-asset-acts{display:flex;gap:5px}
    .h3d-asset-acts .h3d-btn{padding:4px 8px;font-size:11px}
    .h3d-addasset{justify-self:start;padding:6px 12px}
    .h3d-params{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:8px;padding:12px}
    .h3d-param label{display:block;margin-bottom:4px;color:var(--h3d-muted);font-size:11px}
    .h3d-select,.h3d-seedrow input{width:100%;border:1px solid #314754;border-radius:6px;background:#121c24;color:var(--h3d-bone);padding:6px 7px;font-size:12px;outline:none;font-family:inherit}
    .h3d-select:focus,.h3d-seedrow input:focus{border-color:#79e9ff}
    .h3d-seedrow{display:flex;gap:5px}
    .h3d-seedrow input{flex:1;min-width:0}
    .h3d-seedrow .h3d-btn{padding:4px 8px;flex:none}
    .h3d-psec .h3d-foot,.h3d-asec .h3d-foot,.h3d-projsec .h3d-foot{padding:0 16px 14px}

    /* ---- 新建项目模态 ---- */
    .h3d-overlay{position:fixed;z-index:1000003;inset:0;display:grid;place-items:center;padding:24px;background:#05090dcf;backdrop-filter:blur(10px)}
    .h3d-dialog{width:min(470px,calc(100vw - 40px));border:1px solid #3a5464;border-radius:14px;background:linear-gradient(160deg,#17232c,#0e161c 62%);box-shadow:0 22px 64px #000b;padding:20px;color:var(--h3d-bone);font:13px/1.5 "Microsoft YaHei UI","Segoe UI",sans-serif}
    .h3d-dialog h3{margin:0 0 6px;font-size:16px}
    .h3d-dialog .h3d-lead{color:var(--h3d-muted);margin:0 0 14px;line-height:1.75;font-size:12px}
    .h3d-dialog input[type=text]{width:100%;border:1px solid #314754;border-radius:6px;background:#121c24;color:var(--h3d-bone);padding:9px 10px;font:13px ui-monospace,Consolas;outline:none;margin-bottom:6px}
    .h3d-dialog input[type=text]:focus{border-color:#79e9ff}
    .h3d-err{color:#ff8585;font-size:11px;min-height:16px;margin-bottom:6px}
    .h3d-check{display:flex;gap:8px;align-items:center;color:var(--h3d-muted);font-size:12px;margin-bottom:14px;cursor:pointer}
    .h3d-dialog-row{display:flex;gap:8px;justify-content:flex-end}

    .h3d-fab{position:fixed;right:16px;top:120px;z-index:80;width:44px;height:44px;border-radius:50%;border:1px solid #3f7385;background:#18303a;color:#7ce9f8;cursor:pointer;font-size:17px}
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
    foot.append(el("div", "h3d-mini-params",
        escapeHtml([ps.geo, ps.len, ps.ctx].filter(Boolean).join(" · ") || "参数待首次运行后显示")));
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
    colC.append(el("div", "h3d-sechead", "<strong>段落流水线</strong><small>卡片 = 生成顺序；✏ 改词 · 🎲 重摇 · 📎 插视频</small>"));
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
    const { state, plan } = data;
    return JSON.stringify({
        dir: state?.dir ?? "",
        done: state?.done ?? 0,
        total: plan?.length ?? 0,
        review: !!state?.review,
        reroll: state?.reroll ?? 0,
        plan: (plan || []).map((it) => it.kind === "insert" ? ["i", it.pos, it.file] : ["p", it.text]),
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

    /* 左栏：项目与链 + 素材与参考 */
    renderLeftColumn(z.lProj, data);
    renderAssetsZone(z.lAssets, data);

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
        const meter = el("div", "h3d-meter");
        meter.innerHTML = `<strong>${done}/${total || "?"} 段${state.review ? " · 审片中" : ""}</strong>
            <p>${escapeHtml([ps.geo, ps.len, ps.ctx].filter(Boolean).join(" · ") || "参数待首次运行后显示")}</p>`;
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
    const locked = !!colC.querySelector(".h3d-editor") || isVideoPlaying(colC);
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

    /* 段落进度轨 */
    if (total > 0) {
        const rail = el("div", "h3d-rail");
        for (let i = 0; i < Math.min(total, MAX_SEG); i++) {
            rail.append(el("span", i < done ? "done" : i === done ? "next" : ""));
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
        addSeg.title = "画布节点上新增一个提示词输入（PrimitiveStringMultiline）";
        addSeg.onclick = () => addPromptSegment(node);
        addrow.append(addSeg);
        if (hasInserts(node)) {
            if (total) {
                const tail = insertButton("尾部插入视频", total + 1);
                tail.title = `上传视频追加到成片末尾（${total + 1}|文件名）`;
                addrow.append(tail);
            }
        }
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
                ? "还没有段落：点上方「＋ 添加一段」或在画布节点填写提示词组，卡片会立即出现。"
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
        body.append(title);

        const seedTxt = isDone && !isInsert && seeds[idx] != null ? `种子 ${seeds[idx]}` : "";
        const seamTxt = isDone && seams[idx]
            ? `接缝 ${seams[idx][0]}${seams[idx][1] == null ? "" : ` / ${seams[idx][1]}dB`}` : "";
        const bridgeTxt = isDone && !isInsert && bridges[idx] != null ? `桥分 ${bridges[idx]}` : "";
        const meta = [seedTxt, seamTxt, bridgeTxt].filter(Boolean).join(" · ");
        if (meta) body.append(el("div", "h3d-cmeta", escapeHtml(meta)));

        const text = isInsert ? it.file : it.text;
        body.append(el("div", "h3d-cprompt",
            escapeHtml(text.slice(0, 160)) + (text.length > 160 ? "…" : "") || (isInsert ? "" : "（未填写）")));

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
            const editable = !!it.entry;
            const editBtn = el("button", "h3d-btn h3d-btn-cyan", "✏ " + (isDone ? "改词重跑" : "填写提示词"));
            editBtn.title = editable
                ? (isDone ? "改这段提示词，保存后自动从该段重做" : "跑前填好这段的画面描述")
                : "历史段只读（画布上无对应提示词输入）";
            if (editable) {
                editBtn.onclick = () => openEditor(card, it.entry, it.text, isDone);
            } else {
                editBtn.disabled = true;
            }
            actions.append(editBtn);
            if (isDone) {
                const rerollBtn = el("button", "h3d-btn", "🎲 重摇此段");
                rerollBtn.title = `设「重跑起始段」=${idx + 1} 并换种子重新生成该段及之后`;
                rerollBtn.onclick = () => doReroll(idx + 1);
                actions.append(rerollBtn);
            } else if (canInsert) {
                actions.append(insertButton(`插视频到段 ${idx + 1} 前`, idx + 1));
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

function openEditor(card, entry, text, isDone) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (card.querySelector(".h3d-editor")) return;
    const editor = el("div", "h3d-editor");
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.rows = 5;
    const row = el("div", "h3d-editor-row");
    const save = el("button", "h3d-btn h3d-btn-cyan", isDone ? "保存并运行" : "保存");
    const run = isDone ? null : el("button", "h3d-btn h3d-btn-cta", "保存并运行");
    const cancel = el("button", "h3d-btn", "取消");
    const doSave = (queue) => {
        if (!writePromptText(node, entry, ta.value)) {
            alert(`写回失败：未找到 ${entry.name} 的输入；若已转输入框请在节点上直接修改`);
            return;
        }
        if (queue) {
            setWidgetValue(node, W_REROLL, 0);
            pendingReset = false;
            queuePrompt();
        }
        scheduleRefresh(200);
        editor.remove();
    };
    save.onclick = () => doSave(isDone);
    if (run) run.onclick = () => doSave(true);
    cancel.onclick = () => editor.remove();
    row.append(save, ...(run ? [run] : []), cancel);
    editor.append(ta, row);
    card.querySelector(".h3d-cbody").append(editor);
    ta.focus();
}

/* ---- 素材与参考（上传即自动接线 LoadImage，免画布连线） ---- */

function inputViewUrl(name) {
    const s = String(name ?? "");
    const i = s.lastIndexOf("/");
    const sub = i < 0 ? "" : s.slice(0, i);
    const file = i < 0 ? s : s.slice(i + 1);
    return `/api/view?type=input&subfolder=${encodeURIComponent(sub)}&filename=${encodeURIComponent(file)}`;
}

async function uploadToInput(file) {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const r = await fetch("/upload/image", { method: "POST", body: fd });
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    const j = await r.json();
    return j.subfolder ? `${j.subfolder}/${j.name}` : j.name;
}

function loadImageWidget(imgNode) {
    return (imgNode.widgets || []).find((w) => w.name === "image") || (imgNode.widgets || [])[0] || null;
}

function linkedImageInfo(node, inp) {
    if (inp.link == null) return { src: null, file: "" };
    const src = node.graph.getNodeById(inp.link.origin_id);
    const w = src && loadImageWidget(src);
    return { src, file: w ? String(w.value ?? "") : "" };
}

/** 参考图片槽位（编号 0 起，与后端 prefix="参考图片_" min=0 一致）。 */
function refImageSlots(node) {
    const pat = /^(?:参考图片组\.)?参考图片_(\d+)$/;
    const out = [];
    for (const inp of node.inputs || []) {
        const m = pat.exec(inp.name);
        if (!m) continue;
        const info = linkedImageInfo(node, inp);
        out.push({
            num: Number(m[1]), name: inp.name,
            idx: node.inputs.indexOf(inp), linked: inp.link != null,
            src: info.src, file: info.file,
        });
    }
    return out.sort((a, b) => a.num - b.num);
}

function ensureRefImageSlot(node, num) {
    const iname = `参考图片组.参考图片_${num}`;
    let inp = (node.inputs || []).find((i) => i.name === iname);
    if (!inp) {
        node.addInput(iname, "IMAGE");
        node.graph.change();
        inp = (node.inputs || []).find((i) => i.name === iname);
    }
    return inp ? node.inputs.indexOf(inp) : -1;
}

/** 断开图片输入；来源若是无其他连线的 LoadImage 则一并移除。 */
function detachImageInput(node, idx) {
    const inp = (node.inputs || [])[idx];
    if (!inp || inp.link == null) return;
    const src = node.graph.getNodeById(inp.link.origin_id);
    node.disconnectInput(idx);
    if (src && src.type === "LoadImage" && !(src.outputs || []).some((o) => o.links && o.links.length)) {
        node.graph.remove(src);
    }
    node.graph.change();
}

/** 创建 LoadImage 节点（摆到采样器左侧）并连到目标输入槽。 */
function attachLoadImage(node, targetIdx, fileName, title) {
    let img = null;
    try { img = LiteGraph.createNode("LoadImage"); } catch (e) { /* 老前端 */ }
    if (!img) { alert("当前前端无法自动创建 LoadImage 节点：请手动加载图片并连到该输入"); return false; }
    detachImageInput(node, targetIdx);
    img.title = title;
    attachLoadImage.n = (attachLoadImage.n || 0) + 1;
    img.pos = [node.pos[0] - 300, node.pos[1] - 40 + attachLoadImage.n * 110];
    node.graph.add(img);
    const w = loadImageWidget(img);
    if (w) {
        w.value = fileName;
        if (typeof w.callback === "function") {
            try { w.callback(fileName); } catch (e) { /* callback 可选 */ }
        }
    }
    img.connect(0, node, targetIdx);
    node.graph.change();
    return true;
}

function pickRefImage(kind) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async () => {
        const f = input.files && input.files[0];
        if (!f) return;
        let name;
        try { name = await uploadToInput(f); }
        catch (e) { alert(`上传失败：${e}`); return; }
        if (kind === "first") {
            const idx = (node.inputs || []).findIndex((i) => i.name === "首帧图片");
            if (idx < 0) { alert("节点上没有「首帧图片」输入（可能被隐藏：右键节点 → Inputs 勾选）"); return; }
            attachLoadImage(node, idx, name, "首帧图片");
        } else {
            const slots = refImageSlots(node);
            if (slots.filter((s) => s.linked).length >= 9) { alert("参考图片最多 9 张"); return; }
            // used 只收已连线槽位：空槽位直接复用，无空槽时取最小未用编号（后端按编号排序压实）
            const used = new Set(slots.filter((s) => s.linked).map((s) => s.num));
            const free = slots.find((s) => !s.linked);
            let num = free ? free.num : 0;
            while (used.has(num) && num <= 9) num++;
            if (num > 9) { alert("参考图片最多 9 张"); return; }
            const idx = ensureRefImageSlot(node, num);
            if (idx < 0) { alert("无法添加参考图片输入槽"); return; }
            attachLoadImage(node, idx, name, `参考图片_${num}`);
        }
        scheduleRefresh(200);
    };
    input.click();
}

function assetCard(title, file) {
    const card = el("div", "h3d-asset" + (file ? " on" : ""));
    const thumb = el("div", "h3d-asset-thumb", file ? "" : "<span>IMG</span>");
    if (file) {
        const im = document.createElement("img");
        im.loading = "lazy";
        im.src = inputViewUrl(file);
        im.alt = title;
        thumb.append(im);
    }
    const copy = el("div", "h3d-asset-copy");
    copy.innerHTML = `<strong>${escapeHtml(title)}</strong><small>${escapeHtml(file || "未设置")}</small>`;
    card.append(thumb, copy);
    return card;
}

function renderAssetsZone(sec, data) {
    const { node } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>素材与参考</strong><small>上传即自动接线；提示词用 &lt;Picture 1..9&gt; 引用</small>"));
    const box = el("div", "h3d-assets");
    if (!node) {
        box.append(el("div", "h3d-empty", "画布上未找到 H3 Seamless Chain 节点"));
        sec.append(box);
        return;
    }

    /* 首帧图片（单槽，与片头插入互斥） */
    const fInp = (node.inputs || []).find((i) => i.name === "首帧图片");
    const fInfo = fInp ? linkedImageInfo(node, fInp) : { file: "" };
    const fIdx = fInp ? node.inputs.indexOf(fInp) : -1;
    const fcard = assetCard("首帧图片", fInfo.file);
    fcard.querySelector("small").textContent = fInfo.file || "未设置（可选，与片头插入互斥）";
    const facts = el("div", "h3d-asset-acts");
    const fup = el("button", "h3d-btn h3d-btn-cyan", fInfo.file ? "替换" : "上传");
    fup.title = "上传图片到 input 目录并自动接线到「首帧图片」";
    fup.onclick = () => pickRefImage("first");
    facts.append(fup);
    if (fInfo.file && fIdx >= 0) {
        const frm = el("button", "h3d-btn h3d-btn-danger", "✕");
        frm.title = "断开首帧图片（自动移除对应 LoadImage 节点）";
        frm.onclick = () => { detachImageInput(node, fIdx); scheduleRefresh(200); };
        facts.append(frm);
    }
    fcard.append(facts);
    box.append(fcard);

    /* 参考图片组（0 起编号，≤9 张；后端按编号排序压实，<Picture i> = 排序后第 i 张） */
    const slots = refImageSlots(node).filter((s) => s.linked);
    slots.forEach((s, i) => {
        const card = assetCard(`参考图片_${s.num} → <Picture ${i + 1}>`, s.file || "已接线");
        const acts = el("div", "h3d-asset-acts");
        const rm = el("button", "h3d-btn h3d-btn-danger", "✕");
        rm.title = "移除这张参考图（断线并删对应 LoadImage 节点）";
        rm.onclick = () => { detachImageInput(node, s.idx); scheduleRefresh(200); };
        acts.append(rm);
        card.append(acts);
        box.append(card);
    });
    if (slots.length < 9) {
        const add = el("button", "h3d-btn h3d-addasset", "＋ 添加参考图片");
        add.title = "上传图片 → 自动创建 LoadImage 并连到下一个参考图片槽";
        add.onclick = () => pickRefImage("ref");
        box.append(add);
    }
    sec.append(box);
    sec.append(el("div", "h3d-foot",
        "参考视频 / 参考音频仍走画布接线（LoadVideo / LoadAudio），见节点内说明"));
}

/* ---- 链参数（面板直写画布控件） ---- */

const PARAM_DEFS = ["宽度", "高度", "每段帧数", "引导帧数", "步数", "CFG", "种子", "采样器", "调度器", "审片模式", "自动保存"];

function paramsSig(node) {
    if (!node) return "n";
    return PARAM_DEFS.map((n) => {
        const w = (node.widgets || []).find((x) => x.name === n);
        return w ? String(w.value) : "-";
    }).join("|");
}

function renderParamsZone(sec, data) {
    const { node } = data;
    sec.replaceChildren();
    sec.append(el("div", "h3d-sechead",
        "<strong>链参数</strong><small>直接写画布节点控件，随工作流保存</small>"));
    if (!node) {
        sec.append(el("div", "h3d-empty", "画布上未找到节点，参数面板不可用"));
        return;
    }
    const grid = el("div", "h3d-params");
    for (const name of PARAM_DEFS) {
        const w = (node.widgets || []).find((x) => x.name === name);
        const field = el("div", "h3d-param");
        field.append(el("label", "", name));
        if (!w) {
            field.append(el("span", "h3d-hint", "—"));
            grid.append(field);
            continue;
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
            inp.step = (w.options && w.options.step) || (name === "CFG" ? 0.1 : 1);
            inp.onchange = () => setWidgetValue(node, name, Number(inp.value));
            row.append(inp);
            if (name === "种子") {
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
        grid.append(field);
    }
    sec.append(grid);
    sec.append(el("div", "h3d-foot",
        "更多参数（桥帧门控 / 接缝混合 / 锚定加噪等）在画布节点上调整"));
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
