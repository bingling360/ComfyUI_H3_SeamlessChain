/**
 * H3 Seamless Chain —— 「长片审片」侧栏面板
 *
 * 数据全部来自断点目录（output/checkpoints/<链>/）：
 *   - checkpoints/h3chain_state.json ：当前链指针 + 上次报告
 *   - <链>/manifest.json             ：逐段种子/哈希/接缝/桥分/缩略图/提示词
 * 经 ComfyUI 自带 /api/view 端点读取，零自建路由、零外部依赖。
 *
 * 面板操作（改的是画布上 H3SeamlessChainSampler 节点的控件，然后排队运行）：
 *   ▶ 继续下一段：直接 Queue（断点自动续接）
 *   🎲 重摇此段：设「重跑起始段」=该段 + 随机种子 → Queue；运行成功后自动复位为 0
 *   ✏ 改词重跑：面板内编辑该段提示词，写回图上提示词输入 → Queue（哈希变化自动从该段重做）
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3SeamlessChainSampler";
const W_REROLL = "重跑起始段";
const W_SEED = "种子";
const W_DIR = "断点目录";

let container = null;
let pendingReset = false;
let refreshTimer = null;

/* ---------- 数据读取 ---------- */

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

function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;",
    }[c]));
}

/* ---------- 画布节点操作 ---------- */

function findNode() {
    const nodes = app.graph?._nodes || [];
    return nodes.find((n) => n.type === NODE_TYPE) || null;
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

function promptInputName(node, n) {
    const candidates = [`提示词组.提示词_${n}`, `提示词_${n}`];
    for (const name of candidates) {
        if ((node.inputs || []).some((inp) => inp.name === name)) return name;
    }
    return null;
}

function writePrompt(node, n, text) {
    const iname = promptInputName(node, n);
    if (!iname) return false;
    const inp = node.inputs.find((i) => i.name === iname);
    if (inp && inp.link != null) {
        const src = node.graph.getNodeById(inp.link.origin_id);
        const w = src && src.widgets && src.widgets[0];
        if (!w) return false;
        w.value = text;
        src.setDirtyCanvas(true, true);
    } else {
        const w = (node.widgets || []).find((w) => w.name === iname);
        if (!w) return false;
        w.value = text;
        node.setDirtyCanvas(true, true);
    }
    node.graph.change();
    return true;
}

function queuePrompt() {
    app.queuePrompt();
}

function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, 900);
}

/* ---------- 渲染 ---------- */

function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
}

function badge(text, cls) {
    return `<span class="h3c-badge ${cls || ""}">${escapeHtml(text)}</span>`;
}

function statusLine(state, mf) {
    const done = state.done ?? (mf?.done ?? 0);
    const total = state.total ?? (mf?.total ?? 0);
    if (!total) return "尚未运行：先在画布上排队一次工作流";
    if (done >= total) return `本链已全部完成（${total}/${total} 段）✓`;
    const anchored = done > 0 ? `，锚定段 ${done} 尾部` : "";
    return `已载入段 1–${done}（断点回放）→ 本次将生成段 ${done + 1}${anchored}`;
}

function renderCards(state, mf) {
    const total = state.total ?? (mf?.total ?? 0);
    const done = state.done ?? (mf?.done ?? 0);
    const hasPrologue = !!(mf?.has_prologue);
    const thumbs = mf?.thumbs || [];
    const vids = mf?.videos || [];
    const seeds = mf?.seeds || [];
    const seams = mf?.seams || [];
    const bridges = mf?.bridge_scores || [];
    const prompts = mf?.prompts || [];
    const wrap = el("div", "h3c-cards");
    for (let idx = 0; idx < total; idx++) {
        const isDone = idx < done;
        const isPrologue = hasPrologue && idx === 0;
        const card = el("div", "h3c-card" + (isDone ? "" : " h3c-todo"));
        const thumbFile = thumbs[idx];
        const videoFile = isDone ? vids[idx] : null;
        const thumbSrc = isDone && thumbFile ? viewUrl(state.dir, thumbFile) : "";
        let media;
        if (videoFile) {
            media = `<video class="h3c-video" controls preload="metadata"${thumbSrc ? ` poster="${thumbSrc}"` : ""} src="${viewUrl(state.dir, videoFile)}"></video>`;
        } else if (thumbSrc) {
            media = `<img loading="lazy" src="${thumbSrc}">`;
        } else {
            media = `<div class="h3c-thumb-empty">${isDone ? "…" : "待生成"}</div>`;
        }
        const stateBadge = !isDone ? badge("待生成", "h3c-b-todo")
            : isPrologue ? badge("序章（上传）", "h3c-b-pro") : badge("已完成", "h3c-b-ok");
        const seedTxt = isDone && seeds[idx] != null ? `种子 ${seeds[idx]}` : "";
        const seamTxt = isDone && seams[idx] ? `接缝 ${seams[idx][0]}${seams[idx][1] == null ? "" : ` / ${seams[idx][1]}dB`}` : "";
        const bridgeTxt = isDone && bridges[idx] != null ? `桥分 ${bridges[idx]}` : "";
        const meta = [seedTxt, seamTxt, bridgeTxt].filter(Boolean).join(" · ");
        const promptTxt = escapeHtml((prompts[idx] || "").slice(0, 60)) + ((prompts[idx] || "").length > 60 ? "…" : "");
        card.innerHTML = `
            <div class="h3c-thumb">${media}</div>
            <div class="h3c-body">
                <div class="h3c-title">段 ${idx + 1} ${stateBadge}</div>
                ${meta ? `<div class="h3c-meta">${escapeHtml(meta)}</div>` : ""}
                ${promptTxt ? `<div class="h3c-prompt" title="双击卡片可编辑提示词">${promptTxt}</div>` : ""}
                <div class="h3c-actions" data-idx="${idx}"></div>
            </div>`;
        const img = card.querySelector(".h3c-thumb img");
        if (img) img.onerror = () => img.replaceWith(el("div", "h3c-thumb-empty", "⚠ 加载失败"));
        const actions = card.querySelector(".h3c-actions");
        if (isDone && !isPrologue) {
            const genIdx = idx - (hasPrologue ? 1 : 0) + 1; // 提示词_N（1-based 生成段）
            const rerollBtn = el("button", "h3c-btn", "🎲 重摇此段");
            rerollBtn.title = `设「重跑起始段」=${idx + 1} 并换种子重新生成该段及之后`;
            rerollBtn.onclick = () => doReroll(idx + 1);
            const editBtn = el("button", "h3c-btn h3c-btn-edit", "✏ 改词重跑");
            editBtn.title = "在面板里改这段的提示词，保存后自动从该段重做";
            editBtn.onclick = () => openEditor(card, genIdx, prompts[idx] || "");
            actions.append(rerollBtn, editBtn);
        } else if (isPrologue) {
            actions.append(el("span", "h3c-hint", "更换上传视频 = 整链重做"));
        }
        wrap.append(card);
    }
    return wrap;
}

function doReroll(segNo) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const okReroll = setWidgetValue(node, W_REROLL, segNo);
    setWidgetValue(node, W_SEED, Math.floor(Math.random() * 2 ** 48));
    pendingReset = okReroll;
    queuePrompt();
    scheduleRefresh();
}

function newProject() {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const t = new Date();
    const pad = (x) => String(x).padStart(2, "0");
    const def = `h3chain_${t.getFullYear()}${pad(t.getMonth() + 1)}${pad(t.getDate())}_${pad(t.getHours())}${pad(t.getMinutes())}${pad(t.getSeconds())}`;
    const name = prompt("新项目断点目录名（新目录 = 空白链，上一条的尾帧引导不会带入）：", def);
    if (!name || !name.trim()) return;
    setWidgetValue(node, W_DIR, name.trim());
    setWidgetValue(node, W_REROLL, 0);
    pendingReset = false;
    scheduleRefresh();
}

function openEditor(card, genIdx, text) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (card.querySelector(".h3c-editor")) return;
    const editor = el("div", "h3c-editor");
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.rows = 5;
    const row = el("div", "h3c-editor-row");
    const save = el("button", "h3c-btn h3c-btn-primary", "保存并运行");
    const cancel = el("button", "h3c-btn", "取消");
    save.onclick = () => {
        if (!writePrompt(node, genIdx, ta.value)) {
            alert(`未找到该段的提示词输入（提示词_${genIdx}）；若用的是节点内输入框，请直接在节点上修改`);
            return;
        }
        setWidgetValue(node, W_REROLL, 0);
        pendingReset = false;
        queuePrompt();
        scheduleRefresh();
    };
    cancel.onclick = () => editor.remove();
    row.append(save, cancel);
    editor.append(ta, row);
    card.querySelector(".h3c-body").append(editor);
    ta.focus();
}

function render(state, mf) {
    if (!container) return;
    container.innerHTML = "";

    // 新建项目按钮：始终可用（跑前 / 跑后都能开下一条空白链）
    const projBar = el("div", "h3c-projbar");
    const newBtn = el("button", "h3c-btn h3c-btn-primary", "＋ 新建项目");
    newBtn.title = "换一个新断点目录 = 空白链：上一条视频的尾帧引导不会带入，从第 1 段重新画起";
    newBtn.onclick = newProject;
    projBar.append(newBtn);
    container.append(projBar);

    if (!state || !state.dir) {
        container.append(el("div", "h3c-empty",
            "还没有链状态：点上方「＋ 新建项目」开一条空白链，或直接运行一次工作流（开启「审片模式」或「断点续拍」）。"));
        return;
    }
    const head = el("div", "h3c-head");
    const p = mf?.params || {};
    const paramsTxt = [p.width && `${p.width}×${p.height}`, p.length && `${p.length}f/段`, p.ctx && `引导${p.ctx}帧`]
        .filter(Boolean).join(" · ");
    head.innerHTML = `
        <div class="h3c-chain">${escapeHtml(state.dir)}</div>
        <div class="h3c-sub">${escapeHtml(paramsTxt)}
            ${state.review ? badge("逐段审片中", "h3c-b-review") : ""}
            ${state.reroll > 0 ? badge(`重跑起始段=${state.reroll}`, "h3c-b-warn") : ""}
        </div>
        <div class="h3c-status">${escapeHtml(statusLine(state, mf))}</div>`;
    container.append(head);

    const done = state.done ?? (mf?.done ?? 0);
    const total = state.total ?? (mf?.total ?? 0);
    if (done < total) {
        const cont = el("button", "h3c-btn h3c-btn-primary h3c-continue", "▶ 继续下一段");
        cont.onclick = () => { queuePrompt(); scheduleRefresh(); };
        container.append(cont);
    } else if (total > 0 && done >= total) {
        // 本链跑完：提示开下一条（按钮已在顶部，这里只给文字引导）
        container.append(el("div", "h3c-next-hint",
            `本链已全部完成 ✓ 想开下一条视频？点顶部「＋ 新建项目」换新目录即可，提示词会沿用作底稿。`));
    }
    container.append(renderCards(state, mf));

    if (state.report) {
        const det = el("details", "h3c-report");
        det.innerHTML = `<summary>上次运行报告</summary><pre>${escapeHtml(state.report)}</pre>`;
        container.append(det);
    }
    const foot = el("div", "h3c-foot", `链目录：output/checkpoints/${escapeHtml(state.dir)}`);
    container.append(foot);
}

async function refresh() {
    const state = await fetchJson("checkpoints", "h3chain_state.json");
    let mf = null;
    if (state && state.dir) mf = await fetchJson(`checkpoints/${state.dir}`, "manifest.json");
    render(state, mf);
}

/* ---------- 样式与入口 ---------- */

const CSS = `
.h3c-panel { display:flex; flex-direction:column; gap:10px; padding:10px; font-size:12px; color:var(--input-text, #ddd); }
.h3c-projbar { display:flex; gap:6px; }
.h3c-projbar .h3c-btn { padding:6px 12px; font-size:12px; }
.h3c-next-hint { padding:8px 10px; border-radius:6px; background:rgba(80,200,120,.12); border:1px solid rgba(80,200,120,.35); line-height:1.6; opacity:.9; }
.h3c-empty { opacity:.8; line-height:1.7; padding:8px 4px; }
.h3c-head .h3c-chain { font-weight:600; font-size:13px; word-break:break-all; }
.h3c-head .h3c-sub { margin-top:3px; opacity:.85; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.h3c-status { margin-top:6px; padding:7px 9px; border-radius:6px; background:rgba(80,160,255,.12); border:1px solid rgba(80,160,255,.35); line-height:1.6; }
.h3c-badge { font-size:10px; padding:1px 6px; border-radius:8px; background:rgba(128,128,128,.25); }
.h3c-b-ok { background:rgba(80,200,120,.22); }
.h3c-b-pro { background:rgba(190,140,255,.22); }
.h3c-b-todo { background:rgba(128,128,128,.2); opacity:.75; }
.h3c-b-review { background:rgba(255,200,60,.2); }
.h3c-b-warn { background:rgba(255,120,90,.25); }
.h3c-cards { display:flex; flex-direction:column; gap:8px; }
.h3c-card { display:flex; gap:8px; padding:8px; border-radius:8px; background:var(--comfy-menu-bg, rgba(32,32,32,.6)); border:1px solid rgba(128,128,128,.25); }
.h3c-todo { opacity:.45; }
.h3c-thumb { width:124px; min-width:124px; aspect-ratio:16/9; border-radius:6px; overflow:hidden; background:rgba(0,0,0,.35); display:flex; align-items:center; justify-content:center; }
.h3c-thumb img { width:100%; height:100%; object-fit:cover; }
.h3c-video { width:100%; height:100%; object-fit:contain; background:#000; }
.h3c-thumb-empty { font-size:11px; opacity:.6; }
.h3c-body { flex:1; min-width:0; display:flex; flex-direction:column; gap:4px; }
.h3c-title { display:flex; gap:6px; align-items:center; font-weight:600; }
.h3c-meta { opacity:.8; font-size:11px; }
.h3c-prompt { font-size:11px; opacity:.75; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h3c-actions { display:flex; gap:6px; margin-top:2px; flex-wrap:wrap; }
.h3c-hint { font-size:10px; opacity:.6; align-self:center; }
.h3c-btn { cursor:pointer; font-size:11px; padding:3px 9px; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(64,64,64,.4); color:inherit; }
.h3c-btn:hover { background:rgba(96,96,96,.55); }
.h3c-btn-primary { background:rgba(60,130,220,.4); border-color:rgba(80,150,255,.55); }
.h3c-btn-primary:hover { background:rgba(80,150,255,.55); }
.h3c-continue { align-self:stretch; text-align:center; padding:7px; font-size:12px; }
.h3c-editor { display:flex; flex-direction:column; gap:5px; margin-top:4px; }
.h3c-editor textarea { width:100%; box-sizing:border-box; font:inherit; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(0,0,0,.35); color:inherit; padding:5px; resize:vertical; }
.h3c-editor-row { display:flex; gap:6px; }
.h3c-report summary { cursor:pointer; opacity:.85; }
.h3c-report pre { white-space:pre-wrap; font-size:11px; max-height:280px; overflow:auto; opacity:.85; }
.h3c-foot { opacity:.55; font-size:10px; word-break:break-all; }
.h3c-fab { position:fixed; right:16px; top:120px; z-index:80; width:44px; height:44px; border-radius:50%; border:1px solid rgba(80,150,255,.5); background:rgba(40,70,110,.85); color:#fff; cursor:pointer; font-size:18px; }
`;

const ICON = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.6"><rect x="3" y="5" width="18" height="14" rx="2"/><path d="M7 5v14M17 5v14M3 9h4M3 15h4M17 9h4M17 15h4"/></svg>`;

function mount(target) {
    target.classList.add("h3c-panel");
    container = target;
    refresh();
}

function mountFallbackDialog() {
    const btn = document.createElement("button");
    btn.className = "h3c-fab";
    btn.title = "长片审片（H3 Seamless Chain）";
    btn.innerHTML = ICON;
    let dlg = null;
    btn.onclick = () => {
        if (dlg && dlg.open) { dlg.close(); return; }
        dlg = document.createElement("dialog");
        dlg.style.cssText = "width:min(520px,92vw);color:#eee;background:#1e1e1e;border:1px solid #444;border-radius:10px;";
        const inner = document.createElement("div");
        dlg.append(inner);
        document.body.append(dlg);
        dlg.showModal();
        container = inner;
        inner.classList.add("h3c-panel");
        refresh();
    };
    document.body.append(btn);
}

app.registerExtension({
    name: "H3SeamlessChain.Panel",
    setup() {
        const style = document.createElement("style");
        style.textContent = CSS;
        document.head.append(style);

        if (app.extensionManager && typeof app.extensionManager.registerSidebarTab === "function") {
            app.extensionManager.registerSidebarTab({
                id: "h3chain-review",
                // 图标必须是已加载图标库的类名（前端内置 PrimeVue）：官方文档
                // docs.comfy.org → 侧边栏标签页；传 SVG 源码不会渲染
                icon: "pi pi-video",
                title: "长片审片",
                tooltip: "H3 Seamless Chain：逐段播放分段视频，一键继续 / 重摇 / 改词重跑",
                type: "custom",
                render: (elTarget) => mount(elTarget),
            });
        } else {
            mountFallbackDialog();
        }

        api.addEventListener("execution_success", () => {
            if (pendingReset) {
                const node = findNode();
                if (node) setWidgetValue(node, W_REROLL, 0);
                pendingReset = false;
            }
            scheduleRefresh();
        });
        api.addEventListener("executing", ({ detail }) => {
            if (detail === null) scheduleRefresh(); // 队列清空
        });
    },
});
