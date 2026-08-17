/**
 * H3 Seamless Chain —— 「长片审片」侧栏面板（项目控制台）
 *
 * 三块数据，全部经 ComfyUI 自带 /api/view 与 /upload/image 端点，零自建路由：
 *   - checkpoints/h3chain_state.json ：当前链指针 + 上次报告
 *   - checkpoints/h3chain_index.json  ：项目索引（面板项目列表的数据源）
 *   - checkpoints/<链>/manifest.json  ：逐段种子/哈希/接缝/桥分/缩略图/提示词
 *   - 画布上 H3SeamlessChainSampler 节点：跑前就读提示词组 / 插入视频控件，
 *     与后端 inserts.build_plan 同构地拼出统一段落计划 -> 跑前即有空卡片。
 *
 * 面板操作（改节点控件，再排队运行）：
 *   ▶ 继续下一段 / 开始生成：Queue（断点自动续接）
 *   ＋ 新建项目：新断点目录名，提示词沿用当前内容作底稿（下一条视频链）
 *   ⇄ 项目列表：点条目把「断点目录」切到该项目
 *   ＋ 添加一段：动态加 提示词_N 输入（优先连 PrimitiveStringMultiline）
 *   📎 插入视频：选文件上传到 input 目录，写入「插入视频」控件 位置|文件名
 *   🎲 重摇此段：设「重跑起始段」=该段全局段号 + 随机种子 → Queue；成功后自动复位
 *   ✏ 改词重跑：面板内编辑提示词写回图上输入 → Queue（哈希变化自动从该段重做）
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3SeamlessChainSampler";
const W_REROLL = "重跑起始段";
const W_SEED = "种子";
const W_DIR = "断点目录";
const W_INSERTS = "插入视频";
const MAX_SEG = 64;

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
 * 空提示词段被后端丢弃（seg_prompts 过滤空串），此处同样不排段，
 * 以空槽名列表返回供“未填写”提示。返回 {plan, drafts}。
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
    refresh();
}

function insertLines(node) {
    return String(getWidgetValue(node, W_INSERTS) ?? "").split("\n").filter((l) => l.trim());
}

function appendInsert(pos, file) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (!setWidgetValue(node, W_INSERTS, [...insertLines(node), `${pos}|${file}`].join("\n"))) {
        alert("节点上没有「插入视频」控件：旧工作流请重新添加该节点");
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
    app.queuePrompt();
}

function scheduleRefresh() {
    clearTimeout(refreshTimer);
    refreshTimer = setTimeout(refresh, 900);
}

/* ---------- 项目与运行动作 ---------- */

function doReroll(segNo) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const okReroll = setWidgetValue(node, W_REROLL, segNo);
    setWidgetValue(node, W_SEED, Math.floor(Math.random() * 2 ** 48));
    pendingReset = okReroll;
    queuePrompt();
    scheduleRefresh();
}

function switchProject(dir) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    setWidgetValue(node, W_DIR, dir);
    setWidgetValue(node, W_REROLL, 0);
    scheduleRefresh();
}

function newProject() {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    const t = new Date();
    const pad = (x) => String(x).padStart(2, "0");
    const def = `h3chain_${t.getFullYear()}${pad(t.getMonth() + 1)}${pad(t.getDate())}_${pad(t.getHours())}${pad(t.getMinutes())}`;
    const name = prompt("新项目断点目录名（提示词沿用当前内容作底稿，改完直接排队即开跑新链）：", def);
    if (!name || !name.trim()) return;
    setWidgetValue(node, W_DIR, name.trim());
    setWidgetValue(node, W_REROLL, 0);
    scheduleRefresh();
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

function fmtTime(ts) {
    if (!ts) return "";
    const d = new Date(ts * 1000);
    const pad = (x) => String(x).padStart(2, "0");
    return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

async function renderProjects(state, node) {
    const wrap = el("div", "h3c-projects");
    const idx = await fetchJson("checkpoints", "h3chain_index.json");
    const projects = (idx && idx.projects) || [];
    const active = state?.dir || "";
    const nodeDir = node ? String(getWidgetValue(node, W_DIR) ?? "").trim() : "";
    if (!projects.length) {
        wrap.append(el("div", "h3c-proj-empty", "暂无历史项目：跑一次（开审片/断点续拍）后会自动登记"));
    }
    for (const p of projects.slice(0, 30)) {
        const dir = p.dir || "?";
        const row = el("div", "h3c-proj" + (dir === active ? " h3c-proj-active" : ""));
        row.innerHTML = `
            <div class="h3c-proj-name">${escapeHtml(dir)}
                ${dir === nodeDir && nodeDir ? badge("已指向", "h3c-b-ok") : ""}</div>
            <div class="h3c-proj-meta">${p.done ?? 0}/${p.total ?? 0} 段 · ${escapeHtml(fmtTime(p.updated_at))}</div>`;
        row.title = "点击把「断点目录」切到该项目（共享参数需与其一致，否则后端会提示换链）";
        row.onclick = () => switchProject(dir);
        wrap.append(row);
    }
    const bar = el("div", "h3c-addrow");
    const add = el("button", "h3c-btn h3c-btn-primary", "＋ 新建项目");
    add.title = "新开一条视频链：换断点目录名，提示词沿用当前内容作底稿";
    add.onclick = newProject;
    bar.append(add);
    wrap.append(bar);
    return wrap;
}

function statusLine(state, mf, plan) {
    const total = plan ? plan.length : (state?.total ?? mf?.total ?? 0);
    const done = state?.done ?? mf?.done ?? 0;
    if (!total) return "尚未配置段落：在下面卡片填写提示词，或点「＋ 添加一段」";
    if (done >= total) return `本链已全部完成（${total}/${total} 段）✓ 可「＋ 新建项目」开下一条`;
    const anchored = done > 0 ? `，锚定段 ${done} 尾部` : "";
    return `已载入段 1–${done}（断点回放）→ 本次将生成段 ${done + 1}${anchored}`;
}

function mediaHtml(state, mf, idx) {
    const done = mf?.done ?? 0;
    if (idx >= done || !state?.dir) return null;
    const thumbFile = (mf.thumbs || [])[idx];
    const videoFile = (mf.videos || [])[idx];
    const thumbSrc = thumbFile ? viewUrl(state.dir, thumbFile) : "";
    if (videoFile) {
        return `<video class="h3c-video" controls preload="metadata"${thumbSrc ? ` poster="${thumbSrc}"` : ""} src="${viewUrl(state.dir, videoFile)}"></video>`;
    }
    if (thumbSrc) return `<img loading="lazy" src="${thumbSrc}">`;
    return null;
}

function renderCards(state, mf, plan, node) {
    const done = mf?.done ?? 0;
    const seeds = mf?.seeds || [];
    const seams = mf?.seams || [];
    const bridges = mf?.bridge_scores || [];
    const wrap = el("div", "h3c-cards");
    plan.forEach((it, idx) => {
        const isDone = idx < done;
        const isInsert = it.kind === "insert";
        const isHead = isInsert && it.pos === 1;
        const card = el("div", "h3c-card" + (isDone ? "" : " h3c-todo"));
        const media = mediaHtml(state, mf, idx);
        const thumbHtml = media
            ?? `<div class="h3c-thumb-empty">${isDone ? "…" : isInsert ? "插入段" : "待生成"}</div>`;
        const stateBadge = !isDone
            ? badge(isInsert ? "插入段（未跑）" : "待生成", "h3c-b-todo")
            : isHead ? badge("片头（上传）", "h3c-b-pro")
            : isInsert ? badge("插入段", "h3c-b-pro") : badge("已完成", "h3c-b-ok");
        const seedTxt = isDone && !isInsert && seeds[idx] != null ? `种子 ${seeds[idx]}` : "";
        const seamTxt = isDone && seams[idx] ? `接缝 ${seams[idx][0]}${seams[idx][1] == null ? "" : ` / ${seams[idx][1]}dB`}` : "";
        const bridgeTxt = isDone && !isInsert && bridges[idx] != null ? `桥分 ${bridges[idx]}` : "";
        const meta = [seedTxt, seamTxt, bridgeTxt].filter(Boolean).join(" · ");
        const title = isInsert ? `段 ${idx + 1}（位置 ${it.pos}）` : `段 ${idx + 1}`;
        const bodyTxt = isInsert ? escapeHtml(it.file) : escapeHtml(it.text.slice(0, 60)) + (it.text.length > 60 ? "…" : "");
        card.innerHTML = `
            <div class="h3c-thumb">${thumbHtml}</div>
            <div class="h3c-body">
                <div class="h3c-title">${title} ${stateBadge}</div>
                ${meta ? `<div class="h3c-meta">${escapeHtml(meta)}</div>` : ""}
                <div class="h3c-prompt" title="${isInsert ? "" : "点 ✏ 编辑这段提示词"}">${bodyTxt || (isInsert ? "" : "（未填写）")}</div>
                <div class="h3c-actions" data-idx="${idx}"></div>
            </div>`;
        const img = card.querySelector(".h3c-thumb img");
        if (img) img.onerror = () => img.replaceWith(el("div", "h3c-thumb-empty", "⚠ 加载失败"));
        const actions = card.querySelector(".h3c-actions");
        if (!node) {
            actions.append(el("span", "h3c-hint", "只读（画布上未找到节点）"));
        } else if (isInsert) {
            if (!isDone) {
                const rm = el("button", "h3c-btn", "✕ 移除插入");
                rm.title = "从「插入视频」清单删除这一条（其后段落自动重做）";
                rm.onclick = () => removeInsert(it.pos, it.file);
                actions.append(rm);
            } else {
                actions.append(el("span", "h3c-hint", "更换/移除插入 = 其后重做"));
            }
            const addBtn = insertButton(`在此段后`, idx + 2);
            actions.append(addBtn);
        } else {
            const editBtn = el("button", "h3c-btn h3c-btn-edit", "✏ " + (isDone ? "改词重跑" : "填写提示词"));
            editBtn.title = isDone ? "改这段提示词，保存后自动从该段重做" : "跑前填好这段的画面描述";
            editBtn.onclick = () => openEditor(card, it.entry, it.text, isDone);
            actions.append(editBtn);
            if (isDone) {
                const rerollBtn = el("button", "h3c-btn", "🎲 重摇此段");
                rerollBtn.title = `设「重跑起始段」=${idx + 1} 并换种子重新生成该段及之后`;
                rerollBtn.onclick = () => doReroll(idx + 1);
                actions.append(rerollBtn);
            } else {
                actions.append(insertButton(`插视频到段 ${idx + 1} 前`, idx + 1));
            }
        }
        wrap.append(card);
    });
    return wrap;
}

function insertButton(label, pos) {
    const b = el("button", "h3c-btn", `📎 ${label}`);
    b.title = `选择本地视频上传到 input 目录，并写入「插入视频」= ${pos}|文件名`;
    b.onclick = () => pickInsertVideo(pos);
    return b;
}

function openEditor(card, entry, text, isDone) {
    const node = findNode();
    if (!node) { alert("画布上未找到 H3 Seamless Chain 节点"); return; }
    if (card.querySelector(".h3c-editor")) return;
    const editor = el("div", "h3c-editor");
    const ta = document.createElement("textarea");
    ta.value = text;
    ta.rows = 5;
    const row = el("div", "h3c-editor-row");
    const save = el("button", "h3c-btn h3c-btn-primary", isDone ? "保存并运行" : "保存");
    const run = isDone ? null : el("button", "h3c-btn", "保存并运行");
    const cancel = el("button", "h3c-btn", "取消");
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
        scheduleRefresh();
        editor.remove();
    };
    save.onclick = () => doSave(isDone);
    if (run) run.onclick = () => doSave(true);
    cancel.onclick = () => editor.remove();
    row.append(save, ...(run ? [run] : []), cancel);
    editor.append(ta, row);
    card.querySelector(".h3c-body").append(editor);
    ta.focus();
}

async function render() {
    if (!container) return;
    const node = findNode();
    const state = await fetchJson("checkpoints", "h3chain_state.json");
    let mf = null;
    if (state && state.dir) mf = await fetchJson(`checkpoints/${state.dir}`, "manifest.json");

    container.innerHTML = "";
    if (!state && !node) {
        container.append(el("div", "h3c-empty",
            "画布上还没有 H3 Seamless Chain 节点，也没有历史链状态。添加节点并开启「审片模式」后，这里会成为项目控制台。"));
        return;
    }

    /* 项目管理区 */
    const projHead = el("div", "h3c-sec-title", "项目");
    container.append(projHead);
    container.append(await renderProjects(state, node));

    /* 当前项目头部 */
    if (state && state.dir) {
        const head = el("div", "h3c-head");
        const p = mf?.params || {};
        const paramsTxt = [p.width && `${p.width}×${p.height}`, p.length && `${p.length}f/段`, p.ctx && `引导${p.ctx}帧`]
            .filter(Boolean).join(" · ");
        head.innerHTML = `
            <div class="h3c-chain">当前：${escapeHtml(state.dir)}</div>
            <div class="h3c-sub">${escapeHtml(paramsTxt)}
                ${state.review ? badge("逐段审片中", "h3c-b-review") : ""}
                ${state.reroll > 0 ? badge(`重跑起始段=${state.reroll}`, "h3c-b-warn") : ""}
            </div>`;
        container.append(head);
    }

    /* 段落计划（跑前空卡片 + 跑后媒体） */
    let plan = null;
    let drafts = [];
    if (node) ({ plan, drafts } = planFromNode(node));
    if (!plan) {
        // 无节点（或被删）：退化为 manifest 驱动的只读卡片
        const total = state?.total ?? mf?.total ?? 0;
        plan = Array.from({ length: total }, (_v, i) => {
            const ins = (mf?.inserts || []).find((x) => x.slot === i);
            return ins ? { kind: "insert", pos: i + 1, file: ins.file || "" }
                       : { kind: "prompt", text: (mf?.prompts || [])[i] || "", entry: null };
        });
    }

    const total = plan.length;
    const done = state?.done ?? mf?.done ?? 0;
    const refreshBtn = el("button", "h3c-btn h3c-refresh", "↻");
    refreshBtn.title = "重新读取节点与断点数据";
    refreshBtn.onclick = refresh;
    const status = el("div", "h3c-status");
    status.innerHTML = `<span>${escapeHtml(statusLine(state, mf, plan))}</span>`;
    status.append(refreshBtn);
    container.append(status);

    if (drafts && drafts.length) {
        container.append(el("div", "h3c-drafts",
            `未填写（运行时跳过）：${drafts.map(escapeHtml).join("、")}`));
    }

    if (total && done < total) {
        const cont = el("button", "h3c-btn h3c-btn-primary h3c-continue",
            done > 0 ? `▶ 继续下一段（段 ${done + 1}）` : `▶ 开始生成（共 ${total} 段）`);
        cont.onclick = () => { queuePrompt(); scheduleRefresh(); };
        container.append(cont);
    }

    container.append(renderCards(state, mf, plan, node));

    const addRow = el("div", "h3c-addrow");
    if (node) {
        const addSeg = el("button", "h3c-btn", "＋ 添加一段");
        addSeg.title = "画布节点上新增一个提示词输入（PrimitiveStringMultiline）";
        addSeg.onclick = () => addPromptSegment(node);
        addRow.append(addSeg);
        if (total) {
            const tail = insertButton("尾部插入视频", total + 1);
            tail.title = `上传视频追加到成片末尾（${total + 1}|文件名）`;
            addRow.append(tail);
        }
    }
    container.append(addRow);

    if (state?.report) {
        const det = el("details", "h3c-report");
        det.innerHTML = `<summary>上次运行报告</summary><pre>${escapeHtml(state.report)}</pre>`;
        container.append(det);
    }
    if (state?.dir) {
        container.append(el("div", "h3c-foot", `链目录：output/checkpoints/${escapeHtml(state.dir)}`));
    }
}

async function refresh() {
    try {
        await render();
    } catch (e) {
        /* 渲染失败不打断 ComfyUI，仅留痕 */
        console.warn("[h3chain-panel] refresh failed:", e);
    }
}

/* ---------- 样式与入口 ---------- */

const CSS = `
.h3c-panel { display:flex; flex-direction:column; gap:10px; padding:10px; font-size:12px; color:var(--input-text, #ddd); }
.h3c-empty { opacity:.8; line-height:1.7; padding:8px 4px; }
.h3c-sec-title { font-weight:600; opacity:.9; }
.h3c-projects { display:flex; flex-direction:column; gap:4px; }
.h3c-proj { display:flex; flex-direction:column; gap:2px; padding:6px 8px; border-radius:6px; border:1px solid rgba(128,128,128,.25); background:rgba(32,32,32,.5); cursor:pointer; }
.h3c-proj:hover { background:rgba(64,64,64,.6); }
.h3c-proj-active { border-color:rgba(80,150,255,.6); }
.h3c-proj-name { font-weight:600; word-break:break-all; display:flex; gap:6px; align-items:center; flex-wrap:wrap; }
.h3c-proj-meta { font-size:10px; opacity:.7; }
.h3c-proj-empty { opacity:.6; padding:2px 4px; }
.h3c-head .h3c-chain { font-weight:600; font-size:13px; word-break:break-all; }
.h3c-head .h3c-sub { margin-top:3px; opacity:.85; display:flex; gap:6px; flex-wrap:wrap; align-items:center; }
.h3c-status { margin-top:2px; padding:7px 9px; border-radius:6px; background:rgba(80,160,255,.12); border:1px solid rgba(80,160,255,.35); line-height:1.6; display:flex; align-items:center; justify-content:space-between; gap:8px; }
.h3c-refresh { padding:1px 8px; }
.h3c-drafts { font-size:10px; opacity:.65; }
.h3c-badge { font-size:10px; padding:1px 6px; border-radius:8px; background:rgba(128,128,128,.25); }
.h3c-b-ok { background:rgba(80,200,120,.22); }
.h3c-b-pro { background:rgba(190,140,255,.22); }
.h3c-b-todo { background:rgba(128,128,128,.2); opacity:.75; }
.h3c-b-review { background:rgba(255,200,60,.2); }
.h3c-b-warn { background:rgba(255,120,90,.25); }
.h3c-cards { display:flex; flex-direction:column; gap:8px; }
.h3c-card { display:flex; gap:8px; padding:8px; border-radius:8px; background:var(--comfy-menu-bg, rgba(32,32,32,.6)); border:1px solid rgba(128,128,128,.25); }
.h3c-todo { opacity:.55; }
.h3c-thumb { width:124px; min-width:124px; aspect-ratio:16/9; border-radius:6px; overflow:hidden; background:rgba(0,0,0,.35); display:flex; align-items:center; justify-content:center; }
.h3c-thumb img { width:100%; height:100%; object-fit:cover; }
.h3c-video { width:100%; height:100%; object-fit:contain; background:#000; }
.h3c-thumb-empty { font-size:11px; opacity:.6; }
.h3c-body { flex:1; min-width:0; display:flex; flex-direction:column; gap:4px; }
.h3c-title { display:flex; gap:6px; align-items:center; font-weight:600; flex-wrap:wrap; }
.h3c-meta { opacity:.8; font-size:11px; }
.h3c-prompt { font-size:11px; opacity:.75; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h3c-actions { display:flex; gap:6px; margin-top:2px; flex-wrap:wrap; }
.h3c-hint { font-size:10px; opacity:.6; align-self:center; }
.h3c-btn { cursor:pointer; font-size:11px; padding:3px 9px; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(64,64,64,.4); color:inherit; }
.h3c-btn:hover { background:rgba(96,96,96,.55); }
.h3c-btn-primary { background:rgba(60,130,220,.4); border-color:rgba(80,150,255,.55); }
.h3c-btn-primary:hover { background:rgba(80,150,255,.55); }
.h3c-continue { align-self:stretch; text-align:center; padding:7px; font-size:12px; }
.h3c-addrow { display:flex; gap:6px; flex-wrap:wrap; }
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
                tooltip: "H3 Seamless Chain：项目管理 + 分镜卡片 + 继续下一段 / 重摇 / 改词 / 插入视频",
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
