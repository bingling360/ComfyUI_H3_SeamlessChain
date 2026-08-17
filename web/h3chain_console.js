/**
 * H3 Chain Console —— 控制台节点前端（H3ChainConsole 节点体 DOM UI）
 *
 * 职责（后端 console.py + routes.py 配合）：
 *   存档管理：读档下拉（GET /h3chain/archives）/ 时间戳新建 / 删除（POST /h3chain/delete）
 *   提示词：各段一框在节点自身控件上（提示词组.提示词_N），⤺回填读 manifest.prompts
 *   上传：首帧图片 / 序章视频（POST /h3chain/upload -> input/h3chain/）
 *   分段浏览：选中存档的缩略图/分段视频/种子/接缝/桥分 + 重摇此段 / 改词重跑 / 继续下一段
 *
 * 数据通路：manifest 与分段视频走 ComfyUI 自带 /api/view（output 子目录），
 * 目录列举与上传走本插件 routes.py（钩子缺失时降级：手输存档名 + 当前链浏览）。
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3ChainConsole";
const SAMPLER = "H3SeamlessChainSampler";
const W_NAME = "存档名";
const W_FILE = "首帧文件";

let pendingReset = null; // {node} 重摇成功后把「重跑起始段」复位为 0
const refreshers = new Set(); // 队列清空时刷新各控制台的存档列表与段卡片

/* ---------- 通用 ---------- */

function el(tag, cls, html) {
    const e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html != null) e.innerHTML = html;
    return e;
}

function escapeHtml(s) {
    return String(s ?? "").replace(/[&<>"']/g, (c) => ({
        "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;', "'": "&#39;",
    }[c]));
}

function viewUrl({ type = "output", subfolder = "", filename }) {
    return `/api/view?type=${type}&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(filename)}`;
}

async function fetchJson(url) {
    try {
        const r = await fetch(url);
        if (!r.ok) return null;
        return JSON.parse(await r.text());
    } catch (e) {
        return null;
    }
}

function widget(node, name) {
    return (node.widgets || []).find((w) => w.name === name) || null;
}

function setWidget(node, name, value) {
    const w = widget(node, name);
    if (!w) return false;
    w.value = value;
    if (typeof w.callback === "function") {
        try { w.callback(value); } catch (e) { /* callback 可选 */ }
    }
    node.setDirtyCanvas(true, true);
    return true;
}

function promptWidgets(node) {
    return (node.widgets || [])
        .filter((w) => /^(提示词组\.)?提示词_\d+$/.test(w.name))
        .sort((a, b) => Number((a.name.match(/(\d+)$/)||[0,0])[1]) - Number((b.name.match(/(\d+)$/)||[0,0])[1]));
}

/** 沿本节点任一输出连线找采样器（找不到再全画布搜，同侧栏面板逻辑）。 */
function findSampler(node) {
    const graph = app.graph;
    for (const out of node.outputs || []) {
        for (const lid of out.links || []) {
            const link = graph.links[lid];
            const target = link && graph.getNodeById(link.target_id);
            if (target && target.type === SAMPLER) return target;
        }
    }
    return (graph._nodes || []).find((n) => n.type === SAMPLER) || null;
}

function queuePrompt() {
    app.queuePrompt();
}

/* ---------- 控制台 UI ---------- */

function stampName() {
    const t = new Date();
    const p = (x) => String(x).padStart(2, "0");
    return `h3_${t.getFullYear()}${p(t.getMonth() + 1)}${p(t.getDate())}_${p(t.getHours())}${p(t.getMinutes())}`;
}

function buildConsole(node) {
    const root = el("div", "h3cc");
    const bar = el("div", "h3cc-bar");
    const sel = el("select", "h3cc-sel");
    sel.title = "读档：选中存档后运行即从其进度续拍；列表来自 output/checkpoints/";
    const newBtn = el("button", "h3cc-btn h3cc-primary", "＋新建");
    newBtn.title = "开一条空白链：自动生成 时间戳 存档名，上一条的尾帧引导不会带入";
    const fillBtn = el("button", "h3cc-btn", "⤺回填提示词");
    fillBtn.title = "把选中存档记录的各段提示词填进本节点的提示词框（续改旧片不用重新打字）";
    const delBtn = el("button", "h3cc-btn", "🗑删除存档");
    delBtn.title = "删除选中存档（latent + 分段视频 + 缩略图，不可恢复）";
    bar.append(sel, newBtn, fillBtn, delBtn);

    const up = el("div", "h3cc-up");
    const file = el("input");
    file.type = "file";
    file.accept = "image/*,video/mp4,video/quicktime,video/x-matroska,audio/wav";
    file.className = "h3cc-file";
    file.title = "上传首帧图片（i2v）或序章视频（成片以其开头，生成段从其结尾续拍）";
    const upHint = el("span", "h3cc-hint", "");
    up.append(file, upHint);

    const status = el("div", "h3cc-status", "载入中…");
    const cards = el("div", "h3cc-cards");
    const foot = el("div", "h3cc-foot", "");

    root.append(bar, up, status, cards, foot);

    /* ----- 存档列表 ----- */

    let archives = [];

    async function loadArchives(keepSel) {
        const data = await fetchJson("/h3chain/archives");
        archives = (data && data.archives) || null;
        if (!archives) {
            // 路由不可用：降级为当前链（state.json 指针）
            const st = await fetchJson(viewUrl({ subfolder: "checkpoints", filename: "h3chain_state.json" }));
            archives = st && st.dir ? [{ name: st.dir, done: st.done, total: st.total }] : [];
            sel.innerHTML = "";
            sel.append(new Option("（路由不可用，仅显示当前链）", ""));
        } else {
            sel.innerHTML = "";
            sel.append(new Option("（手输/自动命名）", ""));
            for (const a of archives) {
                const d = new Date((a.updated_at || 0) * 1000);
                const p = (x) => String(x).padStart(2, "0");
                const date = `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
                const p1 = (a.prompts && a.prompts.find((x) => x && !x.startsWith("「序章"))) || "";
                const label = `${a.name} · ${a.done}/${a.total} · ${date}${p1 ? " · " + p1.slice(0, 18) : ""}`;
                sel.append(new Option(label, a.name));
            }
        }
        const cur = keepSel !== undefined ? keepSel : String(widget(node, W_NAME)?.value ?? "");
        sel.value = cur;
        if (sel.value !== cur) sel.selectedIndex = 0;
    }

    /* ----- 分段卡片 ----- */

    async function renderCards(name) {
        cards.innerHTML = "";
        if (!name) {
            status.textContent = "未选存档：新建一条，或选择历史存档读档续拍。";
            foot.textContent = "";
            return;
        }
        const mf = await fetchJson(viewUrl({ subfolder: `checkpoints/${name}`, filename: "manifest.json" }));
        if (!mf) {
            status.textContent = `存档「${name}」还没有 manifest：填好提示词运行一次即自动建立。`;
            foot.textContent = "";
            return;
        }
        const total = mf.total || 0;
        const done = mf.done || 0;
        status.innerHTML = `存档 <b>${escapeHtml(name)}</b> · 已完成 ${done}/${total} 段`
            + (done >= total && total ? " ✓ 本链已全部完成" : ` → 运行将生成段 ${done + 1}`)
            + (mf.has_prologue ? " · 含序章" : "");
        const hasPro = !!mf.has_prologue;
        const prompts = mf.prompts || [];
        for (let idx = 0; idx < total; idx++) {
            const isDone = idx < done;
            const isPro = hasPro && idx === 0;
            const card = el("div", "h3cc-card" + (isDone ? "" : " h3cc-todo"));
            const videoFile = isDone ? (mf.videos || [])[idx] : null;
            const thumbFile = isDone ? (mf.thumbs || [])[idx] : null;
            const poster = thumbFile ? viewUrl({ subfolder: `checkpoints/${name}`, filename: thumbFile }) : "";
            const media = videoFile
                ? `<video controls preload="metadata"${poster ? ` poster="${poster}"` : ""} src="${viewUrl({ subfolder: `checkpoints/${name}`, filename: videoFile })}"></video>`
                : `<div class="h3cc-empty">${isDone ? "…" : "待生成"}</div>`;
            const seam = isDone && (mf.seams || [])[idx];
            const bridge = isDone && (mf.bridge_scores || [])[idx];
            const seed = isDone && (mf.seeds || [])[idx];
            const meta = [
                seed != null ? `种子 ${seed}` : "",
                seam ? `接缝 ${seam[0]}${seam[1] == null ? "" : `/${seam[1]}dB`}` : "",
                bridge != null ? `桥分 ${bridge}` : "",
            ].filter(Boolean).join(" · ");
            card.innerHTML = `
                <div class="h3cc-thumb">${media}</div>
                <div class="h3cc-body">
                    <div class="h3cc-title">段 ${idx + 1} ${isDone ? (isPro ? "序章" : "✓") : "待生成"}</div>
                    ${meta ? `<div class="h3cc-meta">${escapeHtml(meta)}</div>` : ""}
                    <div class="h3cc-prompt">${escapeHtml(((prompts[idx] || "").slice(0, 60)) || "—")}</div>
                    <div class="h3cc-actions"></div>
                </div>`;
            const actions = card.querySelector(".h3cc-actions");
            if (isDone && !isPro) {
                const genIdx = idx - (hasPro ? 1 : 0) + 1; // 提示词_N（1-based 生成段）
                const roll = el("button", "h3cc-btn", "🎲重摇此段");
                roll.title = "从该段起丢弃存档重新生成（自动换种子并复位）";
                roll.onclick = () => doReroll(idx + 1);
                const edit = el("button", "h3cc-btn", "✏改词重跑");
                edit.title = "在本节点改这段提示词，保存后自动从该段重做";
                edit.onclick = () => openEditor(card, genIdx, prompts[idx] || "");
                actions.append(roll, edit);
            }
            cards.append(card);
        }
        if (done < total) {
            const cont = el("button", "h3cc-btn h3cc-primary h3cc-continue", "▶ 继续下一段");
            cont.onclick = () => { queuePrompt(); };
            cards.append(cont);
        }
        foot.textContent = `存档目录：output/checkpoints/${name}（中间 latent，跑完可删）`;
    }

    function doReroll(segNo) {
        const sampler = findSampler(node);
        if (!sampler) { alert("画布上未找到 H3 Seamless Chain 采样器节点，请先用控制台输出连线接上"); return; }
        const ok = setWidget(sampler, "重跑起始段", segNo);
        setWidget(sampler, "种子", Math.floor(Math.random() * 2 ** 48));
        if (ok) pendingReset = sampler;
        queuePrompt();
    }

    function openEditor(card, genIdx, text) {
        if (card.querySelector(".h3cc-editor")) return;
        const box = el("div", "h3cc-editor");
        const ta = document.createElement("textarea");
        ta.value = text;
        ta.rows = 3;
        const row = el("div", "h3cc-row");
        const save = el("button", "h3cc-btn h3cc-primary", "保存并运行");
        const cancel = el("button", "h3cc-btn", "取消");
        save.onclick = () => {
            const ws = promptWidgets(node);
            const w = ws[genIdx - 1];
            if (!w) {
                alert(`本节点提示词框不足（需要第 ${genIdx} 框）：请在节点「提示词组」上点 ＋ 增加后再改词`);
                return;
            }
            w.value = ta.value;
            if (typeof w.callback === "function") { try { w.callback(ta.value); } catch (e) {} }
            node.setDirtyCanvas(true, true);
            const sampler = findSampler(node);
            if (sampler) setWidget(sampler, "重跑起始段", 0);
            pendingReset = null;
            box.remove();
            queuePrompt();
        };
        cancel.onclick = () => box.remove();
        row.append(save, cancel);
        box.append(ta, row);
        card.querySelector(".h3cc-body").append(box);
        ta.focus();
    }

    /* ----- 存档操作 ----- */

    newBtn.onclick = () => {
        const name = stampName();
        setWidget(node, W_NAME, name);
        sel.value = name;
        if (sel.value !== name) sel.append(new Option(name + " · 新", name)), sel.value = name;
        renderCards(name);
    };

    fillBtn.onclick = async () => {
        const name = String(widget(node, W_NAME)?.value ?? "").trim();
        const mf = name ? await fetchJson(viewUrl({ subfolder: `checkpoints/${name}`, filename: "manifest.json" })) : null;
        const src = mf ? (mf.prompts || []).filter((p) => p && !p.startsWith("「序章")) : [];
        if (!src.length) { alert("选中存档没有记录提示词（或尚未运行）"); return; }
        const ws = promptWidgets(node);
        if (ws.length < src.length) {
            alert(`提示词框不足：存档有 ${src.length} 段，本节点现有 ${ws.length} 框。请在「提示词组」上点 ＋ 增加到 ${src.length} 框后再回填`);
            return;
        }
        src.forEach((p, i) => {
            ws[i].value = p;
            if (typeof ws[i].callback === "function") { try { ws[i].callback(p); } catch (e) {} }
        });
        node.setDirtyCanvas(true, true);
        status.textContent = `已回填 ${src.length} 段提示词（来自 ${name}）`;
    };

    delBtn.onclick = async () => {
        const name = sel.value || String(widget(node, W_NAME)?.value ?? "").trim();
        if (!name) { alert("先在下拉里选中要删除的存档"); return; }
        if (!confirm(`删除存档「${name}」？\n含 latent、分段视频与缩略图，不可恢复。`)) return;
        const r = await fetch("/h3chain/delete", {
            method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ archive: name }),
        });
        if (!r.ok) { alert("删除失败（路由不可用或存档不存在）"); return; }
        if (String(widget(node, W_NAME)?.value ?? "") === name) setWidget(node, W_NAME, "");
        await loadArchives("");
        renderCards("");
    };

    sel.onchange = () => {
        setWidget(node, W_NAME, sel.value);
        renderCards(sel.value);
    };

    /* ----- 上传 ----- */

    file.onchange = async () => {
        const f = file.files && file.files[0];
        if (!f) return;
        upHint.textContent = "上传中…";
        const fd = new FormData();
        fd.append("file", f);
        try {
            const r = await fetch("/h3chain/upload", { method: "POST", body: fd });
            const data = await r.json();
            if (!r.ok) throw new Error(data.error || "上传失败");
            setWidget(node, W_FILE, data.filename);
            const base = data.filename.split("/").slice(1).join("/");
            const sub = data.filename.split("/")[0];
            const url = viewUrl({ type: "input", subfolder: sub, filename: base });
            upHint.innerHTML = data.kind === "video"
                ? `序章：<a href="${url}" target="_blank">${escapeHtml(base)}</a>（视频解码为序章段）`
                : `首帧：<a href="${url}" target="_blank">${escapeHtml(base)}</a>`;
            const clear = el("button", "h3cc-btn", "✕");
            clear.title = "清除上传（不再用首帧/序章）";
            clear.onclick = () => { setWidget(node, W_FILE, ""); upHint.textContent = ""; file.value = ""; };
            upHint.append(clear);
        } catch (e) {
            upHint.textContent = "上传失败：" + (e.message || e);
        }
        file.value = "";
    };

    /* ----- 初始与刷新 ----- */

    async function refresh() {
        await loadArchives();
        renderCards(String(widget(node, W_NAME)?.value ?? "").trim());
    }
    refresh();
    refreshers.add(refresh);

    return root;
}

/* ---------- 样式与入口 ---------- */

const CSS = `
.h3cc { display:flex; flex-direction:column; gap:8px; min-width:460px; font-size:12px; color:var(--input-text,#ddd); }
.h3cc-bar { display:flex; gap:6px; flex-wrap:wrap; }
.h3cc-sel { flex:1; min-width:220px; font:inherit; }
.h3cc-btn { cursor:pointer; font-size:11px; padding:3px 9px; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(64,64,64,.4); color:inherit; }
.h3cc-btn:hover { background:rgba(96,96,96,.55); }
.h3cc-primary { background:rgba(60,130,220,.4); border-color:rgba(80,150,255,.55); }
.h3cc-up { display:flex; gap:8px; align-items:center; flex-wrap:wrap; }
.h3cc-file { max-width:240px; font-size:11px; }
.h3cc-hint { font-size:11px; opacity:.85; display:flex; gap:6px; align-items:center; }
.h3cc-hint a { color:#8ab8ff; }
.h3cc-status { padding:6px 9px; border-radius:6px; background:rgba(80,160,255,.1); border:1px solid rgba(80,160,255,.3); line-height:1.6; }
.h3cc-cards { display:flex; flex-direction:column; gap:8px; max-height:560px; overflow-y:auto; }
.h3cc-card { display:flex; gap:8px; padding:8px; border-radius:8px; background:var(--comfy-menu-bg, rgba(32,32,32,.6)); border:1px solid rgba(128,128,128,.25); }
.h3cc-todo { opacity:.45; }
.h3cc-thumb { width:150px; min-width:150px; aspect-ratio:16/9; border-radius:6px; overflow:hidden; background:#000; display:flex; align-items:center; justify-content:center; }
.h3cc-thumb video { width:100%; height:100%; object-fit:contain; }
.h3cc-empty { font-size:11px; opacity:.6; }
.h3cc-body { flex:1; min-width:0; display:flex; flex-direction:column; gap:4px; }
.h3cc-title { font-weight:600; }
.h3cc-meta { opacity:.8; font-size:11px; }
.h3cc-prompt { font-size:11px; opacity:.75; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.h3cc-actions { display:flex; gap:6px; margin-top:2px; }
.h3cc-editor { display:flex; flex-direction:column; gap:5px; margin-top:4px; }
.h3cc-editor textarea { width:100%; box-sizing:border-box; font:inherit; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(0,0,0,.35); color:inherit; padding:5px; resize:vertical; }
.h3cc-row { display:flex; gap:6px; }
.h3cc-continue { align-self:stretch; text-align:center; padding:7px; font-size:12px; }
.h3cc-foot { opacity:.55; font-size:10px; word-break:break-all; }
`;

app.registerExtension({
    name: "H3SeamlessChain.Console",
    setup() {
        const style = document.createElement("style");
        style.textContent = CSS;
        document.head.append(style);

        api.addEventListener("execution_success", () => {
            if (pendingReset) {
                setWidget(pendingReset, "重跑起始段", 0);
                pendingReset = null;
            }
        });
        api.addEventListener("executing", ({ detail }) => {
            if (detail === null) for (const fn of refreshers) fn();
        });
    },
    async nodeCreated(node) {
        if (node.type !== NODE_TYPE || typeof node.addDOMWidget !== "function") return;
        node.addDOMWidget("h3chain_console", "h3console", buildConsole(node), {
            hideOnZoom: true,
            getHeight: (elRoot) => Math.min(elRoot.scrollHeight + 10, 640),
        });
    },
});
