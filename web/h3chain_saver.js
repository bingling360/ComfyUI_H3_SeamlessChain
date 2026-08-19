/**
 * H3 Chain Saver —— 成片保存节点前端（H3ChainSaver 节点体画廊）
 *
 * 三个视图：成片历史（h3saver_history.json，新在前）/ 本次分段（当前存档 seg mp4）/
 * 当前成片（最新一条大图）。删除走 POST /h3chain/delete（限本节点输出目录内）。
 * 数据经 ComfyUI 自带 /api/view 读取；输出目录随「输出前缀」控件变化。
 */
import { app } from "/scripts/app.js";
import { api } from "/scripts/api.js";

const NODE_TYPE = "H3ChainSaver";
const W_PREFIX = "输出前缀";

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

function viewUrl({ subfolder = "", filename }) {
    return `/api/view?type=output&subfolder=${encodeURIComponent(subfolder)}&filename=${encodeURIComponent(filename)}`;
}

async function fetchJson(sub, file) {
    try {
        const r = await fetch(viewUrl({ subfolder: sub, filename: file }));
        if (!r.ok) return null;
        return JSON.parse(await r.text());
    } catch (e) {
        return null;
    }
}

function fmtTime(iso) {
    return iso ? iso.replace("T", " ").slice(5, 16) : "";
}

function buildGallery(node) {
    const root = el("div", "h3sv");
    const tabs = el("div", "h3sv-tabs");
    const body = el("div", "h3sv-body");
    root.append(tabs, body);

    let tab = "history";

    async function prefix() {
        const w = (node.widgets || []).find((w) => w.name === W_PREFIX);
        return String(w && w.value || "h3_chain").trim() || "h3_chain";
    }

    async function renderHistory() {
        const pre = await prefix();
        const items = (await fetchJson(pre, "h3saver_history.json")) || [];
        body.innerHTML = "";
        if (!items.length) {
            body.append(el("div", "h3sv-empty", `还没有成片：跑一次工作流后，最终视频与分段会出现在 output/${pre}/`));
            return;
        }
        for (const it of items) {
            const url = viewUrl({ subfolder: pre, filename: it.file });
            const card = el("div", "h3sv-card");
            card.innerHTML = `
                <video controls preload="metadata" src="${url}"></video>
                <div class="h3sv-meta">
                    <b>${escapeHtml(it.file)}</b><br>
                    ${escapeHtml(fmtTime(it.time))} · ${it.frames || "?"} 帧${it.archive ? " · 存档 " + escapeHtml(it.archive) : ""}${it.segments ? " · 分段×" + it.segments : ""}
                </div>
                <div class="h3sv-acts">
                    <a class="h3sv-btn" href="${url}" target="_blank">↗打开</a>
                    <a class="h3sv-btn" href="${url}" download="${escapeHtml(it.file)}">⬇下载</a>
                    <button class="h3sv-btn h3sv-del">🗑</button>
                </div>`;
            card.querySelector(".h3sv-del").onclick = async (ev) => {
                const btn = ev.target;
                if (btn.dataset.armed !== "1") {
                    btn.dataset.armed = "1";
                    btn.textContent = "确认?";
                    setTimeout(() => { btn.dataset.armed = ""; btn.textContent = "🗑"; }, 2500);
                    return;
                }
                const r = await fetch("/h3chain/delete", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ file: `${pre}/${it.file}` }),
                });
                if (r.ok) renderHistory();
                else if (r.status === 404 || r.status === 405)
                    alert("删除路由未注册（HTTP " + r.status + "）：请重启 ComfyUI 加载修复后的插件后刷新浏览器");
                else alert("删除失败（路由不可用或文件已被移动）");
            };
            body.append(card);
        }
    }

    async function renderCurrent() {
        const pre = await prefix();
        const items = (await fetchJson(pre, "h3saver_history.json")) || [];
        body.innerHTML = "";
        if (!items.length) {
            body.append(el("div", "h3sv-empty", "还没有成片"));
            return;
        }
        const it = items[0];
        body.append(el("div", "h3sv-curtitle",
            `当前成片：${escapeHtml(it.file)} · ${escapeHtml(fmtTime(it.time))}`));
        const v = document.createElement("video");
        v.controls = true;
        v.preload = "metadata";
        v.src = viewUrl({ subfolder: pre, filename: it.file });
        body.append(v);
    }

    async function renderSegments() {
        body.innerHTML = "";
        const st = await fetchJson("checkpoints", "h3chain_state.json");
        const dir = st && st.dir;
        if (!dir) {
            body.append(el("div", "h3sv-empty", "还没有当前链：跑一次采样后这里显示其分段视频"));
            return;
        }
        const mf = (await fetchJson(`checkpoints/${dir}`, "manifest.json")) || {};
        const vids = mf.videos || [];
        const thumbs = mf.thumbs || [];
        if (!vids.length) {
            body.append(el("div", "h3sv-empty", `存档「${dir}」还没有分段视频（至少完成一段后出现）`));
            return;
        }
        body.append(el("div", "h3sv-curtitle",
            `本次分段（存档 ${escapeHtml(dir)} · ${mf.done || vids.length}/${mf.total || "?"} 段）`));
        for (let i = 0; i < vids.length; i++) {
            const poster = thumbs[i] ? ` poster="${viewUrl({ subfolder: `checkpoints/${dir}`, filename: thumbs[i] })}"` : "";
            const wrap = el("div", "h3sv-seg");
            wrap.innerHTML = `<div class="h3sv-seg-n">段 ${i + 1}</div>
                <video controls preload="metadata"${poster} src="${viewUrl({ subfolder: `checkpoints/${dir}`, filename: vids[i] })}"></video>`;
            body.append(wrap);
        }
    }

    const views = { history: renderHistory, current: renderCurrent, segs: renderSegments };
    const labels = { history: "成片历史", segs: "本次分段", current: "当前成片" };

    function renderTabs() {
        tabs.innerHTML = "";
        for (const k of Object.keys(labels)) {
            const b = el("button", "h3sv-tab" + (k === tab ? " h3sv-tab-on"), labels[k]);
            b.onclick = () => { tab = k; renderTabs(); views[k](); };
            tabs.append(b);
        }
    }

    renderTabs();
    renderHistory();
    return { root, refresh: () => views[tab]() };
}

const CSS = `
.h3sv { display:flex; flex-direction:column; gap:8px; min-width:460px; font-size:12px; color:var(--input-text,#ddd); }
.h3sv-tabs { display:flex; gap:6px; }
.h3sv-tab { cursor:pointer; font-size:11px; padding:3px 10px; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(64,64,64,.4); color:inherit; }
.h3sv-tab-on { background:rgba(60,130,220,.5); border-color:rgba(80,150,255,.6); }
.h3sv-body { display:flex; flex-direction:column; gap:8px; max-height:520px; overflow-y:auto; }
.h3sv-empty { opacity:.8; line-height:1.7; padding:6px 2px; }
.h3sv-curtitle { opacity:.9; }
.h3sv-card { display:flex; flex-direction:column; gap:4px; padding:8px; border-radius:8px; background:var(--comfy-menu-bg, rgba(32,32,32,.6)); border:1px solid rgba(128,128,128,.25); }
.h3sv-card video { width:100%; aspect-ratio:16/9; object-fit:contain; background:#000; border-radius:6px; }
.h3sv-meta { font-size:11px; opacity:.85; line-height:1.6; }
.h3sv-acts { display:flex; gap:8px; }
.h3sv-btn { cursor:pointer; font-size:11px; padding:2px 9px; border-radius:6px; border:1px solid rgba(128,128,128,.4); background:rgba(64,64,64,.4); color:inherit; text-decoration:none; }
.h3sv-del:hover { background:rgba(200,70,60,.5); }
.h3sv-seg { display:flex; flex-direction:column; gap:2px; }
.h3sv-seg-n { font-size:11px; opacity:.8; }
.h3sv-seg video { width:100%; aspect-ratio:16/9; object-fit:contain; background:#000; border-radius:6px; }
`;

const refreshers = new Set();

app.registerExtension({
    name: "H3SeamlessChain.Saver",
    setup() {
        const style = document.createElement("style");
        style.textContent = CSS;
        document.head.append(style);
        api.addEventListener("execution_success", () => { for (const fn of refreshers) fn(); });
    },
    async nodeCreated(node) {
        if (node.type !== NODE_TYPE || typeof node.addDOMWidget !== "function") return;
        const g = buildGallery(node);
        refreshers.add(g.refresh);
        const w = (node.widgets || []).find((w) => w.name === W_PREFIX);
        if (w && typeof w.callback === "function") {
            const orig = w.callback.bind(w);
            w.callback = (v) => { orig(v); g.refresh(); };
        } else if (w) {
            const origDesc = Object.getOwnPropertyDescriptor(w, "value");
            if (origDesc && origDesc.set) {
                Object.defineProperty(w, "value", { ...origDesc, set(v) { origDesc.set.call(this, v); g.refresh(); } });
            }
        }
        node.addDOMWidget("h3chain_saver", "h3saver", g.root, {
            hideOnZoom: true,
            getHeight: (elRoot) => Math.min(elRoot.scrollHeight + 10, 620),
        });
    },
});
