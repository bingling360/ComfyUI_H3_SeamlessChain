/* h3_director.js 纯函数校验 + 默认工作流模板结构校验：node tests/_js_harness.js */
const fs = require("fs");
const path = require("path");
let src = fs.readFileSync(path.join(__dirname, "../web/h3_director.js"), "utf8");
src = src
    .replace('import { app } from "/scripts/app.js";', "const app = { graph: null, registerExtension() {} };")
    .replace('import { api } from "/scripts/api.js";', "const api = { addEventListener() {}, fetchApi: async () => ({ ok: true }) };");
const mod = new Function("alert", "confirm", src + "\nreturn { resolveCanvas, snapFrames, matchCanvasCombo, remapOldWidgetValues, getDs, setDs, defaultDs, addAsset, toggleSegmentRef, KIND_CAPS, removeRefImage };")(() => {}, () => true);
let fails = 0;
function eq(name, got, want) {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) { fails += 1; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
}
function ok(name, cond) {
    if (!cond) { fails += 1; console.error(`FAIL ${name}`); }
}

eq("16:9+1.0", mod.resolveCanvas("16:9", "1.0"), [1344, 768]);
eq("9:16+1.0", mod.resolveCanvas("9:16", "1.0"), [768, 1344]);
eq("16:9+0.5", mod.resolveCanvas("16:9", "0.5"), [960, 544]);
eq("21:9+1.0 上限收敛", mod.resolveCanvas("21:9", "1.0"), [1344, 576]);
eq("非法AR", mod.resolveCanvas("自定义", "1.0"), null);

eq("snap 5.0", mod.snapFrames(5.0), 124);
eq("snap 5.2", mod.snapFrames(5.2), 124);
eq("snap 6.0", mod.snapFrames(6.0), 141);
eq("snap 0.5", mod.snapFrames(0.5), 5);
eq("snap 15", mod.snapFrames(15), 362);

eq("match 1344x768", mod.matchCanvasCombo(1344, 768), ["16:9", "1.0"]);
eq("match 864x480 无组合", mod.matchCanvasCombo(864, 480), null);

const remapped = mod.remapOldWidgetValues([864, 480, 124, "22", 0, "fixed", 25, 1.0, "res_multistep", "simple"]);
eq("迁移头部", remapped.slice(0, 8), ["自定义", "0.5", 864, 480, 5.2, "22", 0, "fixed"]);
eq("迁移尾部保留", remapped.slice(8), [25, 1.0, "res_multistep", "simple"]);
const remapped2 = mod.remapOldWidgetValues([1344, 768, 124, "22", 9, "randomize", 30, 2.0]);
eq("迁移命中组合", remapped2.slice(0, 7), ["16:9", "1.0", 1344, 768, 5.2, "22", 9]);

/* v1 状态迁移 + v2 读写回环 */
const node = { widgets: [{ name: "导演台状态", value: JSON.stringify({ mode: "多参视频", prompts: ["画面"], ref_images: ["a.png", "b.png"] }) }], setDirtyCanvas() {} };
const ds = mod.getDs(node);
eq("v1->v2 标签", ds.ref_assets.map((a) => a.label), ["图片1", "图片2"]);
eq("v1->v2 kind 默认图片", ds.ref_assets.map((a) => a.kind), ["image", "image"]);
eq("v1->v2 ref_images 同步", ds.ref_images, ["a.png", "b.png"]);
ds.segments[0].seconds = 6.5;
ds.segments[0].refs = ["图片1"];
mod.setDs(node, ds);
const ds2 = mod.getDs(node);
eq("秒数回写", ds2.segments[0].seconds, 6.5);
eq("refs 回写", ds2.segments[0].refs, ["图片1"]);
eq("标签引用有效", ds2.ref_assets[0].label, "图片1");

/* v2 三类素材：kind 解析 / 缺省标签按类别编号 / ref_images 仅图片类 */
const node2 = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "多参视频", prompts: ["x"],
    ref_assets: [
        { file: "a.png", label: "角色1" },
        { file: "v.mp4", kind: "video" },
        { file: "s.mp3", kind: "audio" },
        { file: "b.png", kind: "bogus" },   // 非法 kind 归一为 image
    ],
}) }], setDirtyCanvas() {} };
const ds3 = mod.getDs(node2);
eq("v2 kind 归一", ds3.ref_assets.map((a) => a.kind), ["image", "video", "audio", "image"]);
eq("v2 缺省标签按类别", ds3.ref_assets.map((a) => a.label), ["角色1", "视频1", "音频1", "图片1"]);
mod.setDs(node2, ds3);
eq("ref_images 仅图片类", mod.getDs(node2).ref_images, ["a.png", "b.png"]);

/* 上限：图片 9 / 视频 3 / 音频 3；超限静默拒绝 */
const node3 = { widgets: [{ name: "导演台状态", value: "" }], setDirtyCanvas() {} };
for (let i = 0; i < 12; i++) mod.addAsset(node3, "image", `i${i}.png`);
for (let i = 0; i < 6; i++) mod.addAsset(node3, "video", `v${i}.mp4`);
for (let i = 0; i < 6; i++) mod.addAsset(node3, "audio", `s${i}.mp3`);
const ds4 = mod.getDs(node3);
eq("图片上限 9", ds4.ref_assets.filter((a) => a.kind === "image").length, mod.KIND_CAPS.image);
eq("视频上限 3", ds4.ref_assets.filter((a) => a.kind === "video").length, mod.KIND_CAPS.video);
eq("音频上限 3", ds4.ref_assets.filter((a) => a.kind === "audio").length, mod.KIND_CAPS.audio);
eq("视频缺省标签", ds4.ref_assets.find((a) => a.kind === "video").label, "视频1");

/* 段级勾选上限：视频 3 个封顶，第 4 个拒绝；图片同理 */
const seg = { scene_prompt: "", character_prompt: "", seconds: null, refs: [] };
node3.widgets[0].value = JSON.stringify({ mode: "多参视频", prompts: ["p1", "p2"],
    ref_assets: [1, 2, 3, 4].map((i) => ({ file: `v${i}.mp4`, kind: "video", label: `视频${i}` }))
        .concat([1, 2, 3, 4, 5, 6, 7, 8, 9, 10].map((i) => ({ file: `i${i}.png`, kind: "image", label: `图${i}` }))),
    segments: [seg, { ...seg }] });
for (let i = 1; i <= 4; i++) mod.toggleSegmentRef(node3, 0, `视频${i}`);
eq("段勾选视频封顶 3", mod.getDs(node3).segments[0].refs, ["视频1", "视频2", "视频3"]);
for (let i = 1; i <= 10; i++) mod.toggleSegmentRef(node3, 1, `图${i}`);
eq("段勾选图片封顶 9", mod.getDs(node3).segments[1].refs.length, mod.KIND_CAPS.image);

/* 删除素材：段级引用联动清理 */
mod.removeRefImage(node3, 0);
ok("删除后无视频1引用", !mod.getDs(node3).segments[0].refs.includes("视频1"));

/* ---- 默认工作流模板结构校验 ---- */
const tplSrc = fs.readFileSync(path.join(__dirname, "../web/h3_default_workflow.js"), "utf8");
const sandbox = new Function("window", tplSrc);
const win = {};
sandbox(win);
const wf = win.H3_DEFAULT_WORKFLOW;
ok("模板导出存在", !!wf);
if (wf) {
    const byId = new Map(wf.nodes.map((n) => [n.id, n]));
    const h3n = [...byId.values()].find((n) => n.type === "H3SeamlessChainSampler");
    ok("主节点存在", !!h3n);
    /* 每条 link 两端节点存在、槽位下标合法、类型一致 */
    for (const [lid, sid, sslot, did, dslot, type] of wf.links) {
        const s = byId.get(sid), d = byId.get(did);
        ok(`link${lid} 源节点存在(${sid})`, !!s);
        ok(`link${lid} 目标节点存在(${did})`, !!d);
        if (!s || !d) continue;
        const so = s.outputs[sslot], di = d.inputs[dslot];
        ok(`link${lid} 源槽位合法`, !!so);
        ok(`link${lid} 目标槽位合法`, !!di);
        if (!so || !di) continue;
        ok(`link${lid} 类型一致 ${so.type}->${di.type}`, so.type === type && di.type === type);
        ok(`link${lid} 槽位回指`, so.link === lid && di.link === lid);
    }
    /* 主节点输入槽顺序与后端 schema 一致（autogrow 全组 0 起编号，含提示词组） */
    const expect = ["模型", "文本编码器", "视频VAE", "音频VAE", "首帧图片", "起始视频", "起始视频音轨",
        "提示词组.提示词_0", "提示词组.提示词_1", "提示词组.提示词_2",
        ...Array.from({ length: 9 }, (_v, i) => `参考图片组.参考图片_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频组.参考视频_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频音轨组.参考视频音轨_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考音频组.参考音频_${i}`)];
    eq("主节点槽位顺序", h3n.inputs.map((i) => i.name), expect);
    /* 提示词组 3 槽都已连线（运行兜底） */
    ok("提示词槽已连线", h3n.inputs.slice(7, 10).every((i) => i.link != null));
    /* 分段输出链：分段图像/音频 → CreateVideo → SaveVideo */
    const cv = [...byId.values()].find((n) => n.type === "CreateVideo");
    const sv = [...byId.values()].find((n) => n.type === "SaveVideo");
    ok("CreateVideo 存在", !!cv);
    ok("SaveVideo 存在", !!sv);
    if (cv && sv) {
        eq("分段图像→CreateVideo.images", [10, 4], [wf.links.find((l) => l[0] === cv.inputs[0].link)[1], wf.links.find((l) => l[0] === cv.inputs[0].link)[2]]);
        eq("分段音频→CreateVideo.audio", [10, 5], [wf.links.find((l) => l[0] === cv.inputs[1].link)[1], wf.links.find((l) => l[0] === cv.inputs[1].link)[2]]);
        ok("CreateVideo→SaveVideo", sv.inputs[0].link != null
            && wf.links.find((l) => l[0] === sv.inputs[0].link)[1] === cv.id);
        eq("CreateVideo fps=24", cv.widgets_values[0], 24.0);
    }
    /* widgets_values 顺序与后端 schema 控件数一致（25 项 = 24 控件 + 种子 control 占 2 位） */
    eq("主节点 widgets 数量", h3n.widgets_values.length, 25);
    /* last_link_id / last_node_id 覆盖全部 */
    eq("last_node_id", wf.last_node_id, Math.max(...wf.nodes.map((n) => n.id)));
    eq("last_link_id", wf.last_link_id, Math.max(...wf.links.map((l) => l[0])));
}

console.log(fails ? `${fails} FAILURES` : "all js harness checks passed");
process.exit(fails ? 1 : 0);
