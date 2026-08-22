/* h3_director.js 纯函数校验 + 默认工作流模板结构校验：node tests/_js_harness.js */
const fs = require("fs");
const path = require("path");
let src = fs.readFileSync(path.join(__dirname, "../web/h3_director.js"), "utf8");
src = src
    .replace('import { app } from "/scripts/app.js";', "const app = { graph: null, registerExtension() {} };")
    .replace('import { api } from "/scripts/api.js";', "const api = { addEventListener() {}, fetchApi: async () => ({ ok: true }) };");
const mod = new Function("alert", "confirm", src + "\nreturn { resolveCanvas, snapFrames, matchCanvasCombo, remapOldWidgetValues, remapV25WidgetValues, getDs, setDs, defaultDs, addAsset, toggleSegmentRef, KIND_CAPS, removeRefImage, defaultSegment, planFromDs, defaultUpscale, setUpscaleField, toggleUpscaleInclude, upTargetCanvas };")(() => {}, () => true);
let fails = 0;
function eq(name, got, want) {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) { fails += 1; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
}
function ok(name, cond) {
    if (!cond) { fails += 1; console.error(`FAIL ${name}`); }
}

eq("16:9+0.2", mod.resolveCanvas("16:9", "0.2"), [608, 352]);
eq("16:9+0.5", mod.resolveCanvas("16:9", "0.5"), [960, 544]);
eq("16:9+0.98", mod.resolveCanvas("16:9", "0.98"), [1344, 768]);   // H3 官方原生
eq("16:9+1.0", mod.resolveCanvas("16:9", "1.0"), [1376, 768]);     // 官方口径 1MP=1024×1024
eq("16:9+1.2", mod.resolveCanvas("16:9", "1.2"), [1504, 832]);
eq("16:9+2.0", mod.resolveCanvas("16:9", "2.0"), [1920, 1088]);
eq("9:16+1.0", mod.resolveCanvas("9:16", "1.0"), [768, 1376]);
eq("21:9+1.0 无截断", mod.resolveCanvas("21:9", "1.0"), [1568, 672]);
eq("1:1+1.0", mod.resolveCanvas("1:1", "1.0"), [1024, 1024]);
eq("1:1+2.0", mod.resolveCanvas("1:1", "2.0"), [1440, 1440]);
eq("非法AR", mod.resolveCanvas("自定义", "1.0"), null);

eq("snap 5.0", mod.snapFrames(5.0), 124);
eq("snap 5.2", mod.snapFrames(5.2), 124);
eq("snap 6.0", mod.snapFrames(6.0), 141);
eq("snap 0.5", mod.snapFrames(0.5), 5);
eq("snap 15", mod.snapFrames(15), 362);

eq("match 1344x768", mod.matchCanvasCombo(1344, 768), ["16:9", 0.98]);  // 0.98 为 MP_LIST 迁移专用档
eq("match 1376x768", mod.matchCanvasCombo(1376, 768), ["16:9", 1]);
eq("match 960x544", mod.matchCanvasCombo(960, 544), ["16:9", 0.5]);
eq("match 864x480", mod.matchCanvasCombo(864, 480), ["16:9", 0.4]);
eq("match 850x480 无组合", mod.matchCanvasCombo(850, 480), null);      // 非 32 倍数必然无组合

const remapped = mod.remapOldWidgetValues([864, 480, 124, "22", 0, "fixed", 25, 1.0, "res_multistep", "simple"]);
eq("迁移头部", remapped.slice(0, 8), ["16:9", 0.4, 864, 480, 5.2, "22", 0, "fixed"]);
eq("迁移尾部保留", remapped.slice(8), [25, 1.0, "res_multistep", "simple"]);
const remapped2 = mod.remapOldWidgetValues([1344, 768, 124, "22", 9, "randomize", 30, 2.0]);
eq("迁移命中组合", remapped2.slice(0, 7), ["16:9", 0.98, 1344, 768, 5.2, "22", 9]);

/* 导演台时代 25 值 → 统一接缝后端 35 值：头部不变、接缝混合映射、新控件默认值插入、尾部保留 */
const v25 = ["16:9", "0.5", 864, 480, 5.0, "22", 0, "fixed", 25, 1.0,
    "res_multistep", "simple", "关闭", "", "标注", 30.0, 34,
    "smoothstep", 6, 0.0, "关闭", "分段", 0, "多参视频", "{\"prompts\":[\"x\"]}"];
const v35 = mod.remapV25WidgetValues(v25);
eq("v25→v35 长度", v35.length, 35);
eq("v25→v35 头部不变", v35.slice(0, 17), v25.slice(0, 17));
eq("v25→v35 接缝混合映射", v35[17], "smoothstep像素混合");
eq("v25→v35 混合帧数保留", v35[18], 6);
eq("v25→v35 新控件默认值", v35.slice(23, 33), [0.45, "39", "自动", 0.06, 1, "关闭", "关闭", 17, 48, "开启"]);
eq("v25→v35 尾部保留", v35.slice(33), ["多参视频", "{\"prompts\":[\"x\"]}"]);
const v35b = mod.remapV25WidgetValues(v25.map((v, i) => (i === 17 ? "关闭" : v)));
eq("v25→v35 关闭值保留", v35b[17], "关闭");

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

/* ---- 段级独立镜头（unlink）：getDs 归一 + setDs 回写回环 ---- */
const nodeU = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["a", "b"],
    segments: [{ scene_prompt: "s", unlink: true }, { unlink: "yes" }],
}) }], setDirtyCanvas() {} };
const dsU = mod.getDs(nodeU);
eq("unlink 布尔归一", dsU.segments.map((s) => s.unlink), [true, true]);
mod.setDs(nodeU, dsU);
eq("unlink 回写回环", mod.getDs(nodeU).segments.map((s) => s.unlink), [true, true]);
eq("defaultSegment unlink 缺省 false", mod.defaultSegment().unlink, false);

/* ---- 插入视频段（ds.inserts）：getDs 规范化（去重/过滤/排序） ---- */
const nodeI = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["a", "b", "c"],
    inserts: [
        { pos: 3, file: "  c.mp4  " },
        { pos: 2, file: "b.mp4" },
        { pos: 2, file: "dup.mp4" },       // 同链位去重保首个（getDs 先到先得：b.mp4）
        { pos: 0, file: "zero.mp4" },      // pos<1 过滤
        { pos: 1.5, file: "frac.mp4" },    // 非整数过滤
        { pos: 4, file: "" },              // 空文件名过滤
        "garbage",                          // 非对象过滤
    ],
}) }], setDirtyCanvas() {} };
const dsI = mod.getDs(nodeI);
eq("inserts 规范化", dsI.inserts, [{ pos: 2, file: "b.mp4" }, { pos: 3, file: "c.mp4" }]);

/* planFromDs 混排：链位 = 提示词段+插入段交错位置（不含序章）。
 * pos=2/3 两个插入段连续占链位 2、3（第 1 个提示词段占 1，其余提示词段垫后）——
 * 与 nodes.py exec_items 混排算法逐位一致 */
const planI = mod.planFromDs(nodeI);
eq("混排序列", planI.plan.map((it) => it.kind === "insert"
    ? ["i", it.pos, it.file] : ["p", it.idx]), [["p", 0], ["i", 2, "b.mp4"], ["i", 3, "c.mp4"], ["p", 1], ["p", 2]]);
eq("混排不吞草稿统计", planI.drafts, []);
/* 尾部插入：pos 超过 prompts 数时插入段垫后 */
const nodeT = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["a"], inserts: [{ pos: 2, file: "tail.mp4" }],
}) }], setDirtyCanvas() {} };
eq("尾部插入混排", mod.planFromDs(nodeT).plan.map((it) => it.kind),
    ["prompt", "insert"]);
/* 插入段不受 drafts 统计影响：空提示词段照常标记 */
const nodeD = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["", "b"], inserts: [{ pos: 1, file: "head.mp4" }],
}) }], setDirtyCanvas() {} };
eq("插入段不进草稿统计", mod.planFromDs(nodeD).drafts, ["段 1"]);

/* ---- 潜空间放大二采（ds.upscale）：归一化 / 回写 / 勾选 / 目标画布 ---- */
eq("defaultUpscale 默认关闭", mod.defaultUpscale().mode, "关闭");
const nodeP = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["a"],
    upscale: { mode: "乱写", model: 5, arch: "3d", scale: 9, denoise: 0.01, steps: 0,
               cfg: -3, precision: "x", include: [1, "2", 2.5, -1, "bad"] },
}) }], setDirtyCanvas() {} };
const dsP = mod.getDs(nodeP);
eq("upscale 非法模式回落关闭", dsP.upscale.mode, "关闭");
eq("upscale 非法模型名回落空串", dsP.upscale.model, "");
eq("upscale arch 只认大写 3D", dsP.upscale.arch, "2D");
eq("upscale scale 钳上限", dsP.upscale.scale, 4);
eq("upscale denoise 钳下限", dsP.upscale.denoise, 0.05);
eq("upscale steps 钳下限取整", dsP.upscale.steps, 1);
eq("upscale cfg 钳下限", dsP.upscale.cfg, 0);
eq("upscale precision 回落 fp16", dsP.upscale.precision, "fp16");
eq("upscale include 只留非负整数", dsP.upscale.include, [1, 2]);
/* 旧 JSON 无 upscale 键 -> 默认注入（关闭，与后端 parse_state 同口径） */
const nodeO = { widgets: [{ name: "导演台状态", value: JSON.stringify({ mode: "文生视频", prompts: ["a"] }) }], setDirtyCanvas() {} };
eq("旧 JSON 注入默认二采配置", mod.getDs(nodeO).upscale, mod.defaultUpscale());

/* setUpscaleField：切入手动模式保留已有勾选、切出清空；参数字段直写 */
mod.setUpscaleField(nodeP, "mode", "手动选择");
mod.toggleUpscaleInclude(nodeP, 3);
mod.toggleUpscaleInclude(nodeP, 1);
eq("手动勾选累积排序", mod.getDs(nodeP).upscale.include, [2, 3]);   // 原 [1,2]，勾3、取消1
mod.setUpscaleField(nodeP, "scale", 1.5);
eq("参数字段直写", mod.getDs(nodeP).upscale.scale, 1.5);
mod.setUpscaleField(nodeP, "mode", "跟随生成");
eq("切模式清勾选", mod.getDs(nodeP).upscale.include, []);
mod.toggleUpscaleInclude(nodeP, 2);
mod.toggleUpscaleInclude(nodeP, 2);
eq("重复勾选=取消", mod.getDs(nodeP).upscale.include, []);

/* upTargetCanvas：latent 偶数对齐（像素 32 倍数），与后端 target_hw 同口径 */
const nodeC = { widgets: [{ name: "宽度", value: 1344 }, { name: "高度", value: 768 }] };
eq("目标画布 2x", mod.upTargetCanvas(nodeC, 2), "2688×1536");
eq("目标画布 1.5x", mod.upTargetCanvas(nodeC, 1.5), "2016×1152");
eq("目标画布 2.6x 取偶", mod.upTargetCanvas(nodeC, 2.6), "3488×2016");
eq("目标画布缺控件", mod.upTargetCanvas({}, 2), "");

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
    const expect = ["模型", "文本编码器", "视频VAE", "音频VAE", "首帧图片", "尾帧锚定", "起始视频", "起始视频音轨",
        "提示词组.提示词_0", "提示词组.提示词_1", "提示词组.提示词_2",
        ...Array.from({ length: 9 }, (_v, i) => `参考图片组.参考图片_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频组.参考视频_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频音轨组.参考视频音轨_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考音频组.参考音频_${i}`)];
    eq("主节点槽位顺序", h3n.inputs.map((i) => i.name), expect);
    /* 提示词组 3 槽都已连线（运行兜底） */
    ok("提示词槽已连线", h3n.inputs.slice(8, 11).every((i) => i.link != null));
    /* 尾帧锚定：槽位 5 预连线到隐藏「尾帧图」LoadImage（任意模式可点亮） */
    const lf = [...byId.values()].find((n) => n.title === "尾帧图");
    ok("尾帧图节点存在", !!lf);
    if (lf) {
        eq("尾帧图节点隐藏", lf.mode, 2);
        eq("尾帧图→槽位5 尾帧锚定", [10, 5],
            [wf.links.find((l) => l[0] === lf.outputs[0].link)[3], wf.links.find((l) => l[0] === lf.outputs[0].link)[4]]);
    }
    /* 分段视频不再单独打包：改由主节点「自动保存」存进项目文件夹（导演台段卡片预览） */
    ok("无遗留分段打包链", ![...byId.values()].some((n) => n.type === "CreateVideo" || n.type === "SaveVideo"));
    /* widgets_values 顺序与后端 schema 控件数一致（35 项 = 34 控件 + 种子 control 占 1 位） */
    eq("主节点 widgets 数量", h3n.widgets_values.length, 35);
    eq("百万像素浮点默认 0.5", h3n.widgets_values[1], 0.5);
    eq("主节点接缝处理默认值", h3n.widgets_values[17], "标准");
    eq("主节点生成模式/导演台状态尾部", h3n.widgets_values.slice(33), ["文生视频", ""]);
    /* last_link_id / last_node_id 覆盖全部 */
    eq("last_node_id", wf.last_node_id, Math.max(...wf.nodes.map((n) => n.id)));
    eq("last_link_id", wf.last_link_id, Math.max(...wf.links.map((l) => l[0])));
}

console.log(fails ? `${fails} FAILURES` : "all js harness checks passed");
process.exit(fails ? 1 : 0);
