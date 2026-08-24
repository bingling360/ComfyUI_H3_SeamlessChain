/* h3_director.js 纯函数校验 + 默认工作流模板结构校验：node tests/_js_harness.js */
const fs = require("fs");
const path = require("path");
let src = fs.readFileSync(path.join(__dirname, "../web/h3_director.js"), "utf8");
src = src
    .replace('import { app } from "/scripts/app.js";', "const app = { graph: null, registerExtension() {} };")
    .replace('import { api } from "/scripts/api.js";', "const api = { addEventListener() {}, fetchApi: async () => ({ ok: true }) };");
const mod = new Function("alert", "confirm", src + "\nreturn { resolveCanvas, snapFrames, matchCanvasCombo, remapOldWidgetValues, remapOldWidgetValuesToCurrent, getDs, setDs, defaultDs, addAsset, toggleSegmentRef, KIND_CAPS, removeRefImage, defaultSegment, planFromDs, defaultUpscale, setUpscaleField, toggleUpscaleInclude, upTargetCanvas, defaultExperiments, normalizeExperiments, expActiveList, expLocked, parseMasterPrompt, exportMasterPrompt, applyMasterPrompt };")(() => {}, () => true);
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

/* 旧布局（35 值陈旧 / 25 值导演台时代，两者 0..22 位一致）→ 当前 29 值 schema：
 * 头部 0..16 不变，中段按当前控件取位（V25 缺失的接缝四件取默认），
 * 生成模式(idx25)/自动成片(idx26 恒"开启")/导演台状态(idx27) 落尾 */
const v35old = ["16:9", 0.5, 864, 480, 5.0, "22", 0, "fixed", 25, 1.0,
    "res_multistep", "simple", "关闭", "", "标注", 30.0, 34,
    "标准", 6, 0.0, "关闭", "分段", 0, 0.45, "39", "自动", 0.06, 1,
    "关闭", "关闭", 17, 48, "开启", "多参视频", "{\"prompts\":[\"x\"]}"];
const mig35 = mod.remapOldWidgetValuesToCurrent(v35old);
eq("v35→v29 长度", mig35.length, 29);
eq("v35→v29 头部不变", mig35.slice(0, 17), v35old.slice(0, 17));
eq("v35→v29 中段取位", mig35.slice(17, 25), [0.0, "关闭", "分段", 0, "自动", 0.06, 1, "关闭"]);
eq("v35→v29 尾部", mig35.slice(25), ["多参视频", "开启", "{\"prompts\":[\"x\"]}", "标准"]);
const v25old = v35old.slice(0, 23).concat(["文生视频", ""]);   // 25 值：尾部为生成模式/导演台状态
const mig25 = mod.remapOldWidgetValuesToCurrent(v25old);
eq("v25→v29 长度", mig25.length, 29);
eq("v25→v29 中段取位+默认", mig25.slice(17, 25), [0.0, "关闭", "分段", 0, "自动", 0.06, 1, "关闭"]);
eq("v25→v29 尾部", mig25.slice(25), ["文生视频", "开启", "", "标准"]);

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
/* 非自定义画幅：按 宽高比×百万像素 换算（与链参数徽章/后端 _resolve_canvas 同源），
 * 宽/高控件旧残留不作数——否则二采徽章会与主徽章打架 */
const nodeA = { widgets: [
    { name: "宽高比", value: "16:9" }, { name: "百万像素", value: 0.98 },
    { name: "宽度", value: 864 }, { name: "高度", value: 480 },   // 过期残留
] };
eq("非自定义按AR×MP换算", mod.upTargetCanvas(nodeA, 2), "2688×1536");
eq("非自定义 1x 即原生画布", mod.upTargetCanvas(nodeA, 1), "1344×768");
const nodeZ = { widgets: [
    { name: "宽高比", value: "自定义" }, { name: "宽度", value: 864 }, { name: "高度", value: 480 },
] };
eq("自定义仍读宽高控件", mod.upTargetCanvas(nodeZ, 2), "1728×960");

/* ---- 总提示词（parseMasterPrompt / exportMasterPrompt / applyMasterPrompt） ---- */
const mp1 = mod.parseMasterPrompt("【段1】\n场景：黄昏教室\n角色：短发少女\n提示词：推近。\n少女说「好」。\n\n【第2段】\n环境音：蝉鸣\n时长：6\n提示词：拉远。");
eq("总提示词 段数", mp1.segs.length, 2);
eq("总提示词 场景", mp1.segs[0].scene, "黄昏教室");
eq("总提示词 角色", mp1.segs[0].character, "短发少女");
eq("总提示词 主体+续行", mp1.segs[0].main, "推近。\n少女说「好」。");
eq("总提示词 未写字段=undefined", mp1.segs[0].soundscape, undefined);
eq("总提示词 环境音/时长", [mp1.segs[1].soundscape, mp1.segs[1].seconds], ["蝉鸣", 6]);
eq("总提示词 空文本零段", mod.parseMasterPrompt("  \n ").segs.length, 0);
eq("总提示词 无段头=单段", mod.parseMasterPrompt("整段都是主体文本").segs[0].main, "整段都是主体文本");
const mp2 = mod.parseMasterPrompt("游离行\n【段落3】\n提示词：x");
eq("段头变体+序号不一致提示", mp2.segs[0].main, "x");
ok("序号不一致进 notes", mp2.notes.some((n) => n.includes("不一致")));
ok("游离行进 notes", mp2.notes.some((n) => n.includes("游离") || n.includes("之前")));
eq("CRLF 容忍", mod.parseMasterPrompt("【段1】\r\n提示词：a\r\n续行").segs[0].main, "a\n续行");
eq("官方标签不受影响", mod.parseMasterPrompt("【段1】\n提示词：integrated_multimodal_description: [Shot 1] 测试").segs[0].main,
    "integrated_multimodal_description: [Shot 1] 测试");
eq("非法时长忽略", mod.parseMasterPrompt("【段1】\n时长：abc\n提示词：x").segs[0].seconds, undefined);
eq("写空=显式清空标记", mod.parseMasterPrompt("【段1】\n场景：\n提示词：x").segs[0].scene, "");
/* 应用：段数重排 + 未提及字段保留（refs/unlink）+ 显式清空生效 + inserts 不动 */
const nodeM = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频",
    prompts: ["旧1", "旧2", "旧3"],
    inserts: [{ pos: 2, file: "keep.mp4" }],
    segments: [
        { scene_prompt: "旧场景", refs: ["图片1"], unlink: true },
        {}, {},
    ],
    ref_assets: [{ file: "a.png", kind: "image", label: "图片1" }],
}) }], setDirtyCanvas() {} };
const mpApply = mod.applyMasterPrompt(nodeM, "【段1】\n场景：\n配乐：钢琴\n提示词：新主体\n\n【段2】\n提示词：第二段");
const dsM = mod.getDs(nodeM);
eq("应用后段数重排", dsM.prompts, ["新主体", "第二段"]);
eq("应用后 segments 同步伸缩", dsM.segments.length, 2);
eq("写空清空场景", dsM.segments[0].scene_prompt, "");
eq("新字段写入", dsM.segments[0].music, "钢琴");
eq("未提及字段保留（refs/unlink）", [dsM.segments[0].refs, dsM.segments[0].unlink], [["图片1"], true]);
eq("inserts 不动", dsM.inserts, [{ pos: 2, file: "keep.mp4" }]);
ok("应用返回解析结果", mpApply.segs.length === 2);
/* 导出 → 解析回环：非空字段逐项一致 */
const nodeE2 = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "文生视频", prompts: ["主体一", "主体二"],
    segments: [
        { scene_prompt: "夜景", character_prompt: "侦探", soundscape: "雨声", music: "爵士", seconds: 6.5 },
        {},
    ],
}) }], setDirtyCanvas() {} };
const rt = mod.parseMasterPrompt(mod.exportMasterPrompt(nodeE2));
eq("回环 段数", rt.segs.length, 2);
eq("回环 五字段+时长", [rt.segs[0].scene, rt.segs[0].character, rt.segs[0].soundscape, rt.segs[0].music, rt.segs[0].seconds],
    ["夜景", "侦探", "雨声", "爵士", 6.5]);
eq("回环 空段仅主体", [rt.segs[1].main, rt.segs[1].scene], ["主体二", undefined]);
/* ---- 总提示词扩展标签：独立镜头 / 参考 / 【完】截断 ---- */
const mp3 = mod.parseMasterPrompt("【段1】\n独立镜头：是\n参考：角色1，图片2、图片3\n提示词：x\n【段2】\n独立镜头：否\n参考：\n提示词：y\n【完】\n参考素材建议：\n角色1 = 三视图……（应整体忽略）");
eq("独立镜头 是", mp3.segs[0].unlink, true);
eq("独立镜头 否", mp3.segs[1].unlink, false);
eq("参考 多分隔符切分", mp3.segs[0].refs, ["角色1", "图片2", "图片3"]);
eq("参考 写空=清空标记", mp3.segs[1].refs, []);
eq("【完】后内容不进主体", mp3.segs[1].main, "y");
ok("【完】后内容零警告", mp3.notes.length === 0);
eq("独立镜头 非法值忽略", mod.parseMasterPrompt("【段1】\n独立镜头：随便\n提示词：x").segs[0].unlink, undefined);
eq("独立镜头 变体 断链", mod.parseMasterPrompt("【段1】\n独立镜头：断链\n提示词：x").segs[0].unlink, true);
/* 应用：参考按已有素材标签过滤，未知剔除并提示；unlink 写入 */
const nodeR = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "多参视频", prompts: ["a"],
    ref_assets: [{ file: "c.png", kind: "image", label: "角色1" }, { file: "s.png", kind: "image", label: "风格板" }],
}) }], setDirtyCanvas() {} };
const rp = mod.applyMasterPrompt(nodeR, "【段1】\n独立镜头：是\n参考：角色1，不存在的标签\n提示词：主体");
const dsR = mod.getDs(nodeR);
eq("参考 过滤未知标签", dsR.segments[0].refs, ["角色1"]);
eq("unlink 应用", dsR.segments[0].unlink, true);
ok("未知标签进提示", rp.notes.some((n) => n.includes("不存在的标签")));
/* 导出回环：unlink/refs/【完】 */
const nodeR2 = { widgets: [{ name: "导演台状态", value: JSON.stringify({
    mode: "多参视频", prompts: ["a"],
    segments: [{ unlink: true, refs: ["角色1", "风格板"] }],
    ref_assets: [{ file: "c.png", kind: "image", label: "角色1" }, { file: "s.png", kind: "image", label: "风格板" }],
}) }], setDirtyCanvas() {} };
const rt2 = mod.parseMasterPrompt(mod.exportMasterPrompt(nodeR2));
eq("回环 unlink/refs", [rt2.segs[0].unlink, rt2.segs[0].refs], [true, ["角色1", "风格板"]]);
ok("导出带【完】", mod.exportMasterPrompt(nodeR2).includes("【完】"));
/* 围栏容忍：跨 agent 输出常被 markdown 代码围栏包裹，围栏行不进解析也不报警 */
const mpF = mod.parseMasterPrompt("```\n【段1】\n提示词：围栏内主体\n【完】\n建议\n```");
eq("围栏容忍 段与主体", [mpF.segs.length, mpF.segs[0].main, mpF.notes.length], [1, "围栏内主体", 0]);
/* 软换行丢失容错（mpReflow）：markdown 渲染界面复制把段内换行合并成空格，
 * 段头/标签糊进同一行——检测行内段头后自动重分行，字段照常对位 */
const mpM = mod.parseMasterPrompt("【段1】 场景：黄昏教室 角色：短发少女 时长：5 提示词：推近。\n\n【第2段】 环境音：蝉鸣 独立镜头：是 提示词：拉远。 【完】 备注：不应进入解析");
eq("软换行丢失 段数", mpM.segs.length, 2);
eq("软换行丢失 字段对位", [mpM.segs[0].scene, mpM.segs[0].character, mpM.segs[0].seconds], ["黄昏教室", "短发少女", 5]);
eq("软换行丢失 主体", mpM.segs[0].main, "推近。");
eq("软换行丢失 段2字段", [mpM.segs[1].soundscape, mpM.segs[1].unlink, mpM.segs[1].main], ["蝉鸣", true, "拉远。"]);
ok("软换行丢失 重分行提示", mpM.notes.some((n) => n.includes("重分行")));
eq("软换行丢失 【完】后截断", mpM.segs[1].main.includes("备注"), false);
/* 未走样文本不触发重分行：主体里的行内标签字样原样保留 */
const mpS = mod.parseMasterPrompt("【段1】\n提示词：他说「参考：无」");
eq("未走样 行内标签不动", mpS.segs[0].main, "他说「参考：无」");

/* ---- 实验性功能（扁平契约 + 主开关）：注入迷你 defs 无头校验 ---- */
const MINI_DEFS = [
    { id: "e1_test", name: "测试1", group: "g", desc: "d", params: [
        { key: "强度", type: "num", def: 5, min: 0, max: 10, step: 1 },
        { key: "模式", type: "enum", def: "甲", opts: ["甲", "乙"] },
    ] },
    { id: "e2_test", name: "测试2", group: "g", desc: "d", params: [] },
];
eq("defaultExperiments 注入 defs", mod.defaultExperiments(MINI_DEFS),
    { params: { e1_test: { "强度": 5, "模式": "甲" }, e2_test: {} } });
const flatIn = { e1_test: true, locked: false, ghost: true,
    params: { e1_test: { "强度": 99, "模式": "丙" } } };
const n1 = mod.normalizeExperiments(flatIn, MINI_DEFS);
eq("扁平 normalize 保留已知开关", n1.e1_test, true);
eq("扁平 normalize 丢弃未知 id", "ghost" in n1, false);
eq("扁平 normalize locked 透传", n1.locked, false);
eq("num 钳位到 max", n1.params.e1_test["强度"], 10);
eq("enum 白名单回退默认", n1.params.e1_test["模式"], "甲");
const n2 = mod.normalizeExperiments(flatIn, null);      // defs 未到达宽进
eq("宽进保留未知布尔键", n2.ghost, true);
eq("宽进参数字典透传", n2.params, flatIn.params);
eq("宽进 locked 透传", n2.locked, false);
/* setDs 回环：写入扁平契约，JSON 原文无 on 键、locked 持久化 */
const nodeE = { widgets: [{ name: "导演台状态", value: "" }], setDirtyCanvas() {} };
const dsE = mod.getDs(nodeE);
dsE.experiments = mod.normalizeExperiments({ e1_test: true, locked: false,
    params: { e1_test: { "强度": 7, "模式": "乙" } } }, MINI_DEFS);
mod.setDs(nodeE, dsE);
const rawE = JSON.parse(nodeE.widgets[0].value);
ok("存档为扁平契约", rawE.experiments.e1_test === true && !("on" in rawE.experiments));
eq("locked 持久化", rawE.experiments.locked, false);
eq("参数随存档保留", rawE.experiments.params.e1_test["强度"], 7);
const dsE2 = mod.getDs(nodeE);
ok("回环保留开关（宽进）", dsE2.experiments.e1_test === true);
/* 主开关逻辑：缺省推导 + 显式 locked 优先 */
eq("主开关缺省 全关=锁定", mod.expLocked({ params: {} }), true);
eq("主开关缺省 有开启=解锁", mod.expLocked({ e1_test: true, params: {} }), false);
eq("主开关显式 locked 优先", mod.expLocked({ e1_test: true, locked: true, params: {} }), true);
eq("expActiveList 只认 true 键", mod.expActiveList({ e1_test: true, locked: false, params: {} }), ["e1_test"]);

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
    const expect = ["模型", "文本编码器", "视频VAE", "音频VAE", "首帧图片", "尾帧图片", "每段尾帧锚定", "起始视频", "起始视频音轨",
        "提示词组.提示词_0", "提示词组.提示词_1", "提示词组.提示词_2",
        ...Array.from({ length: 9 }, (_v, i) => `参考图片组.参考图片_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频组.参考视频_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考视频音轨组.参考视频音轨_${i}`),
        ...Array.from({ length: 3 }, (_v, i) => `参考音频组.参考音频_${i}`)];
    eq("主节点槽位顺序", h3n.inputs.map((i) => i.name), expect);
    /* 提示词组 3 槽都已连线（运行兜底） */
    ok("提示词槽已连线", h3n.inputs.slice(9, 12).every((i) => i.link != null));
    /* 尾帧图片：槽位 5 预连线到隐藏「目标尾帧图」LoadImage（FL2VA 剧情终点，仅首帧模式） */
    const ef = [...byId.values()].find((n) => n.title === "目标尾帧图");
    ok("目标尾帧图节点存在", !!ef);
    if (ef) {
        eq("目标尾帧图节点隐藏", ef.mode, 2);
        eq("目标尾帧图→槽位5 尾帧图片", [10, 5],
            [wf.links.find((l) => l[0] === ef.outputs[0].link)[3], wf.links.find((l) => l[0] === ef.outputs[0].link)[4]]);
    }
    /* 每段尾帧锚定：槽位 6 预连线到隐藏「尾帧图」LoadImage（任意模式可点亮） */
    const lf = [...byId.values()].find((n) => n.title === "尾帧图");
    ok("尾帧图节点存在", !!lf);
    if (lf) {
        eq("尾帧图节点隐藏", lf.mode, 2);
        eq("尾帧图→槽位6 每段尾帧锚定", [10, 6],
            [wf.links.find((l) => l[0] === lf.outputs[0].link)[3], wf.links.find((l) => l[0] === lf.outputs[0].link)[4]]);
    }
    /* 分段视频不再单独打包：改由主节点「自动保存」存进项目文件夹（导演台段卡片预览） */
    ok("无遗留分段打包链", ![...byId.values()].some((n) => n.type === "CreateVideo" || n.type === "SaveVideo"));
    /* widgets_values 顺序与后端 schema 控件数一致（29 项 = 28 控件 + 种子 control 占 1 位） */
    eq("主节点 widgets 数量", h3n.widgets_values.length, 29);
    eq("百万像素浮点默认 0.5", h3n.widgets_values[1], 0.5);
    eq("主节点锚定加噪默认值", h3n.widgets_values[17], 0.0);
    eq("主节点自动保存默认值", h3n.widgets_values[19], "分段");
    eq("主节点尾部 生成模式/自动成片/导演台状态/一采编码", h3n.widgets_values.slice(25), ["文生视频", "开启", "", "标准"]);
    /* last_link_id / last_node_id 覆盖全部 */
    eq("last_node_id", wf.last_node_id, Math.max(...wf.nodes.map((n) => n.id)));
    eq("last_link_id", wf.last_link_id, Math.max(...wf.links.map((l) => l[0])));
}

console.log(fails ? `${fails} FAILURES` : "all js harness checks passed");
process.exit(fails ? 1 : 0);
