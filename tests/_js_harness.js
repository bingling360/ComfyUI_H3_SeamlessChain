/* h3_director.js 纯函数校验：node tests/_js_harness.js（跑完即删） */
const fs = require("fs");
let src = fs.readFileSync(require("path").join(__dirname, "../web/h3_director.js"), "utf8");
src = src
    .replace('import { app } from "/scripts/app.js";', "const app = { graph: null, registerExtension() {} };")
    .replace('import { api } from "/scripts/api.js";', "const api = { addEventListener() {}, fetchApi: async () => ({ ok: true }) };");
const mod = new Function(src + "\nreturn { resolveCanvas, snapFrames, matchCanvasCombo, remapOldWidgetValues, getDs, setDs, defaultDs };")();
let fails = 0;
function eq(name, got, want) {
    const ok = JSON.stringify(got) === JSON.stringify(want);
    if (!ok) { fails += 1; console.error(`FAIL ${name}: got ${JSON.stringify(got)} want ${JSON.stringify(want)}`); }
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
eq("v1->v2 ref_images 同步", ds.ref_images, ["a.png", "b.png"]);
ds.segments[0].seconds = 6.5;
ds.segments[0].refs = ["图片1"];
mod.setDs(node, ds);
const ds2 = mod.getDs(node);
eq("秒数回写", ds2.segments[0].seconds, 6.5);
eq("refs 回写", ds2.segments[0].refs, ["图片1"]);
eq("标签引用有效", ds2.ref_assets[0].label, "图片1");

console.log(fails ? `${fails} FAILURES` : "all js harness checks passed");
process.exit(fails ? 1 : 0);
