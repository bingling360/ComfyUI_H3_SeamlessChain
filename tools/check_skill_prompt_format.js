/* 校验：skill「h3-video-prompts」的示例文本能被 parseMasterPrompt 干净解析 */
const fs = require("fs");
const path = require("path");
const repo = "D:/h3长视频节点开发/ComfyUI_H3_SeamlessChain";
let src = fs.readFileSync(path.join(repo, "web/h3_director.js"), "utf8");
src = src
    .replace('import { app } from "/scripts/app.js";', "const app = { graph: null, registerExtension() {} };")
    .replace('import { api } from "/scripts/api.js";', "const api = { addEventListener() {}, fetchFetch: async () => ({}) };");
const mod = new Function("alert", "confirm", src + "\nreturn { parseMasterPrompt };")(() => {}, () => true);

const skill = fs.readFileSync("D:/h3长视频节点开发/.agents/skills/h3-video-prompts/SKILL.md", "utf8");
/* 抽取 ``` 围栏里的第一个多段示例（完整示例块） */
const fences = [...skill.matchAll(/```\n([\s\S]*?)```/g)].map((m) => m[1]);
const example = fences.find((t) => t.includes("【段1】") && t.includes("【段2】"));
if (!example) { console.error("FAIL: 未在 SKILL.md 中找到多段示例"); process.exit(1); }
const p = mod.parseMasterPrompt(example);
const problems = [];
if (p.segs.length !== 3) problems.push(`段数=${p.segs.length} 应为 3`);
if (p.notes.length) problems.push("notes: " + p.notes.join("；"));
const s1 = p.segs[0];
if (s1.scene !== "日系动画风格，黄昏的教室，暖橘色侧光斜照在课桌上") problems.push("段1场景错: " + s1.scene);
if (s1.seconds !== 5) problems.push("段1时长错: " + s1.seconds);
if (!s1.main.includes("「今天也早点回去吧。」")) problems.push("段1对白错: " + s1.main);
if (p.segs[2].music !== undefined) problems.push("段3未写配乐应为undefined");
if (!p.segs[2].main.includes("侧跟机位")) problems.push("段3主体错: " + p.segs[2].main);
/* 语法块（含 <占位符> 的那个）也应解析为 1 段；时长占位符不是数字，允许该条警告 */
const syntax = fences.find((t) => t.includes("<风格"));
const ps = mod.parseMasterPrompt(syntax);
const syntaxNotes = ps.notes.filter((n) => !n.includes("不是有效秒数"));
if (ps.segs.length !== 1 || syntaxNotes.length) problems.push("语法块解析异常: " + JSON.stringify(ps.notes));
if (problems.length) { console.error("FAIL:\n" + problems.join("\n")); process.exit(1); }
console.log("OK: skill 示例与语法块均通过真实解析器（3 段、零警告、字段全对位）");
