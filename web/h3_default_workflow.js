/* H3 长片导演台 · 配套默认工作流模板
 *
 * 打开导演台时若画布上没有 H3SeamlessChainSampler，可一键载入本模板：
 * - 模型加载器（ref2va UNET / Qwen3-VL CLIP / 视频+音频 VAE）+ 主节点 + 成片保存
 * - 「素材池 · 自动管理」节点组：首帧图 + 参考图×9 + 参考视频×3 + 参考音频×3
 *   全部预连线到主节点对应输入槽；不用时 mode=2(Never)+折叠 = 隐藏但连线常驻，
 *   导演台上传/删除素材时点亮/隐藏对应节点（连线与配置保留，不动态建线）
 * - 提示词不走画布（状态驱动：导演台 JSON 优先），故不预置提示词节点
 *
 * 注意：主节点 widgets_values 顺序须与 nodes.py define_schema 的 widget 顺序一致。
 */
(function () {
  "use strict";

  const H3 = "H3SeamlessChainSampler";

  // 主节点 widgets_values（顺序 = schema widget 顺序）：
  // 宽高比, 百万像素, 宽度, 高度, 每段时长, 引导帧数, 种子+control, 步数, CFG,
  // 采样器, 调度器, 自动存档, 存档目录, 桥帧门控, 清晰度阈值, 回退上限,
  // 接缝混合, 混合帧数, 锚定加噪, 审片模式, 自动保存, 重跑起始段, 生成模式, 导演台状态
  const H3_WIDGETS = [
    "16:9", "0.5", 864, 480, 5.0, "22",
    0, "fixed", 25, 1.0,
    "res_multistep", "simple",
    "关闭", "", "标注", 30.0, 34,
    "smoothstep", 6, 0.0,
    "关闭", "分段+成片", 0, "文生视频", "",
  ];

  function inp(name, type, link, extra) {
    return Object.assign({ name, type, link }, extra || {});
  }

  function refSlot(idx, type, link) {
    // autogrow 组输入槽：label=短名，name=组.槽名
    const group = type === "IMAGE" && idx.startsWith("参考图片") ? "参考图片组"
      : idx.startsWith("参考视频音轨") ? "参考视频音轨组"
        : idx.startsWith("参考视频") ? "参考视频组" : "参考音频组";
    return { label: idx, name: `${group}.${idx}`, shape: 7, type, link };
  }

  const nodes = [];
  const links = [];
  let linkId = 0;
  const L = (src, s, dst, d, type) => {
    linkId += 1;
    links.push([linkId, src, s, dst, d, type]);
    return linkId;
  };

  // ---- 模型加载 ----
  const unet = {
    id: 1, type: "UNETLoader", pos: [-720, 60], size: [640, 90], flags: {}, order: 0, mode: 0,
    inputs: [], outputs: [inp("MODEL", "MODEL", null)],
    properties: { "Node name for S&R": "UNETLoader" },
    widgets_values: ["minimax_h3_ref2va_pruned_int8_convrot.safetensors", "default"],
  };
  const clip = {
    id: 2, type: "CLIPLoader", pos: [-720, 200], size: [640, 120], flags: {}, order: 1, mode: 0,
    inputs: [], outputs: [inp("CLIP", "CLIP", null)],
    properties: { "Node name for S&R": "CLIPLoader" },
    widgets_values: ["qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "minimax", "default"],
  };
  const vaeV = {
    id: 3, type: "VAELoader", title: "视频 VAE", pos: [-720, 370], size: [640, 70], flags: {}, order: 2, mode: 0,
    inputs: [], outputs: [inp("VAE", "VAE", null)],
    properties: { "Node name for S&R": "VAELoader" },
    widgets_values: ["minimax_h3_video_vae_fp16.safetensors"],
  };
  const vaeA = {
    id: 4, type: "VAELoader", title: "音频 VAE", pos: [-720, 480], size: [640, 70], flags: {}, order: 3, mode: 0,
    inputs: [], outputs: [inp("VAE", "VAE", null)],
    properties: { "Node name for S&R": "VAELoader" },
    widgets_values: ["minimax_h3_audio_vae_fp32.safetensors"],
  };
  nodes.push(unet, clip, vaeV, vaeA);

  // ---- 主节点：预声明全部素材输入槽（连线常驻） ----
  const h3Inputs = [
    inp("模型", "MODEL", L(1, 0, 10, 0, "MODEL")),
    inp("文本编码器", "CLIP", L(2, 0, 10, 1, "CLIP")),
    inp("视频VAE", "VAE", L(3, 0, 10, 2, "VAE")),
    inp("音频VAE", "VAE", L(4, 0, 10, 3, "VAE")),
  ];
  const h3 = {
    id: 10, type: H3, title: "H3 Seamless Chain · 导演台主节点",
    pos: [40, 40], size: [720, 1020], flags: {}, order: 10, mode: 0,
    inputs: h3Inputs,
    outputs: [
      inp("图像", "IMAGE", null),
      inp("音频", "AUDIO", null),
      inp("帧率", "INT", null),
      inp("报告", "STRING", null),
      inp("分段图像", "IMAGE", null),
      inp("分段音频", "AUDIO", null),
    ],
    properties: { "Node name for S&R": H3 },
    widgets_values: H3_WIDGETS,
  };
  nodes.push(h3);

  // ---- 成片保存 ----
  const saver = {
    id: 11, type: "H3ChainSaver", title: "成片保存（自动落盘+画廊）",
    pos: [900, 120], size: [380, 220], flags: {}, order: 11, mode: 0,
    inputs: [
      inp("图像", "IMAGE", L(10, 0, 11, 0, "IMAGE")),
      inp("音频", "AUDIO", L(10, 1, 11, 1, "AUDIO")),
    ],
    outputs: [inp("成片文件名", "STRING", null)],
    properties: { "Node name for S&R": "H3ChainSaver" },
    widgets_values: [24, "h3_chain", 20],
  };
  nodes.push(saver);

  // ---- 素材池 · 自动管理（mode=2 隐藏 + 折叠，连线常驻） ----
  function hiddenNode(id, type, title, pos, widgets, outputs) {
    return {
      id, type, title, pos, size: [280, 300], flags: { collapsed: true }, order: 5, mode: 2,
      inputs: [], outputs, properties: { "Node name for S&R": type }, widgets_values: widgets,
    };
  }
  const imgOut = (link) => [
    inp("IMAGE", "IMAGE", link),
    inp("MASK", "MASK", null),
  ];

  // 首帧图（槽位 4）
  nodes.push(hiddenNode(20, "LoadImage", "首帧图", [40, 1160], ["", "image"],
    imgOut(L(20, 0, 10, 4, "IMAGE"))));
  h3Inputs.push(inp("首帧图片", "IMAGE", null));

  // 参考图·1..9（槽位 5..13）
  for (let i = 0; i < 9; i++) {
    const id = 21 + i;
    const col = i % 3, row = Math.floor(i / 3);
    const slot = 5 + i;
    const lid = L(id, 0, 10, slot, "IMAGE");
    nodes.push(hiddenNode(id, "LoadImage", `参考图·${i + 1}`,
      [380 + col * 320, 1160 + row * 340], ["", "image"], imgOut(lid)));
    h3Inputs.push(refSlot(`参考图片_${i}`, "IMAGE", lid));
  }

  // 参考视频×3 + 配套 GetVideoComponents（槽位 14..16 图像 / 17..19 音轨）
  for (let k = 0; k < 3; k++) {
    const lvId = 30 + k, gvcId = 33 + k;
    const imgLid = L(gvcId, 0, 10, 14 + k, "IMAGE");
    const audLid = L(gvcId, 1, 10, 17 + k, "AUDIO");
    nodes.push(hiddenNode(lvId, "LoadVideo", `参考视频·${k + 1}`,
      [1380, 1160 + k * 240], [""],
      [inp("video", "VIDEO", L(lvId, 0, gvcId, 0, "VIDEO"))]));
    nodes.push({
      id: gvcId, type: "GetVideoComponents", title: `拆分视频·${k + 1}`,
      pos: [1700, 1160 + k * 240], size: [280, 150], flags: { collapsed: true }, order: 6, mode: 2,
      inputs: [inp("video", "VIDEO", null)],
      outputs: [
        inp("images", "IMAGE", imgLid),
        inp("audio", "AUDIO", audLid),
        inp("fps", "FLOAT", null),
        inp("bit_depth", "INT", null),
      ],
      properties: { "Node name for S&R": "GetVideoComponents" }, widgets_values: [],
    });
    h3Inputs.push(refSlot(`参考视频_${k}`, "IMAGE", imgLid));
    h3Inputs.push(refSlot(`参考视频音轨_${k}`, "AUDIO", audLid));
  }

  // 参考音频×3（槽位 20..22）
  for (let k = 0; k < 3; k++) {
    const id = 36 + k;
    const lid = L(id, 0, 10, 20 + k, "AUDIO");
    nodes.push(hiddenNode(id, "LoadAudio", `参考音频·${k + 1}`,
      [2040, 1160 + k * 240], [""], [inp("audio", "AUDIO", lid), inp("name", "STRING", null)]));
    h3Inputs.push(refSlot(`参考音频_${k}`, "AUDIO", lid));
  }

  // ---- 使用说明 ----
  nodes.push({
    id: 40, type: "MarkdownNote", title: "导演台使用说明",
    pos: [40, 2320], size: [620, 320], flags: {}, order: 12, mode: 0,
    inputs: [], outputs: [], properties: {}, widgets_values: [
      "# H3 长片导演台 · 配套工作流\n\n" +
      "- 生成控制在左侧「长片导演台」侧栏：提示词/素材/参数一体化，无需手动连点节点\n" +
      "- 「素材池 · 自动管理」组的节点由导演台自动点亮/隐藏：**连线常驻，不用时只是隐藏**，请勿删除\n" +
      "- 默认 ref2va UNET（多参模式）；纯文生/首帧模式请在导演台切换，或把 UNETLoader 换成 fl2va 权重\n" +
      "- 每段时长/宽高比/百万像素/种子/步数在导演台右栏「链参数」；其余参数收在「⚙ 高级设置」",
    ],
  });

  window.H3_DEFAULT_WORKFLOW = {
    id: "h3-chain-director-default",
    revision: 0,
    last_node_id: 40,
    last_link_id: linkId,
    nodes,
    links,
    groups: [
      { id: 1, title: "模型加载", bounding: [-760, 0, 720, 620], color: "#3f789e", flags: {} },
      { id: 2, title: "导演台主链", bounding: [0, 0, 820, 1100], color: "#88A", flags: {} },
      {
        id: 3, title: "素材池 · 自动管理（导演台控制，勿删；隐藏=未使用）",
        bounding: [0, 1100, 2360, 1180], color: "#b58b2a", flags: {},
      },
    ],
    config: {},
    extra: { ds: { scale: 0.55, offset: [300, 80] } },
    version: 0.4,
  };
})();
