# ComfyUI_H3_SeamlessChain

MiniMax H3 多段视频**段间引导（无缝续拍）**单节点插件。

一个节点搞定：多段提示词 → 逐段采样 → 段间无缝续接（视频 + 音频双流）→ 裁剪拼接 → 输出完整 `IMAGE + AUDIO`。

## 原理（一句话）

生成第 N+1 段时，把第 N 段结尾 `context_frames` 帧从**采样输出的 latent 里直接切片**（不解码、不重编码 → 零颜色漂移），按官方 conditioning 协议（`minimax_keyframes`，锚 `resolved_frame_index=0`）钉在第 N+1 段头部，采样每步重注入，顺着上段尾部的运动轨迹继续画；解码后裁掉头部重叠桥再拼接 → 接缝处自然连贯。

## 特点

- **官方原生协议，零 monkey-patch**：conditioning / latent 构造直接调用官方 `MiniMaxH3ImageToVideo` / `MiniMaxH3ReferenceToVideo`，采样走官方 `common_ksampler`，抗 ComfyUI 升级。
- **零额外依赖**：requirements.txt 为空。只用 ComfyUI 自带的 torch 与 comfy 核心，不装 opencv / scenedetect / ffmpeg。
- **无缓存架构**：循环在单次执行内完成，段间 handoff 全是局部变量——旧方案「stale-cache 回放」类 bug 在此架构下不存在。
- **音频双流续接**：guide 同时钉入上段尾部的音频窗（需新版 ComfyUI，见下）。

## 安装

```bash
cd ComfyUI/custom_nodes
git clone <本插件地址> ComfyUI_H3_SeamlessChain
# 或直接把 ComfyUI_H3_SeamlessChain 文件夹拷进来
pip install -r ComfyUI_H3_SeamlessChain/requirements.txt   # 无依赖，可跳过
```
重启 ComfyUI。AutoDL 环境（`/root/miniconda3`）同样步骤，无需额外装包。

## 版本要求

- ComfyUI **≥ v0.30.0**（含官方 MiniMax H3 节点），否则插件不加载并在控制台提示。
- 段间引导携带**音频**锚定需要 ComfyUI 含 [PR #15439](https://github.com/Comfy-Org/ComfyUI/pull/15439)（2026-08-09 之后构建）。旧版本会自动降级为**仅视频**引导（报告中注明）；**r2v 链在旧版本上引导会与参考素材冲突失效，务必升级**。

## 使用（节点界面为中文）

1. 按官方 MiniMax H3 工作流加载模型：
   - **UNET**：t2v / i2v 用 `minimax_h3_fl2va_*`；r2v 用 `minimax_h3_ref2va_*`
   - UNET → 官方 **ModelSamplingMiniMaxH3**（shift video 12 / audio 3）→ 本节点「模型」
   - **CLIP**：type 必须选 `minimax`（Qwen3-VL）→ 本节点「文本编码器」
   - 视频 VAE（`minimax_h3_video_vae`）→「视频VAE」；音频 VAE（`minimax_h3_audio_vae`）→「音频VAE」
2. **每段提示词独立输入**：「提示词组」按段展开（提示词_1、提示词_2 …），可以直接写在框里，也可以接 `PrimitiveStringMultiline` 等外部节点；**一个输入框 = 一段视频**，段顺序按编号，点输入区旁的 + / - 增减段（最多 64 段）。
3. **参考素材（r2v 链）**：四组 autogrow 输入，用法与官方 Reference to Video 完全一致——
   - 「参考图片组」最多 **9 张**（`<Picture i>` 引用）
   - 「参考视频组」最多 **3 个**（`<Video k>` 引用）：接 `LoadVideo` 的 `frames` 输出
   - 「参考视频音轨组」：与同号参考视频配对（参考视频音轨_0 配 参考视频_0），接同一个 `LoadVideo` 的 `audio` 输出，自动获得 `<Audio j>` 标签
   - 「参考音频组」最多 **3 条**独立音频（配乐/音效，`<Audio j>` 引用）：接 `LoadAudio` / `LoadVideo`
   - 标签编号**按连接顺序从 1 数**（官方语义）；中间留空的输入框会被自动压实跳过，不会错位
   - 接了任一参考素材 → 整条链走 r2v，请换 `ref2va` UNET，且不要再接「首帧图片」
4. 输出：「图像」「帧率」「音频」接官方 `Create Video` → `Save Video`；「报告」可右键预览每段执行摘要。
5. **分段单独保存**：「分段图像」「分段音频」为逐段展开输出（`OUTPUT_IS_LIST`），再接一组 `Create Video` → `Save Video`，运行时每段各存一个视频文件（自动编号），内容与成品中该段的画面完全一致，方便逐段检查或自行剪辑；不需要时该支路删掉即可，不影响成品输出。

### 任务链自动判定（无需选 task_type）

| 连接 | 链路 | UNET |
|---|---|---|
| 什么都不接 | 纯 t2v | fl2va |
| 「首帧图片」 | 首段 i2v + 续段 t2v | fl2va |
| 任一「参考」组 | 每段 r2v（提示词用 `<Picture i>` / `<Video k>` / `<Audio j>` 引用） | ref2va |

### 关键参数（均为节点上的中文控件名）

- `每段帧数`：每段**可见**帧数 @24fps（124 ≈ 5 秒，模型训练范围约 124–362）。段 2 起实际采样帧数自动对齐到 `可见 + 引导帧数` 的 17k+5 网格。
- `引导帧数`：段间重叠桥，只能取 5 / 22 / 39 / 56（模型 17k+5 帧网格点）。越大衔接越顺、越慢越吃显存，默认 22。
- `种子`：每段自动 +1（段 i 用 `种子 + i`）。
- 默认采样参数：25 步、res_multistep + simple、CFG 1.0（蒸馏模型）。

## 示例工作流

- `example_workflows/h3_chain_t2v.json`：三段 t2v 续拍（UNET → SigmaShift → 本节点 → Create Video → Save Video），三段提示词各接一个 `PrimitiveStringMultiline`。
- `example_workflows/h3_chain_r2v_official_base.json`：**官方 r2v 模板骨架移植版（优先使用）**——直接以 ComfyUI 官方 `video_minimax_h3_r2v` 工作流为基底改造，模型加载 / CreateVideo / SaveVideo 节点保持官方原样序列化（含 cnr_id / models 元数据），UNET 直连本节点（与官方 r2v 模板一致，无 SigmaShift），并把官方示例参考图（红披风少年 / 机甲龙）沿用为三段提示词示例。节点上四组参考输入（图片×9 / 视频×3 / 视频音轨×3 / 音频×3）均已展开，「参考视频_0 + 参考视频音轨_0」预接了 `LoadVideo` 的 frames/audio 双输出演示接法；已含「分段保存」支路（每段单独存一个视频）。

## 已知边界（V1）

- 不含 fl2v 尾帧组、v2v / rv2v 源视频编辑（属内容生成方式扩展，非段间引导核心，可后续加）。
- 音频锚定为官方「从锚点向后展开」语义；有节拍的音乐接缝处如察觉轻微不自然，可尝试把 `引导帧数` 提到 39 或 56。

## 参数与铁律速查（来自官方）

- 引导帧数必须踩 5 / 22 / 39 / 56 网格点，其他值直接报错。
- CLIP type 必须选 `minimax`（Qwen3-VL）。
- t2v / i2v → fl2va UNET；r2v → ref2va UNET，两种 UNET 别混接。
