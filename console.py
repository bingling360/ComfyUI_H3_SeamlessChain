"""H3ChainConsole —— 长片生产控制台：存档管理 + 各段提示词 + 首帧/序章上传。

原「长片审片」侧栏面板节点化：画布上的控制台承担「导演台」职责——
- 提示词组：每段一框（单行），按行 join 成「提示词清单」传给采样器
  （改词=改自己控件，不再跨节点写）
- 存档名：读档下拉 / 时间戳新建 / 删除（前端 web/h3chain_console.js + routes.py）
- 首帧文件：上传图片（i2v 首段）或视频（序章，解码为 帧IMAGE + 音轨AUDIO）

节点无模型输入、不采样，仅整理数据与加载文件；分段浏览与重摇按钮在
节点面板 DOM 里（见 web/h3chain_console.js）。
"""

import os

from comfy_api.latest import io

UPLOAD_SUBDIR = "h3chain"  # 与 routes.py 的上传目录一致（input/h3chain/）


def join_prompts(group):
    """autogrow 提示词 dict -> 按行 join 的清单串（空值剔除，保持段序）。"""
    if not group:
        return ""
    def num(k):
        try:
            return int(str(k).rsplit("_", 1)[-1])
        except ValueError:
            return 0
    vals = [str(v).strip() for _, v in sorted(group.items(), key=lambda kv: num(kv[0]))
            if v is not None]
    return "\n".join(v for v in vals if v)


def split_prompts(listing):
    """提示词清单串 -> 段提示词列表（与采样器侧拆分同一规则，空行剔除）。"""
    return [ln.strip() for ln in str(listing or "").splitlines() if ln.strip()]


def load_upload(filename):
    """首帧文件名 -> (首帧 IMAGE, 序章帧 IMAGE, 序章音轨 AUDIO)。

    文件名是相对 input 的注记路径（h3chain/<名字>，routes.py 上传时写入）。
    图片 -> 仅首帧；视频 -> 序章帧 + 原声（无音轨时音轨为 None，序章按静音）。
    文件缺失抛 ValueError（上传后被手动删除属异常态，应报错而非静默忽略）。
    """
    fn = str(filename or "").strip()
    if not fn:
        return None, None, None
    path = _resolve_input(fn)
    ext = os.path.splitext(fn)[1].lower()
    if ext in (".mp4", ".mov", ".mkv", ".webm"):
        from .media import decode_av
        frames, wav, sr = decode_av(path)
        return None, frames, (wav if wav.shape[-1] > 0 and sr else None)
    img = _load_image(path)
    return img, None, None


def _resolve_input(fn):
    from folder_paths import get_annotated_filepath
    path = get_annotated_filepath(fn, default_dir=UPLOAD_SUBDIR)
    if not os.path.exists(path):
        raise ValueError(f"上传文件不存在：{fn}（请在控制台面板重新上传）")
    return path


def _load_image(path):
    import numpy
    import torch
    from PIL import Image, ImageOps

    img = ImageOps.exif_transpose(Image.open(path))
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = numpy.array(img).astype(numpy.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)  # [1,H,W,C]，与官方 LoadImage 同形


class H3ChainConsole(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="H3ChainConsole",
            display_name="H3 Chain Console (控制台)",
            category="MiniMaxH3",
            description="长片生产控制台：存档读档/新建/删除、各段提示词填写（一行一段传给采样器）、"
                        "上传首帧图片或序章视频；节点面板内浏览分段缩略图与视频、重摇此段、改词重跑。",
            inputs=[
                io.String.Input("存档名", default="",
                                tooltip="空=采样器按参数指纹自动命名。填名字即读档：已有存档自动续接，"
                                        "新名字=空白链。节点面板可下拉选择历史存档、一键新建（时间戳名）或删除"),
                io.String.Input("首帧文件", default="",
                                tooltip="上传的首帧图片或序章视频文件名（控制台面板「上传」按钮写入）。"
                                        "图片→「首帧」输出（i2v，fl2va UNET）；视频→「序章视频+序章音轨」输出。留空=不用"),
                io.Autogrow.Input("提示词组", optional=True,
                                  template=io.Autogrow.TemplatePrefix(
                                      input=io.String.Input("提示词",
                                                            placeholder="这一段的画面描述，一行一段按顺序续拍"),
                                      prefix="提示词_", min=1, max=64)),
            ],
            outputs=[
                io.Image.Output("首帧", tooltip="上传图片时的第 1 段起始帧（i2v）；未上传为空"),
                io.Image.Output("序章视频", tooltip="上传视频时的序章帧序列；未上传为空"),
                io.Audio.Output("序章音轨", tooltip="上传视频的原声；无音轨或未上传为空（序章按静音）"),
                io.String.Output("提示词清单", tooltip="各段提示词按行 join（空行剔除），接采样器「提示词清单」"),
                io.String.Output("存档名", tooltip="存档名 passthrough（空串=自动命名），接采样器「存档目录」"),
            ],
        )

    @classmethod
    def execute(cls, 存档名="", 首帧文件="", 提示词组=None):
        first, prologue, prologue_audio = load_upload(首帧文件)
        return {
            "首帧": first,
            "序章视频": prologue,
            "序章音轨": prologue_audio,
            "提示词清单": join_prompts(提示词组),
            "存档名": str(存档名 or "").strip(),
        }
