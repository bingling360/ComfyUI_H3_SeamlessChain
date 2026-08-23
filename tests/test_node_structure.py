"""stub 环境下验证节点结构、guide 切片与输入校验：python tests/test_node_structure.py"""
import os
import sys
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


class FakeTensor:
    def __init__(self, shape, wav_len=0):
        self.shape = list(shape)
        self.wav_len = wav_len

    def __getitem__(self, s):
        shape = list(self.shape)
        if isinstance(s, tuple):
            idxs = list(s)
            if Ellipsis in idxs:
                pos = idxs.index(Ellipsis)
                fill = len(shape) - (len(idxs) - 1)
                idxs = idxs[:pos] + [slice(None)] * fill + idxs[pos + 1:]
            for i, idx in enumerate(idxs):
                if isinstance(idx, slice):
                    start, stop, step = idx.indices(shape[i])
                    shape[i] = len(range(start, stop, step))
                else:
                    shape[i] = 1
        return FakeTensor(shape, self.wav_len)

    def clone(self):
        return FakeTensor(list(self.shape), self.wav_len)


class FakeNested:
    def __init__(self, video, audio):
        self.video = video
        self.audio = audio

    def unbind(self):
        return (self.video, self.audio)

    @property
    def tensors(self):
        return (self.video, self.audio)


_TORCH_STUBBED = False


def _setdefault_module(name, module):
    """只在缺失时安装 stub：保持对象身份稳定（nodes.py 导入时绑定的模块引用
    与 sys.modules 条目必须同一对象，否则测试补丁打在另一份上）。"""
    if name not in sys.modules:
        sys.modules[name] = module
    return sys.modules[name]


def _install_stubs(with_audio_support=False):
    global _TORCH_STUBBED
    try:
        import torch as _real_torch
        # 真 torch 带 __version__；本文件装的假 stub 不带。重复调用时
        # sys.modules 里已是 stub，不能用 import 成功与否判断真假。
        # 有真 torch（AutoDL/ComfyUI 环境）就不装假 stub——否则合集运行时
        # sys.modules 污染会让 test_metrics/test_qc/test_refine/test_smart_cut
        # 等真 torch 测试拿到假模块集体翻车
        _TORCH_STUBBED = not hasattr(_real_torch, "__version__")
    except ImportError:
        _TORCH_STUBBED = True
    if _TORCH_STUBBED:
        torch = types.ModuleType("torch")
        torch.cat = lambda xs, dim=0: FakeTensor([1])
        torch.std = lambda *a, **k: FakeTensor([1])
        # torch.nn.functional 占位（seam_doctor 顶层 import，stub 下只需可导入）
        torch_nn = types.ModuleType("torch.nn")
        torch_nn_f = types.ModuleType("torch.nn.functional")
        torch_nn_f.interpolate = lambda *a, **k: FakeTensor([1])
        torch_nn_f.conv2d = lambda *a, **k: FakeTensor([1])
        torch_nn.functional = torch_nn_f
        torch.nn = torch_nn
        _setdefault_module("torch", torch)
        _setdefault_module("torch.nn", torch_nn)
        _setdefault_module("torch.nn.functional", torch_nn_f)

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.common_ksampler = lambda *a, **k: ({},)
    _setdefault_module("nodes", nodes_mod)

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values):
        return (conditioning[0], {**conditioning[1], **values})
    node_helpers.conditioning_set_values = conditioning_set_values
    _setdefault_module("node_helpers", node_helpers)

    folder_paths_mod = types.SimpleNamespace(
        get_annotated_filepath=lambda name: name,
        get_output_directory=lambda: "output",
    )
    _setdefault_module("folder_paths", folder_paths_mod)

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, n=1):
            pass
    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils
    comfy = _setdefault_module("comfy", comfy)
    _setdefault_module("comfy.utils", comfy_utils)

    samplers = types.ModuleType("comfy.samplers")

    class _KS:
        SAMPLERS = ["res_multistep"]
        SCHEDULERS = ["simple"]
    samplers.KSampler = _KS
    comfy.samplers = samplers
    _setdefault_module("comfy.samplers", samplers)

    ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax")
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    mm.FRAME_RESCALE = 5.0 / 3.0
    if with_audio_support:
        mm._ref_t_span = lambda blk: 0.0
    minimax.model = mm
    ldm.minimax = minimax
    _setdefault_module("comfy.ldm", ldm)
    _setdefault_module("comfy.ldm.minimax", minimax)
    _setdefault_module("comfy.ldm.minimax.model", mm)

    comfy_extras = types.ModuleType("comfy_extras")
    nmh = types.ModuleType("comfy_extras.nodes_minimax_h3")
    nmh.MiniMaxH3ImageToVideo = object
    nmh.MiniMaxH3ReferenceToVideo = object
    comfy_extras.nodes_minimax_h3 = nmh
    _setdefault_module("comfy_extras", comfy_extras)
    _setdefault_module("comfy_extras.nodes_minimax_h3", nmh)

    class _Any:
        def __init__(self, *a, **k):
            self.args = a
            self.id = a[0] if a else k.get("id")
            self.kwargs = k
            for key, v in k.items():
                setattr(self, key, v)
            self.is_output_list = bool(k.get("is_output_list"))

        def __getitem__(self, i):
            return self.args[i]

    class io_mod:
        ComfyNode = object
        Schema = _Any
        NodeOutput = _Any

        class Autogrow:
            Input = _Any

            class TemplatePrefix:
                def __init__(self, *a, **k):
                    self.args, self.kwargs = a, k

    for name in ["Model", "Clip", "Vae", "Int", "Float", "Combo", "String",
                 "Image", "Audio", "Latent", "Conditioning"]:
        setattr(io_mod, name, type(name, (), {"Input": _Any, "Output": _Any}))

    comfy_api = types.ModuleType("comfy_api")
    latest = types.ModuleType("comfy_api.latest")
    latest.io = io_mod
    latest.ComfyExtension = object
    comfy_api.latest = latest
    _setdefault_module("comfy_api", comfy_api)
    _setdefault_module("comfy_api.latest", latest)


def test_structure():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    schema = cls.define_schema()
    ids = [inp.id for inp in schema.inputs]
    for key in ["模型", "文本编码器", "视频VAE", "音频VAE", "宽高比", "百万像素", "宽度", "高度",
                "每段时长", "引导帧数", "种子", "步数", "CFG", "采样器", "调度器",
                "自动存档", "存档目录", "桥帧门控", "清晰度阈值", "回退上限", "锚定加噪",
                "审片模式", "自动保存", "自动成片", "重跑起始段",
                "接缝重摇", "重摇阈值", "重摇上限", "递减锚定",
                "首帧图片", "尾帧图片", "每段尾帧锚定", "起始视频", "起始视频音轨",
                "提示词组",
                "参考图片组", "参考视频组", "参考视频音轨组", "参考音频组"]:
        assert key in ids, f"missing input: {key}"
    by_id = {inp.id: inp for inp in schema.inputs}
    assert by_id["宽高比"].kwargs.get("options") == ["自定义", "21:9", "16:9", "9:16", "4:3", "3:4", "1:1"]
    assert by_id["宽高比"].kwargs.get("default") == "16:9"
    # 百万像素：官方 Resolution Selector 同款浮点箭头控件（0.1–2.0 步进 0.1，共 20 档）
    assert by_id["百万像素"].kwargs.get("default") == 0.5
    assert by_id["百万像素"].kwargs.get("min") == 0.1
    assert by_id["百万像素"].kwargs.get("max") == 2.0
    assert by_id["百万像素"].kwargs.get("step") == 0.1
    assert "options" not in by_id["百万像素"].kwargs
    assert by_id["每段时长"].kwargs.get("default") == 5.0
    assert by_id["引导帧数"].kwargs.get("options") == ["5", "22", "39", "56"]
    assert by_id["种子"].kwargs.get("control_after_generate") is True
    assert by_id["自动存档"].kwargs.get("options") == ["关闭", "自动存档"]
    assert by_id["自动存档"].kwargs.get("advanced") is True   # 已并入「自动保存」分段档，仅兼容旧工作流
    assert by_id["自动保存"].kwargs.get("options") == ["关闭", "分段"]
    assert by_id["自动保存"].kwargs.get("default") == "分段"
    assert by_id["自动成片"].kwargs.get("options") == ["关闭", "开启"]
    assert by_id["自动成片"].kwargs.get("default") == "开启"
    assert by_id["自动成片"].kwargs.get("advanced") is True
    assert "提示词清单" not in ids                   # 控制台已回退，清单输入随之移除
    assert "接缝混合" not in ids                     # 旧控件已随「接缝处理」一并剔除
    assert "接缝处理" not in ids                     # 潜空间精修后处理已剔除（第一阶段）
    assert "精修强度" not in ids and "精修窗口" not in ids and "混合帧数" not in ids
    assert "智能切镜" not in ids and "切镜最多丢帧" not in ids
    assert "全链丢弃预算" not in ids and "自适应精修" not in ids
    assert by_id["桥帧门控"].kwargs.get("options") == ["关闭", "标注", "自动回退"]
    # 保留的核心接缝控件：桥帧门控 / 重摇（排除抽卡坏段）+ 下阶段技术储备（锚定/递减锚定）
    assert by_id["锚定加噪"].kwargs.get("default") == 0.0
    assert by_id["锚定加噪"].kwargs.get("min") == 0.0 and by_id["锚定加噪"].kwargs.get("max") == 0.5
    assert by_id["接缝重摇"].kwargs.get("options") == ["关闭", "自动"]
    assert by_id["接缝重摇"].kwargs.get("default") == "自动"
    assert by_id["重摇阈值"].kwargs.get("default") == 0.06
    assert by_id["重摇上限"].kwargs.get("default") == 1
    assert by_id["重摇上限"].kwargs.get("min") == 0 and by_id["重摇上限"].kwargs.get("max") == 3
    assert by_id["递减锚定"].kwargs.get("options") == ["关闭", "0.3", "0.5", "0.7"]
    assert by_id["递减锚定"].kwargs.get("default") == "关闭"
    assert "段内分片" not in ids                        # 已回退
    assert by_id["审片模式"].kwargs.get("options") == ["关闭", "逐段确认"]
    assert by_id["审片模式"].kwargs.get("default") == "关闭"
    assert by_id["重跑起始段"].kwargs.get("default") == 0
    assert by_id["重跑起始段"].kwargs.get("min") == 0
    assert ids.index("递减锚定") < ids.index("生成模式")  # 接缝区控件在生成模式之前（widget 区尾部）
    outs = [(o.id, o.is_output_list) for o in schema.outputs]
    assert outs == [("图像", False), ("音频", False), ("帧率", False), ("报告", False),
                    ("分段图像", True), ("分段音频", True)]
    print("PASS test_structure")


def test_capability_probe():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    assert plugin_nodes.KEYFRAME_AUDIO_SUPPORTED is False  # stub 未提供 _ref_t_span
    print("PASS test_capability_probe")


class _FakeMask:
    def __init__(self, n):
        self.n = n

    def __invert__(self):
        return self

    def sum(self):
        return self.n


def test_bridge_fallback_probe():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    mm = sys.modules["comfy.ldm.minimax.model"]
    if _TORCH_STUBBED:
        # 假 torch 没有 zeros，补一个；真 torch 自带且不能覆盖（会污染后续真 torch 测试）
        sys.modules["torch"].zeros = lambda *a, **k: FakeTensor(list(a))

    def layout_with(cond_rows):
        return lambda *a, **k: types.SimpleNamespace(img_update=_FakeMask(cond_rows))

    # 旧协议：keyframe cond 行数恒为单帧 16 行 -> 降级为单帧桥
    mm.PackedLayout = layout_with(16)
    plugin_nodes._full_bridge_cache = None
    assert plugin_nodes.full_bridge_supported() is False
    kf = plugin_nodes._tail_keyframe(FakeTensor([1, 24, 37, 30, 54]),
                                     FakeTensor([1, 32, 2, 207], 207), 22, True,
                                     full_bridge=False)
    assert kf["latent"].shape[2] == 1 and "audio_latent" not in kf

    # 新协议：2 token 得 32 行 -> 完整桥
    mm.PackedLayout = layout_with(32)
    plugin_nodes._full_bridge_cache = None
    assert plugin_nodes.full_bridge_supported() is True

    # PackedLayout 缺失（无 ComfyUI 环境）-> 安全降级
    del mm.PackedLayout
    plugin_nodes._full_bridge_cache = None
    assert plugin_nodes.full_bridge_supported() is False
    plugin_nodes._full_bridge_cache = None
    print("PASS test_bridge_fallback_probe")


def test_anchor_noise():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    cond = ("t", {"k": 1})
    out0 = plugin_nodes._apply_anchor_noise(cond, 0.0)
    assert out0[1] == {"k": 1}                          # 0 = 关闭，cond 原样
    out = plugin_nodes._apply_anchor_noise(cond, 0.2)
    assert out[1]["minimax_visual_cond_noise_aug"] == 0.8   # SkyReels 同值
    assert out[1]["minimax_audio_cond_noise_aug"] == 0.9    # 音频加噪减半
    assert out[1]["k"] == 1                                 # 原键保留
    schema = cls.define_schema()
    aug_input = {inp.id: inp for inp in schema.inputs}["锚定加噪"]
    assert aug_input.kwargs.get("default") == 0.0 and aug_input.kwargs.get("max") == 0.5
    print("PASS test_anchor_noise")


def test_is_changed():
    import math
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    assert math.isnan(cls.IS_CHANGED(审片模式="逐段确认"))         # 审片激活 -> 强制执行
    assert math.isnan(cls.IS_CHANGED(自动存档="自动存档"))         # 存档激活 -> 强制执行
    assert math.isnan(cls.IS_CHANGED(自动存档="自动续跑"))         # 旧工作流遗留值仍激活
    assert math.isnan(cls.IS_CHANGED(审片模式="关闭", 自动存档="关闭"))  # 自动保存默认开 -> 强制执行
    assert math.isnan(cls.IS_CHANGED(审片模式="关闭", 自动存档="关闭", 自动保存="分段"))  # 分段档也强制重跑
    assert cls.IS_CHANGED(审片模式="关闭", 自动存档="关闭", 自动保存="关闭") == ""  # 全关 -> 可缓存
    assert math.isnan(cls.IS_CHANGED())                           # 无参调用兜底（自动保存默认开）
    print("PASS test_is_changed")


def test_tail_keyframe_slices():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    video = FakeTensor([1, 24, 37, 30, 54])            # 124 帧段，37 video tokens
    audio = FakeTensor([1, 32, 2, 207], wav_len=207)
    kf = plugin_nodes._tail_keyframe(video, audio, 22, True)
    assert kf["resolved_frame_index"] == 0
    assert kf["latent"].shape[2] == 7                  # 22 帧 = 7 video tokens
    assert kf["audio_latent"].shape[-1] == 37
    kf5 = plugin_nodes._tail_keyframe(video, audio, 5, True)
    assert kf5["latent"].shape[2] == 2 and kf5["audio_latent"].shape[-1] == 8
    kf39 = plugin_nodes._tail_keyframe(video, audio, 39, True)
    assert kf39["latent"].shape[2] == 12 and kf39["audio_latent"].shape[-1] == 65
    kf_noaudio = plugin_nodes._tail_keyframe(video, audio, 22, False)
    assert "audio_latent" not in kf_noaudio
    # 末端对齐：end_tokens=30（=102 帧边界）时锚定窗末端与 kept 末端重合，窗宽不变
    kfb = plugin_nodes._tail_keyframe(video, audio, 22, True, end_tokens=30)
    assert kfb["latent"].shape[2] == 7 and kfb["audio_latent"].shape[-1] == 37
    print("PASS test_tail_keyframe_slices")


class _Clip:
    def tokenize(self, s, **k):
        return ("tokens", s)

    def encode_from_tokens_scheduled(self, t):
        return ("cond", {})


def test_run_validation():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    common = {"模型": None, "文本编码器": _Clip(), "视频VAE": None, "音频VAE": None,
              "宽高比": "自定义", "百万像素": "0.5", "宽度": 864, "高度": 480, "每段时长": 5.0,
              "引导帧数": 22,
              "种子": 0, "步数": 25, "CFG": 1.0, "采样器": "res_multistep", "调度器": "simple"}
    try:
        cls.execute(**common, 提示词组={"提示词_1": "  ", "提示词_2": ""})
        raise AssertionError("should reject empty prompts")
    except ValueError as e:
        assert "提示词" in str(e)
    try:
        cls.execute(**common, 提示词组={"提示词_1": "a", "提示词_2": "b"},
                    首帧图片=FakeTensor([1, 480, 864, 3]),
                    参考图片组={"参考图片_0": FakeTensor([1, 480, 864, 3])})
        raise AssertionError("should reject first_frame + refs")
    except ValueError as e:
        assert "不能同时" in str(e)
    try:
        cls.execute(**common, 提示词组={"提示词_1": "a", "提示词_2": "b"},
                    首帧图片=FakeTensor([1, 480, 864, 3]),
                    起始视频=FakeTensor([124, 480, 864, 3]))
        raise AssertionError("should reject prologue + first_frame")
    except ValueError as e:
        assert "不能同时" in str(e)
    print("PASS test_run_validation")


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    test_structure()
    test_capability_probe()
    test_bridge_fallback_probe()
    test_anchor_noise()
    test_is_changed()
    test_tail_keyframe_slices()
    test_run_validation()
    print("all tests passed")
