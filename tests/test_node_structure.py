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


def _install_stubs(with_audio_support=False):
    torch = types.ModuleType("torch")
    torch.cat = lambda xs, dim=0: FakeTensor([1])
    torch.std = lambda *a, **k: FakeTensor([1])
    sys.modules["torch"] = torch

    nodes_mod = types.ModuleType("nodes")
    nodes_mod.common_ksampler = lambda *a, **k: ({},)
    sys.modules["nodes"] = nodes_mod

    node_helpers = types.ModuleType("node_helpers")

    def conditioning_set_values(conditioning, values):
        return (conditioning[0], {**conditioning[1], **values})
    node_helpers.conditioning_set_values = conditioning_set_values
    sys.modules["node_helpers"] = node_helpers

    comfy = types.ModuleType("comfy")
    comfy_utils = types.ModuleType("comfy.utils")

    class ProgressBar:
        def __init__(self, total):
            pass

        def update(self, n=1):
            pass
    comfy_utils.ProgressBar = ProgressBar
    comfy.utils = comfy_utils
    sys.modules["comfy"] = comfy
    sys.modules["comfy.utils"] = comfy_utils

    samplers = types.ModuleType("comfy.samplers")

    class _KS:
        SAMPLERS = ["res_multistep"]
        SCHEDULERS = ["simple"]
    samplers.KSampler = _KS
    comfy.samplers = samplers
    sys.modules["comfy.samplers"] = samplers

    ldm = types.ModuleType("comfy.ldm")
    minimax = types.ModuleType("comfy.ldm.minimax")
    mm = types.ModuleType("comfy.ldm.minimax.model")
    mm.FRAME_PER_TOKEN = (1, 4, 4, 4, 4)
    mm.FRAME_RESCALE = 5.0 / 3.0
    if with_audio_support:
        mm._ref_t_span = lambda blk: 0.0
    minimax.model = mm
    ldm.minimax = minimax
    sys.modules["comfy.ldm"] = ldm
    sys.modules["comfy.ldm.minimax"] = minimax
    sys.modules["comfy.ldm.minimax.model"] = mm

    comfy_extras = types.ModuleType("comfy_extras")
    nmh = types.ModuleType("comfy_extras.nodes_minimax_h3")
    nmh.MiniMaxH3ImageToVideo = object
    nmh.MiniMaxH3ReferenceToVideo = object
    comfy_extras.nodes_minimax_h3 = nmh
    sys.modules["comfy_extras"] = comfy_extras
    sys.modules["comfy_extras.nodes_minimax_h3"] = nmh

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
    sys.modules["comfy_api"] = comfy_api
    sys.modules["comfy_api.latest"] = latest


def test_structure():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    schema = cls.define_schema()
    ids = [inp.id for inp in schema.inputs]
    for key in ["模型", "文本编码器", "视频VAE", "音频VAE", "宽度", "高度",
                "每段帧数", "引导帧数", "种子", "步数", "CFG", "采样器", "调度器",
                "断点续拍", "断点目录", "桥帧门控", "清晰度阈值", "回退上限",
                "审片模式", "重跑起始段",
                "首帧图片", "提示词组", "参考图片组", "参考视频组", "参考视频音轨组", "参考音频组"]:
        assert key in ids, f"missing input: {key}"
    by_id = {inp.id: inp for inp in schema.inputs}
    assert by_id["引导帧数"].kwargs.get("options") == ["5", "22", "39", "56"]
    assert by_id["种子"].kwargs.get("control_after_generate") is True
    assert by_id["断点续拍"].kwargs.get("options") == ["关闭", "自动续跑"]
    assert by_id["桥帧门控"].kwargs.get("options") == ["关闭", "标注", "自动回退"]
    assert by_id["审片模式"].kwargs.get("options") == ["关闭", "逐段确认"]
    assert by_id["审片模式"].kwargs.get("default") == "关闭"
    assert by_id["重跑起始段"].kwargs.get("default") == 0
    assert by_id["重跑起始段"].kwargs.get("min") == 0
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


def test_is_changed():
    import math
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    assert math.isnan(cls.IS_CHANGED(审片模式="逐段确认"))         # 审片激活 -> 强制执行
    assert math.isnan(cls.IS_CHANGED(断点续拍="自动续跑"))         # 断点激活 -> 强制执行
    assert cls.IS_CHANGED(审片模式="关闭", 断点续拍="关闭") == ""  # 默认 -> 可缓存
    assert cls.IS_CHANGED() == ""                                  # 无参调用兜底
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
    # 桥帧门控回退：偏移 5 token（=17 帧）+ 对应音频 token，切片长度不变
    kfb = plugin_nodes._tail_keyframe(video, audio, 22, True, back_tokens=5, back_audio=28)
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
              "宽度": 864, "高度": 480, "每段帧数": 124, "引导帧数": 22,
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
    print("PASS test_run_validation")


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    test_structure()
    test_capability_probe()
    test_bridge_fallback_probe()
    test_is_changed()
    test_tail_keyframe_slices()
    test_run_validation()
    print("all tests passed")
