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


def test_structure():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    cls = plugin_nodes.H3SeamlessChainSampler
    it = cls.INPUT_TYPES()
    for key in ["模型", "文本编码器", "视频VAE", "音频VAE", "提示词", "宽度", "高度",
                "每段帧数", "引导帧数", "种子", "步数", "CFG", "采样器", "调度器"]:
        assert key in it["required"], f"missing required input: {key}"
    assert "首帧图片" in it["optional"] and "参考图片" in it["optional"]
    assert it["required"]["引导帧数"][0] == [5, 22, 39, 56]
    assert cls.RETURN_TYPES == ("IMAGE", "AUDIO", "INT", "STRING", "IMAGE", "AUDIO")
    assert cls.RETURN_NAMES == ("图像", "音频", "帧率", "报告", "分段图像", "分段音频")
    assert cls.OUTPUT_IS_LIST == (False, False, False, False, True, True)
    assert cls.FUNCTION == "run" and cls.CATEGORY == "MiniMaxH3"
    print("PASS test_structure")


def test_capability_probe():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    assert plugin_nodes.KEYFRAME_AUDIO_SUPPORTED is False  # stub 未提供 _ref_t_span
    print("PASS test_capability_probe")


def test_tail_keyframe_slices():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    av = FakeNested(FakeTensor([1, 24, 37, 30, 54]),   # 124 帧段，37 video tokens
                    FakeTensor([1, 32, 2, 207], wav_len=207))
    kf = plugin_nodes._tail_keyframe(av, 22, with_audio=True)
    assert kf["resolved_frame_index"] == 0
    assert kf["latent"].shape[2] == 7      # 22 帧 = 7 video tokens
    assert kf["audio_latent"].shape[-1] == 37
    kf5 = plugin_nodes._tail_keyframe(av, 5, with_audio=True)
    assert kf5["latent"].shape[2] == 2 and kf5["audio_latent"].shape[-1] == 8
    kf39 = plugin_nodes._tail_keyframe(av, 39, with_audio=True)
    assert kf39["latent"].shape[2] == 12 and kf39["audio_latent"].shape[-1] == 65
    kf_noaudio = plugin_nodes._tail_keyframe(av, 22, with_audio=False)
    assert "audio_latent" not in kf_noaudio
    print("PASS test_tail_keyframe_slices")


class _Clip:
    def tokenize(self, s, **k):
        return ("tokens", s)

    def encode_from_tokens_scheduled(self, t):
        return ("cond", {})


def test_run_validation():
    from ComfyUI_H3_SeamlessChain import nodes as plugin_nodes
    node = plugin_nodes.H3SeamlessChainSampler()
    common = {"模型": None, "文本编码器": _Clip(), "视频VAE": None, "音频VAE": None,
              "宽度": 864, "高度": 480, "每段帧数": 124, "引导帧数": 22,
              "种子": 0, "步数": 25, "CFG": 1.0, "采样器": "res_multistep", "调度器": "simple"}
    try:
        node.run(**common, 提示词="  \n ")
        raise AssertionError("should reject empty prompts")
    except ValueError as e:
        assert "提示词" in str(e)
    try:
        node.run(**common, 提示词="a\nb",
                 首帧图片=FakeTensor([1, 480, 864, 3]),
                 参考图片=FakeTensor([2, 480, 864, 3]))
        raise AssertionError("should reject first_frame + ref_images")
    except ValueError as e:
        assert "不能同时连接" in str(e)
    print("PASS test_run_validation")


if __name__ == "__main__":
    _install_stubs(with_audio_support=False)
    test_structure()
    test_capability_probe()
    test_tail_keyframe_slices()
    test_run_validation()
    print("all tests passed")
