"""潜空间放大二采单测（真 torch + einops）：python tests/test_upscale.py

覆盖：parse_state 参数归一化、二采参数指纹与重做判定（pending_slots /
_record_valid / base_hash）、latent 放大数学（target_hw 偶数对齐 /
resize_latent_bilinear）、2D/3D 放大网络前向形状与权重加载（架构不匹配
明确报错 / MODEL_CACHE 缓存 / _detect_arch 结构推断）、upseg_* 存档联动
（save_upsegment 落盘回读 / truncate 清扫 / projects.upscale_reset）。
run_pass 本体依赖 ComfyUI 运行环境，不在本测范围（异常兜底在 nodes.py
调用点：任何失败降级为报告，不丢基础链产物）。
注意不能用 test_node_structure 的 stub（其 torch 是假的）——本测需要真
torch 切片与前向；包 __init__ 在无 ComfyUI 环境下自行降级（静默导入）。
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import types

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))  # 插件父目录（包名可导入）
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # tests 目录

try:
    import torch
    import einops  # noqa: F401  （upscale_net 顶层依赖，导入即验证）
    if not hasattr(torch, "rand"):
        # 合集运行时 test_node_structure 的假 torch stub 可能已在 sys.modules——视为无 torch
        raise ImportError
except ImportError:
    torch = None

if torch is not None:
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        from ComfyUI_H3_SeamlessChain import checkpoint
        from ComfyUI_H3_SeamlessChain import upscale
        from ComfyUI_H3_SeamlessChain import upscale_net
else:
    if "pytest" in sys.modules:
        import pytest
        pytest.skip("需要真 torch（stub/无 torch 环境跳过）", allow_module_level=True)


@contextlib.contextmanager
def _env():
    """临时 output + models 目录（folder_paths stub，含 latent_upscale_models 注册所需键）。"""
    out = tempfile.mkdtemp()
    models = os.path.join(out, "models")
    lups = os.path.join(models, "latent_upscale_models")
    os.makedirs(lups)
    folders = {}
    sys.modules["folder_paths"] = types.SimpleNamespace(
        folder_names_and_paths=folders,
        models_dir=models,
        add_model_folder_path=lambda name, path: folders.setdefault(name, ([path], set())),
        get_folder_paths=lambda name: folders.get(name, ([], set()))[0],
        get_output_directory=lambda: out,
    )
    try:
        yield out, lups
    finally:
        del sys.modules["folder_paths"]
        # 插件模块一并出缓存：upscale_net 的 _FOLDER_REGISTERED / MODEL_CACHE
        # 挂模块全局，不清的话第二个用例拿到旧临时目录
        for k in [k for k in sys.modules if k.startswith("ComfyUI_H3_SeamlessChain")]:
            del sys.modules[k]
        shutil.rmtree(out, ignore_errors=True)


# ---- 状态解析与指纹（纯逻辑） ----

def test_parse_state_off():
    for ds in (None, {}, {"upscale": None}, {"upscale": {}},
               {"upscale": {"mode": "关闭"}}, {"upscale": {"mode": " off "}},
               {"upscale": {"mode": "跟随生成", "on": False}}):
        assert upscale.parse_state(ds) is None, ds


def test_parse_state_normalize():
    cfg = upscale.parse_state({"upscale": {"mode": "乱写"}})   # 非法模式回落跟随生成
    assert cfg["mode"] == "跟随生成" and cfg["include"] is None
    assert cfg["model"] == "" and cfg["arch"] == "2D"
    assert cfg["scale"] == 2.0 and cfg["denoise"] == 0.45
    assert cfg["steps"] == 15 and cfg["cfg"] == 1.0
    assert cfg["precision"] == "fp32"

    full = upscale.parse_state({"upscale": {
        "mode": "手动选择", "model": " up2d.pth ", "arch": "3d", "scale": "9",
        "denoise": "0.01", "steps": "0", "cfg": "-5", "precision": "x",
        "include": [2, "1", 1.0, "bad", None]}})
    assert full["mode"] == "手动选择"
    assert full["model"] == "up2d.pth"          # strip 空白
    assert full["arch"] == "3D"                 # 大小写归一
    assert full["scale"] == 4.0                 # 钳上限
    assert full["denoise"] == 0.05              # 钳下限
    assert full["steps"] == 1
    assert full["cfg"] == 0.0
    assert full["precision"] == "fp32"          # 非法精度回落
    assert full["include"] == [1, 2]            # 去重排序，垃圾项剔除

    nan = upscale.parse_state({"upscale": {"mode": "跟随生成", "scale": float("nan"),
                                           "denoise": float("nan")}})
    assert nan["scale"] == 2.0 and nan["denoise"] == 0.45   # NaN 回落默认


def test_params_hash():
    a = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
    b = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                         "include": [0, 3]}})
    h = upscale.params_hash(a)
    assert len(h) == 8
    assert h == upscale.params_hash(b)          # mode/include 不进指纹（不影响单段输出）
    assert h == upscale.params_hash(upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}}))
    for key, val in (("denoise", 0.6), ("scale", 1.5), ("steps", 20), ("cfg", 2.0),
                     ("arch", "3D"), ("precision", "fp16"), ("model", "other.pth")):
        c = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth", key: val}})
        assert h != upscale.params_hash(c), key


def test_base_hash():
    mf = {"prompt_hashes": ["h0", "h1"], "seeds": [10, 20]}
    assert upscale.base_hash(mf, 0) == "h0|10"
    assert upscale.base_hash(mf, 1) == "h1|20"
    assert upscale.base_hash(mf, 9) == "|"       # 越界=双空串


# ---- 重做判定 ----

def test_pending_slots():
    with tempfile.TemporaryDirectory() as root:
        cfg = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
        ph = upscale.params_hash(cfg)
        mf = {"done": 3, "total": 4, "prompt_hashes": ["a", "b", "c"],
              "seeds": [1, 2, 3], "upscale": {"segs": [None, None, None]}}
        assert upscale.pending_slots(mf, root, cfg) == [0, 1, 2]   # 无记录全待做（done 截断）

        # 段0 有效记录：hash/base_hash 匹配且 mp4/last 文件在
        files0 = checkpoint.upseg_paths(root, 0)
        for k in ("mp4", "last"):
            open(os.path.join(root, files0[k]), "wb").close()
        mf["upscale"]["segs"][0] = {"hash": ph, "base_hash": "a|1", "done": True,
                                    "files": files0}
        assert upscale.pending_slots(mf, root, cfg) == [1, 2]

        # 段1 记录 hash 不匹配（二采参数变过）
        mf["upscale"]["segs"][1] = {"hash": "deadbeef", "base_hash": "b|2", "done": True,
                                    "files": checkpoint.upseg_paths(root, 1)}
        # 段2 base_hash 不匹配（基础链改词/换种子）
        mf["upscale"]["segs"][2] = {"hash": ph, "base_hash": "X|9", "done": True,
                                    "files": checkpoint.upseg_paths(root, 2)}
        assert upscale.pending_slots(mf, root, cfg) == [1, 2]

        # 段0 mp4 被删 -> 记录失效
        os.remove(os.path.join(root, files0["mp4"]))
        assert upscale.pending_slots(mf, root, cfg) == [0, 1, 2]

        # 手动模式：只算勾选段（含勾选但无记录/失效的）
        cfg2 = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                                "include": [1]}})
        assert upscale.pending_slots(mf, root, cfg2) == [1]
        cfg3 = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                                "include": []}})
        assert upscale.pending_slots(mf, root, cfg3) == []


# ---- latent 放大数学 ----

def test_target_hw():
    assert upscale.target_hw(24, 40, 2.0) == (48, 80)
    assert upscale.target_hw(25, 41, 2.0) == (50, 82)    # round 后取偶（像素 32 对齐）
    assert upscale.target_hw(24, 40, 1.0) == (24, 40)    # 偶尺寸恒等
    assert upscale.target_hw(25, 41, 1.0) == (26, 42)    # 奇 latent 补偶
    assert upscale.target_hw(1, 1, 1.5) == (2, 2)        # 最小 2


def test_resize_latent_bilinear():
    z = torch.randn(1, 24, 7, 4, 6)
    assert upscale.resize_latent_bilinear(z, 4, 6) is z             # 同尺寸原样返回
    out = upscale.resize_latent_bilinear(z, 8, 12)
    assert out.shape == (1, 24, 7, 8, 12)                           # T/C 不变
    assert not torch.equal(out, z)


def _tiny2d(temporal=0):
    return upscale_net.VideoLatentResizer(in_channels=24, in_blocks=1, out_blocks=1,
                                          channels=32, dropout=0.0, attn=False,
                                          temporal_every=temporal, temporal_kernel=3)


def _tiny3d(temporal=0):
    return upscale_net.LatentResizer3D(in_channels=24, in_blocks=1, out_blocks=1,
                                       channels=32, dropout=0.0, attn=False,
                                       temporal_every=temporal, temporal_kernel=3)


def test_network_forward_2d():
    x = torch.randn(1, 24, 7, 4, 6)
    for net in (_tiny2d(0), _tiny2d(2)):
        y = net(x, scale=2.0, target_hw=(8, 12))
        assert y.shape == (1, 24, 7, 8, 12)             # T 不变，H/W ×2
        y2 = net(x, scale=1.5)
        assert y2.shape == (1, 24, 7, 6, 9)             # scale 路径（round(4*1.5)=6）
    # 无时序块的同尺寸直通；带时序块的变体无捷径恒走全网——生产路径
    # upscale_video 先判同尺寸短路，不会把同尺寸喂给网络
    y0 = _tiny2d(0)(x, scale=1.0, target_hw=(4, 6))
    assert y0.shape == x.shape and torch.equal(y0, x)


def test_network_forward_3d():
    x = torch.randn(1, 24, 7, 4, 6)
    for net in (_tiny3d(0), _tiny3d(2)):
        y = net(x, scale=2.0, target_size=(7, 8, 12))
        assert y.shape == (1, 24, 7, 8, 12)
        y2 = net(x, scale=2.0)                          # scale 路径放大 T 会被拒？——
        # 3D 的 scale 路径按 x.shape[-3:] 全乘（含 T）：官方 3D 网络从未走该分支
        # （调用方恒传 target_size 钉死 T），这里只验证不抛错且 C 不变
        assert y2.shape[1] == 24
        assert net(x, scale=1.0, target_size=(7, 4, 6)) is x


def test_upscale_video_shapes():
    x = torch.randn(1, 24, 7, 4, 6)
    # scale=1.0 且偶尺寸：原样克隆（纯二采不放大）
    same = upscale.upscale_video(x, _tiny2d(), 1.0)
    assert same.shape == x.shape and torch.equal(same, x)
    assert same is not x                                   # 克隆不共享存储
    # 2D / 3D 主干放大：T 不变，H/W 偶数对齐
    for net, arch in ((_tiny2d(), "2D"), (_tiny3d(), "3D")):
        y = upscale.upscale_video(x, net, 2.0, arch)
        assert y.shape == (1, 24, 7, 8, 12), arch
        assert y.dtype == torch.float32                    # 输出统一 float32


def test_load_model_and_cache():
    with _env() as (out, lups):
        m2 = _tiny2d()
        torch.save(m2.state_dict(), os.path.join(lups, "tiny2d.pth"))
        m3 = _tiny3d()
        torch.save(m3.state_dict(), os.path.join(lups, "tiny3d.pth"))

        net = upscale_net.load_model("tiny2d.pth", "cpu", "fp32", "2D")
        assert isinstance(net, upscale_net.VideoLatentResizer)
        y = net(torch.randn(1, 24, 7, 4, 6), scale=2.0, target_hw=(8, 12))
        assert y.shape == (1, 24, 7, 8, 12)                # 加载后前向可用
        assert upscale_net.load_model("tiny2d.pth", "cpu", "fp32", "2D") is net  # 缓存命中
        fp16 = upscale_net.load_model("tiny2d.pth", "cpu", "fp16", "2D")
        assert fp16 is not net and fp16.resizer.conv_in.weight.dtype == torch.float16
        net3 = upscale_net.load_model("tiny3d.pth", "cpu", "fp32", "3D")
        assert isinstance(net3, upscale_net.LatentResizer3D)

        # 架构不匹配 -> 明确 ValueError（面板选错架构的引导文案）
        try:
            upscale_net.load_model("tiny2d.pth", "cpu", "fp32", "3D")
            raise AssertionError("应抛架构不匹配")
        except ValueError as e:
            assert "2D 残差" in str(e) and "3D" in str(e)
        try:
            upscale_net.load_model("tiny3d.pth", "cpu", "fp32", "2D")
            raise AssertionError("应抛架构不匹配")
        except ValueError as e:
            assert "纯 3D" in str(e)

        # 文件不存在 -> FileNotFoundError
        try:
            upscale_net.load_model("nope.pth", "cpu", "fp32", "2D")
            raise AssertionError("应抛 FileNotFoundError")
        except FileNotFoundError:
            pass

        # 目录扫描：只列权重文件、排序
        open(os.path.join(lups, "zz.txt"), "wb").close()
        assert upscale_net.scan_models() == ["tiny2d.pth", "tiny3d.pth"]


def test_detect_arch_from_weights():
    """_detect_arch 从权重键推断结构（块数/通道/时序开关），与训练配置同径。"""
    with _env():
        sd2 = _tiny2d(2).state_dict()
        cfg = upscale_net._detect_arch(sd2, "2D")
        assert cfg["in_channels"] == 24 and cfg["channels"] == 32
        assert cfg["in_blocks"] == 1 and cfg["out_blocks"] == 1
        assert cfg["temporal_every"] == 2 and cfg["temporal_kernel"] == 3
        assert cfg["attn"] is False                          # 推理强制关 attn
        cfg0 = upscale_net._detect_arch(_tiny2d(0).state_dict(), "2D")
        assert cfg0["temporal_every"] == 0

        cfg3 = upscale_net._detect_arch(_tiny3d(2).state_dict(), "3D")
        assert cfg3["channels"] == 32 and cfg3["in_blocks"] == 1
        assert cfg3["temporal_every"] == 2 and cfg3["temporal_kernel"] == 3
        cfg30 = upscale_net._detect_arch(_tiny3d(0).state_dict(), "3D")
        assert cfg30["temporal_every"] == 0


# ---- 存档联动 ----

def test_upseg_save_load_and_truncate():
    with _env() as (out, _):
        root = os.path.join(out, "h3_projects", "甲")
        os.makedirs(root)
        v = torch.randn(1, 24, 7, 4, 6)
        a = torch.randn(1, 32, 2, 1, 40)
        checkpoint.save_upsegment(root, 1, v, a)
        paths = checkpoint.upseg_paths(root, 1)
        assert paths == {"pt": "upseg_001.pt", "mp4": "upseg_001.mp4",
                         "thumb": "upthumb_001.png", "last": "uplast_001.png"}
        assert os.path.isfile(os.path.join(root, paths["pt"]))
        # load_segment 走 seg_*：二采 latent 回读用 torch.load 直接验证
        with open(os.path.join(root, paths["pt"]), "rb") as f:
            payload = torch.load(f, map_location="cpu", weights_only=True)
        assert torch.equal(payload["video"], v) and torch.equal(payload["audio"], a)

        # truncate：二采记录与文件随基础段联动清理（≥start 全清）
        mf = {"schema": "h3seamless/ckpt-v3", "done": 3, "total": 3,
              "prompt_hashes": ["a", "b", "c"], "seeds": [1, 2, 3],
              "upscale": {"segs": [
                  {"hash": "h", "base_hash": "a|1", "done": True, "files": checkpoint.upseg_paths(root, 0)},
                  {"hash": "h", "base_hash": "b|2", "done": True, "files": checkpoint.upseg_paths(root, 1)},
                  {"hash": "h", "base_hash": "c|3", "done": True, "files": checkpoint.upseg_paths(root, 2)}]}}
        checkpoint.save_manifest(root, mf)
        for g in range(3):
            for f in checkpoint.upseg_paths(root, g).values():
                open(os.path.join(root, f), "wb").close()
        out_mf = checkpoint.truncate(root, mf, 2)
        assert len(out_mf["upscale"]["segs"]) == 2
        assert os.path.isfile(os.path.join(root, "upseg_000.mp4"))
        assert os.path.isfile(os.path.join(root, "upseg_001.pt"))
        assert not os.path.exists(os.path.join(root, "upseg_002.mp4"))
        assert not os.path.exists(os.path.join(root, "uplast_002.png"))
        # manifest 已即时落盘
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
            assert len(json.load(f)["upscale"]["segs"]) == 2


def test_upscale_reset():
    with _env() as (out, _):
        root = os.path.join(out, "h3_projects", "乙")
        os.makedirs(root)
        files1 = checkpoint.upseg_paths(root, 1)
        mf = {"schema": "h3seamless/ckpt-v3", "done": 2, "total": 2,
              "prompt_hashes": ["a", "b"], "seeds": [1, 2],
              "upscale": {"segs": [
                  {"hash": "h", "base_hash": "a|1", "done": True, "files": checkpoint.upseg_paths(root, 0)},
                  {"hash": "h", "base_hash": "b|2", "done": True, "files": files1}]}}
        checkpoint.save_manifest(root, mf)
        for f in files1.values():
            open(os.path.join(root, f), "wb").close()

        from ComfyUI_H3_SeamlessChain import projects
        out_mf = projects.upscale_reset("乙", 2)
        assert out_mf["upscale"]["segs"][1] is None          # 记录清掉
        assert out_mf["upscale"]["segs"][0] is not None      # 其余段不动
        for f in files1.values():
            assert not os.path.exists(os.path.join(root, f))  # 产物文件删掉
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as fh:
            assert json.load(fh)["upscale"]["segs"][1] is None  # 真落盘

        # 记录短于段号（从未二采过）：补 None 后清（幂等无害）
        out_mf2 = projects.upscale_reset("乙", 2)
        assert out_mf2["upscale"]["segs"][1] is None

        # 错误路径
        for name, seg in (("乙", "x"), ("乙", 0), ("乙", -1), ("乙", 3), ("nope", 1),
                          ("../up", 1), ("", 1)):
            try:
                projects.upscale_reset(name, seg)
                raise AssertionError(f"应报错: {name!r} {seg!r}")
            except ValueError:
                pass


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("all tests passed")
