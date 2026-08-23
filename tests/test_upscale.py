"""潜空间放大二采单测（真 torch + einops）：python tests/test_upscale.py

覆盖：parse_state 参数归一化、二采参数指纹与重做判定（record_stale /
_record_valid / base_hash）、latent 放大数学（target_hw 偶数对齐 /
resize_latent_bilinear）、2D/3D 放大网络前向形状与权重加载（架构不匹配
明确报错 / MODEL_CACHE 缓存 / _detect_arch 结构推断）、高清分段同名存档
联动（upscale_files 命名 / truncate 清扫 / try_final 记录门控 /
projects.upscale_reset 自愈重建）。
render_latent / render_segment 本体依赖 ComfyUI 运行环境，不在本测范围
（异常兜底在 nodes.py 调用点：任何失败降级为报告，不丢基础链产物）。
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
    assert cfg["scale"] == 2.0
    assert cfg["denoise"] == 0.35              # 未存过字段=新默认（无 schema 键按 v1 迁移也无损）
    assert cfg["steps"] == 6                   # 未存过 steps=新默认精化步数
    assert cfg["cfg"] == 1.0
    assert cfg["precision"] == "fp16"
    assert cfg["time_bias"] == 0.0
    assert cfg["mix"] == 0.0

    full = upscale.parse_state({"upscale": {
        "schema": 2, "mode": "手动选择", "model": " up2d.pth ", "arch": "3d", "scale": "9",
        "denoise": "0.01", "steps": "0", "cfg": "-5", "precision": "x",
        "time_bias": "9", "mix": "2",
        "include": [2, "1", 1.0, "bad", None]}})
    assert full["mode"] == "手动选择"
    assert full["model"] == "up2d.pth"          # strip 空白
    assert full["arch"] == "3D"                 # 大小写归一
    assert full["scale"] == 4.0                 # 钳上限
    assert full["denoise"] == 0.05              # 钳下限
    assert full["steps"] == 1
    assert full["cfg"] == 0.0
    assert full["precision"] == "fp16"          # 非法精度回落
    assert full["time_bias"] == 0.2             # 钳上限
    assert full["mix"] == 1.0                   # 钳上限
    assert full["include"] == [1, 2]            # 去重排序，垃圾项剔除

    nan = upscale.parse_state({"upscale": {"mode": "跟随生成", "scale": float("nan"),
                                           "denoise": float("nan")}})
    assert nan["scale"] == 2.0 and nan["denoise"] == 0.35   # NaN 回落默认


def test_parse_state_schema_migration():
    """v1「steps=调度总步数」-> v2「steps=精化步数」：显式存过的按 int(N×强度) 折算保行为。"""
    v1 = upscale.parse_state({"upscale": {"mode": "跟随生成", "denoise": 0.45, "steps": 15}})
    assert v1["denoise"] == 0.45 and v1["steps"] == 6      # int(15×0.45)=6，恰为新默认步数
    v1b = upscale.parse_state({"upscale": {"mode": "跟随生成", "denoise": 0.3, "steps": 20}})
    assert v1b["denoise"] == 0.3 and v1b["steps"] == 6     # int(20×0.3)
    v1c = upscale.parse_state({"upscale": {"mode": "跟随生成", "denoise": 0.05, "steps": 10}})
    assert v1c["steps"] == 1                               # int(0.5)=0 钳到 1
    # v1 完全没存过二采参数（没碰过面板）= 直接落 v2 新默认
    v1d = upscale.parse_state({"upscale": {"mode": "跟随生成"}})
    assert v1d["denoise"] == 0.35 and v1d["steps"] == 6
    # schema 2：steps 即精化步数（不乘强度），新默认 0.35/6
    v2 = upscale.parse_state({"upscale": {"mode": "跟随生成", "schema": 2}})
    assert v2["denoise"] == 0.35 and v2["steps"] == 6
    v2b = upscale.parse_state({"upscale": {"mode": "跟随生成", "schema": 2,
                                           "denoise": 0.4, "steps": 3}})
    assert v2b["denoise"] == 0.4 and v2b["steps"] == 3


def test_tail_refine_args():
    """(尾段起始σ, 精化步数 n) -> common_ksampler(steps=N, denoise)：int(N·denoise) 恒为 n。"""
    N, d = upscale.tail_refine_args(0.35, 6)               # 新默认档
    assert N == 17 and int(N * d) == 6 and abs(d - 6.5 / 17) < 1e-9
    N, d = upscale.tail_refine_args(0.45, 6)               # v1 旧默认迁移档
    assert N == 13 and int(N * d) == 6
    # σ≥0.95 = 全量重采（明确请求：丢弃放大 latent 从纯噪声起步）
    assert upscale.tail_refine_args(1.0, 6) == (6, 1.0)
    assert upscale.tail_refine_args(0.95, 4) == (4, 1.0)
    N, d = upscale.tail_refine_args(0.94, 6)
    assert N == 7 and d == 6.5 / 7            # 0.94 仍是部分精化（不撞 denoise=1）
    # 高σ小步数角落：整数网格无法同时满足「σ₀≈s 且 n 步」，退到 n/(n+1) 且绝不撞全量重采
    N, d = upscale.tail_refine_args(0.9, 2)
    assert N == 3 and 0.05 < d < 1.0 and int(N * d) == 2   # 2 步 @ σ≈0.667
    N, d = upscale.tail_refine_args(0.9, 4)
    assert N == 5 and 0.05 < d < 1.0 and int(N * d) == 4   # 4 步 @ σ≈0.8
    # 输入钳制：步数 ≥1、σ ≥0.05
    assert upscale.tail_refine_args(0.5, 0) == (2, 0.75)
    N, d = upscale.tail_refine_args(0.01, 6)               # σ 钳到 0.05
    assert N == 100 and int(N * d) == 5
    # 常用区（σ≤0.6、步数≥3）不变量：步数精确、永不撞全量重采、σ₀ 贴请求值
    for s in (0.05, 0.12, 0.25, 0.35, 0.45, 0.6):
        for n in (3, 5, 6, 8, 12, 25, 50, 100):
            N, d = upscale.tail_refine_args(s, n)
            n_eff = int(N * d)
            assert 1 <= N <= 100 and 0.05 < d < 1.0
            assert n_eff == n or (N == 100 and n_eff < n)  # 只允许 100 格上限压步数
            assert abs(n_eff / N - s) < 0.07                # 网格量化误差上限


def test_params_hash():
    a = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
    b = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                         "include": [0, 3]}})
    h = upscale.params_hash(a)
    assert len(h) == 8
    assert h == upscale.params_hash(b)          # mode/include 不进指纹（不影响单段输出）
    assert h == upscale.params_hash(upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}}))
    for key, val in (("denoise", 0.6), ("scale", 1.5), ("steps", 20), ("cfg", 2.0),
                     ("arch", "3D"), ("precision", "fp32"), ("model", "other.pth")):
        c = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth", key: val}})
        assert h != upscale.params_hash(c), key


def test_latent_hf_energy():
    """高频能量度量：零张量=0、噪声>平滑、维度兜底、抽样确定性。"""
    z = torch.zeros(1, 24, 6, 16, 16)
    assert upscale.latent_hf_energy(z) == 0.0
    smooth = torch.zeros(1, 24, 30, 32, 32)
    smooth[:, :, :, :, 8:] = 1.0                      # 单阶跃（低频为主）
    noisy = smooth + torch.randn(1, 24, 30, 32, 32) * 0.05
    e_smooth = upscale.latent_hf_energy(smooth)
    e_noisy = upscale.latent_hf_energy(noisy)
    assert 0.0 < e_smooth < e_noisy
    assert upscale.latent_hf_energy(noisy) == e_noisy            # 确定性（同输入同值）
    assert upscale.latent_hf_energy(None) == 0.0                 # 兜底
    assert upscale.latent_hf_energy(torch.zeros(24, 30, 32, 32)) == 0.0   # 非 5D=0
    assert upscale.latent_hf_energy(torch.zeros(1, 24, 6, 2, 2)) == 0.0   # 空间<3=0


def test_hf_gain_ratio():
    assert upscale.hf_gain_ratio(0.0, 5.0) == 0.0      # 基线≈0 视为无增益
    assert upscale.hf_gain_ratio(1e-12, 5.0) == 0.0
    assert abs(upscale.hf_gain_ratio(1.0, 1.25) - 0.25) < 1e-9
    assert abs(upscale.hf_gain_ratio(1.0, 0.8) + 0.2) < 1e-9


def test_freq_mix_latents():
    """频域细节混合：0=原样精化输出（关）、1=低频base+高频refined、线性、退化兜底。"""
    f = upscale.freq_mix_latents

    def _lp(x):
        xp = torch.nn.functional.pad(x, (1, 1, 1, 1), mode="replicate")
        return torch.nn.functional.avg_pool2d(xp, 3, stride=1)

    torch.manual_seed(7)
    base = torch.randn(1, 8, 9, 12, 10)
    refined = base + torch.randn_like(base) * 0.2
    keep_r, keep_b = refined.clone(), base.clone()   # 入参快照（任何调用前取——防共享存储隐式改写）

    # 0=关：原样返回精化输出本身（不克隆、零开销——默认行为与指纹均不变）
    assert f(base, refined, 0.0) is refined
    assert f(base, refined, -1.0) is refined
    assert f(None, refined, 1.0) is refined                # base 缺失兜底
    # 1=全混合：低频走 base、高频走 refined（拉普拉斯分层恒等式）
    full = f(base, refined, 1.0)
    lp_b, lp_r = _lp(base[0]), _lp(refined[0])
    assert torch.allclose(full[0], lp_b + (refined[0] - lp_r), atol=1e-5)
    # 中间值线性：out = refined + r·(lp(base) − lp(refined))
    half = f(base, refined, 0.5)
    assert torch.allclose(half[0], refined[0] + 0.5 * (lp_b - lp_r), atol=1e-5)
    assert torch.allclose(f(base, refined, 1.5)[0], full[0], atol=1e-5)   # 钳上限
    # 反漂移语义：精化只做低频漂移（零高频）时，全混合后 = 纯放大 latent
    cb = torch.full((1, 4, 5, 8, 8), 2.0)
    cr = torch.full((1, 4, 5, 8, 8), -1.0)
    assert torch.allclose(f(cb, cr, 1.0), cb, atol=1e-6)
    # 纯函数性：不改入参（detach 共享存储陷阱——曾因跳过克隆把 refined 原地改写）、确定性、统一 float32
    assert torch.equal(f(base, refined, 0.5), f(base, refined, 0.5))
    assert torch.equal(refined, keep_r) and torch.equal(base, keep_b)
    assert f(base.half(), refined.half(), 1.0).dtype == torch.float32
    # 分块边界：chunk=1 与默认 chunk 逐位一致（池化逐帧独立，切块无边界效应）
    assert torch.equal(f(base, refined, 0.7, chunk=1), f(base, refined, 0.7, chunk=8))
    # 退化形状兜底：形状不一致 / 空间<3 / 非 5D → 原样 refined（保交付产物）
    mis = refined[:, :, :5]
    assert f(base, mis, 1.0) is mis
    small_b, small_r = torch.zeros(1, 4, 3, 2, 2), torch.ones(1, 4, 3, 2, 2)
    assert f(small_b, small_r, 1.0) is small_r
    flat_b, flat_r = base.reshape(-1), refined.reshape(-1)
    assert f(flat_b, flat_r, 1.0) is flat_r


def test_time_bias_sigma():
    """尾窗 smoothstep 偏置：窗外不动、窗内渐入、clamp≥0、关闭态原样。"""
    f = upscale.time_bias_sigma
    assert f(0.35, 0.35, 0.03) == 0.35          # 精化起点（p=0）不偏置
    assert f(0.20, 0.35, 0.03) == 0.20          # p≈0.43 < start 0.70 不偏置
    seen = f(0.035, 0.35, 0.03)                 # p=0.9 → 窗内
    assert 0.035 - 0.03 < seen < 0.035          # 部分偏置（smoothstep 权重 <1）
    full = f(0.0, 0.35, 0.03)                   # p=1.0 → 权重=1 → 满偏置到 clamp
    assert full == 0.0
    assert f(0.01, 0.35, 0.05) == 0.0           # 偏置越过 0 → clamp 0
    assert f(0.05, 0.35, 0.0) == 0.05           # bias=0 关
    assert f(0.05, 0.0, 0.03) == 0.05           # σ₀=0 关
    # 窗口参数：扩窗到全程 → 任意 σ 都有偏置（权重>0）
    assert f(0.30, 0.35, 0.03, start_progress=0.0, end_progress=1.0) < 0.30


def test_time_bias_guard():
    """dit.forward patch：窗外 timestep 原样、窗内替换为偏置值、恢复即还原。"""
    calls = []

    class _Dit:
        def forward(self, x, timestep, context, transformer_options={}, **kwargs):
            calls.append(float(timestep.flatten()[0]))
            return x

    dit = _Dit()
    restore = upscale.time_bias_guard(dit, sigma_start=0.35, bias=0.03)
    dit.forward(None, torch.tensor([350.0]), None)          # σ=0.35 → p=0 窗外
    assert calls[-1] == 350.0
    dit.forward(None, torch.tensor([35.0]), None)           # σ=0.035 → p=0.9 窗内
    assert 0.0 < calls[-1] < 35.0                           # 偏置后 <原值且未触底
    dit.forward(None, torch.tensor([17.5]), None)           # σ=0.0175 → 偏置越过 0 → clamp
    assert calls[-1] == 0.0
    restore()
    dit.forward(None, torch.tensor([35.0]), None)           # 恢复后原样
    assert calls[-1] == 35.0


def test_params_hash_time_bias():
    """time_bias 条件化指纹：0=不进指纹（既有记录不失效），>0 改变指纹。"""
    base = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
    off = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                           "time_bias": 0}})
    on = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                          "time_bias": 0.03}})
    h = upscale.params_hash(base)
    assert h == upscale.params_hash(off)         # 默认关闭不改变哈希
    assert h != upscale.params_hash(on)          # 启用才进指纹
    # 解析归一：默认 0 / 钳上限 0.2 / 垃圾回落 0
    assert base["time_bias"] == 0.0
    assert upscale.parse_state({"upscale": {"mode": "跟随生成",
                                            "time_bias": 0.5}})["time_bias"] == 0.2
    assert upscale.parse_state({"upscale": {"mode": "跟随生成",
                                            "time_bias": "bad"}})["time_bias"] == 0.0


def test_params_hash_mix():
    """mix 条件化指纹：0=不进指纹（既有记录不失效），>0 改变指纹；与 time_bias 独立叠加。"""
    base = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
    off = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                           "mix": 0}})
    on = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                          "mix": 0.5}})
    h = upscale.params_hash(base)
    assert h == upscale.params_hash(off)         # 默认关闭不改变哈希
    assert h != upscale.params_hash(on)          # 启用才进指纹
    # 记录口径同指纹（write_record.params 由 _hash_params 生成）
    assert "mix" not in upscale._hash_params(off) and upscale._hash_params(on)["mix"] == 0.5
    # 解析归一：默认 0 / 钳上限 1.0 / 负值回落下限 0 / 垃圾回落 0
    assert base["mix"] == 0.0
    assert upscale.parse_state({"upscale": {"mode": "跟随生成",
                                            "mix": 2.0}})["mix"] == 1.0
    assert upscale.parse_state({"upscale": {"mode": "跟随生成",
                                            "mix": -0.5}})["mix"] == 0.0
    assert upscale.parse_state({"upscale": {"mode": "跟随生成",
                                            "mix": "bad"}})["mix"] == 0.0
    # 双增强叠加：各自 >0 都进指纹，联合再变
    both = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                            "time_bias": 0.03, "mix": 0.5}})
    assert upscale.params_hash(both) != upscale.params_hash(on)
    tb_only = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth",
                                               "time_bias": 0.03}})
    assert upscale.params_hash(both) != upscale.params_hash(tb_only)
    assert upscale.params_hash(both) != h


def test_base_hash():
    mf = {"prompt_hashes": ["h0", "h1"], "seeds": [10, 20]}
    assert upscale.base_hash(mf, 0) == "h0|10"
    assert upscale.base_hash(mf, 1) == "h1|20"
    assert upscale.base_hash(mf, 9) == "|"       # 越界=双空串


# ---- 重做判定 ----

def test_record_stale():
    with tempfile.TemporaryDirectory() as root:
        cfg = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
        ph = upscale.params_hash(cfg)
        mf = {"done": 3, "total": 4, "prompt_hashes": ["a", "b", "c"],
              "seeds": [1, 2, 3], "upscale": {"segs": [None, None, None]}}
        # 无记录全待做；manifest=None（无存档）同样全待做
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg, g)] == [0, 1, 2]
        assert upscale.record_stale(None, root, cfg, 0) is True

        # 段0 有效记录：hash/base_hash 匹配且 mp4/thumb/last 文件齐（同名合并存储）
        files0 = checkpoint.upscale_files(0)
        for k in ("mp4", "thumb", "last"):
            open(os.path.join(root, files0[k]), "wb").close()
        mf["upscale"]["segs"][0] = {"hash": ph, "base_hash": "a|1", "done": True,
                                    "files": files0}
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg, g)] == [1, 2]

        # 段1 记录 hash 不匹配（二采参数变过）；段2 base_hash 不匹配（基础链改词/换种子）
        mf["upscale"]["segs"][1] = {"hash": "deadbeef", "base_hash": "b|2", "done": True,
                                    "files": checkpoint.upscale_files(1)}
        mf["upscale"]["segs"][2] = {"hash": ph, "base_hash": "X|9", "done": True,
                                    "files": checkpoint.upscale_files(2)}
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg, g)] == [1, 2]

        # 段0 mp4 被删 -> 记录失效（seg_NNN.mp4 即高清产物，文件缺失=重做）
        os.remove(os.path.join(root, files0["mp4"]))
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg, g)] == [0, 1, 2]

        # 手动模式：范围外段不待做（in_scope 短路，与记录无关）
        cfg2 = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                                "include": [1]}})
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg2, g)] == [1]
        cfg3 = upscale.parse_state({"upscale": {"mode": "手动选择", "model": "m.pth",
                                                "include": []}})
        assert [g for g in range(3) if upscale.record_stale(mf, root, cfg3, g)] == []

        # bh 显式传入（主循环按本地 full_hashes/seeds 现算，不依赖磁盘 manifest 写入时机）
        mf["upscale"]["segs"][1] = {"hash": ph, "base_hash": "b|2", "done": True,
                                    "files": checkpoint.upscale_files(1)}
        for k in ("mp4", "thumb", "last"):
            open(os.path.join(root, checkpoint.upscale_files(1)[k]), "wb").close()
        assert upscale.record_stale(mf, root, cfg, 1, "b|2") is False   # bh 匹配
        assert upscale.record_stale(mf, root, cfg, 1, "X|9") is True    # bh 不匹配


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

def test_upscale_files_and_truncate():
    with _env() as (out, _):
        root = os.path.join(out, "h3_projects", "甲")
        os.makedirs(root)
        # 高清分段与基础段同名合并存储：段视频就是二采结果（单份产物）
        assert checkpoint.upscale_files(1) == {"mp4": "seg_001.mp4",
                                               "thumb": "thumb_001.png",
                                               "last": "uplast_001.png"}
        assert checkpoint.upscale_legacy_files(1) == ("upseg_001.pt", "upseg_001.mp4",
                                                      "upthumb_001.png")
        # truncate：二采记录与文件随基础段联动清理（≥start 全清，含旧版独立产物）
        mf = {"schema": "h3seamless/ckpt-v3", "done": 3, "total": 3,
              "prompt_hashes": ["a", "b", "c"], "seeds": [1, 2, 3],
              "upscale": {"segs": [
                  {"hash": "h", "base_hash": "a|1", "done": True, "files": checkpoint.upscale_files(0)},
                  {"hash": "h", "base_hash": "b|2", "done": True, "files": checkpoint.upscale_files(1)},
                  {"hash": "h", "base_hash": "c|3", "done": True, "files": checkpoint.upscale_files(2)}]}}
        checkpoint.save_manifest(root, mf)
        for g in range(3):
            for f in checkpoint.upscale_files(g).values():
                open(os.path.join(root, f), "wb").close()
        open(os.path.join(root, "upseg_002.pt"), "wb").close()      # 旧版独立产物
        open(os.path.join(root, "upthumb_002.png"), "wb").close()
        out_mf = checkpoint.truncate(root, mf, 2)
        assert len(out_mf["upscale"]["segs"]) == 2
        assert os.path.isfile(os.path.join(root, "seg_000.mp4"))        # <start 保留
        assert os.path.isfile(os.path.join(root, "uplast_001.png"))
        assert not os.path.exists(os.path.join(root, "seg_002.mp4"))    # ≥start 高清产物随基础段清
        assert not os.path.exists(os.path.join(root, "thumb_002.png"))
        assert not os.path.exists(os.path.join(root, "uplast_002.png"))
        assert not os.path.exists(os.path.join(root, "upseg_002.pt"))   # 旧版独立产物一并清
        assert not os.path.exists(os.path.join(root, "upthumb_002.png"))
        # manifest 已即时落盘
        with open(os.path.join(root, "manifest.json"), encoding="utf-8") as f:
            assert len(json.load(f)["upscale"]["segs"]) == 2


def test_try_final_gating():
    """高清成片门控：链未完成静默回退；记录不全/尺寸混排给可见提示（不触 media）。"""
    with _env() as (out, _):
        root = os.path.join(out, "h3_projects", "丙")
        os.makedirs(root)
        cfg = upscale.parse_state({"upscale": {"mode": "跟随生成", "model": "m.pth"}})
        ph = upscale.params_hash(cfg)
        report = []
        # 无 manifest / 链未完成：静默 False（审片进行中是正常态，不刷屏）
        assert upscale.try_final(root, cfg, report) is False
        assert report == []
        mf = {"schema": "h3seamless/ckpt-v3", "done": 1, "total": 2,
              "prompt_hashes": ["a", "b"], "seeds": [1, 2]}
        checkpoint.save_manifest(root, mf)
        assert upscale.try_final(root, cfg, report) is False
        assert report == []

        # 链完成但记录不全（手动模式未全选）-> 提示 + 回退
        mf["done"] = 2
        checkpoint.save_manifest(root, mf)
        report = []
        assert upscale.try_final(root, cfg, report) is False
        assert "记录不全" in report[-1]

        # 记录在但产物文件缺失 -> 提示 + 回退
        mf["upscale"] = {"segs": [
            {"hash": ph, "base_hash": "a|1", "done": True, "files": checkpoint.upscale_files(0)},
            {"hash": ph, "base_hash": "b|2", "done": True, "files": checkpoint.upscale_files(1)}]}
        checkpoint.save_manifest(root, mf)
        report = []
        assert upscale.try_final(root, cfg, report) is False
        assert "无有效高清记录" in report[-1]

        # 记录齐但尺寸不一致（手动模式混排）-> 提示 + 回退
        for g in range(2):
            for f in checkpoint.upscale_files(g).values():
                open(os.path.join(root, f), "wb").close()
        mf["upscale"]["segs"][0]["size"] = [1376, 768]
        mf["upscale"]["segs"][1]["size"] = [2752, 1536]
        checkpoint.save_manifest(root, mf)
        report = []
        assert upscale.try_final(root, cfg, report) is False
        assert "尺寸不一致" in report[-1]


def test_upscale_reset():
    with _env() as (out, _):
        root = os.path.join(out, "h3_projects", "乙")
        os.makedirs(root)
        files1 = checkpoint.upscale_files(1)
        mf = {"schema": "h3seamless/ckpt-v3", "done": 2, "total": 2,
              "prompt_hashes": ["a", "b"], "seeds": [1, 2],
              "upscale": {"segs": [
                  {"hash": "h", "base_hash": "a|1", "done": True, "files": checkpoint.upscale_files(0)},
                  {"hash": "h", "base_hash": "b|2", "done": True, "files": files1}]}}
        checkpoint.save_manifest(root, mf)
        for f in files1.values():
            open(os.path.join(root, f), "wb").close()
        open(os.path.join(root, "upseg_001.pt"), "wb").close()   # 旧版独立产物一并清

        from ComfyUI_H3_SeamlessChain import projects
        out_mf = projects.upscale_reset("乙", 2)
        assert out_mf["upscale"]["segs"][1] is None          # 记录清掉
        assert out_mf["upscale"]["segs"][0] is not None      # 其余段不动
        for f in (*files1.values(), "upseg_001.pt"):
            assert not os.path.exists(os.path.join(root, f))  # 高清产物删掉（同名合并+旧版）
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
