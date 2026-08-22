"""实验性功能框架纯逻辑单测：python tests/test_experiments.py（无需 torch / ComfyUI）。

覆盖：开关归一化、params 合并、未知 id 忽略、fingerprint 差异、
FORCE_DISABLED 强制全关、checkpoint.assert_match 对 experiments 的双向严格比对、
truncate 对 memory_anchor 的截断清理。
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import experiments
import checkpoint


def _ctx(**overrides):
    """构造一个 standard 的 ds.experiments 结构。"""
    ex = {
        "e1_bridge_shard": True,
        "e2_memory_anchor": False,
        "params": {
            "e1_bridge_shard": {"滑窗token": 34, "子片帧数": 68, "重叠token": 2},
        },
    }
    ex.update(overrides)
    return ex


def test_defs_wellformed():
    assert "e1_bridge_shard" in experiments.EXPERIMENT_DEFS
    for exp_id, meta in experiments.EXPERIMENT_DEFS.items():
        assert meta["name"] and meta["group"] and meta["default"] is False
        assert isinstance(meta["params"], tuple) and meta["params"]
        # 全量参数元数据（端点下发前端的唯一权威）：与兼容视图一一对应、数值域合法
        pm = meta["params_meta"]
        assert [p["key"] for p in pm] == list(meta["params"])
        for p in pm:
            assert p["type"] in ("num", "enum")
            if p["type"] == "num":
                assert p["min"] <= p["def"] <= p["max"] and p["step"] > 0
            else:
                assert p["def"] in p["opts"]


def test_experiment_defs_payload():
    import json
    payload = experiments.experiment_defs_payload()
    # JSON 可序列化（端点直接下发）
    json.dumps(payload, ensure_ascii=False)
    assert payload["ok"] is True
    assert "force_disabled" in payload
    exps = payload["experiments"]
    assert len(exps) == len(experiments.EXPERIMENT_DEFS)
    assert {e["id"] for e in exps} == set(experiments.EXPERIMENT_DEFS)
    for e in exps:
        assert e["name"] and e["group"] and e["desc"]
        for p in e["params"]:
            assert p["key"] and p["type"] in ("num", "enum")


def test_nested_on_not_recognized():
    # 扁平契约唯一：早期嵌套 {on: {...}} 脏数据不再被识别（兼容层已删除）
    c = experiments.resolve({"experiments": {"on": {"e1_bridge_shard": True}, "params": {}}})
    assert not c.enabled
    assert not c.has("e1_bridge_shard")
    # locked 是前端 UI 键，后端天然忽略、不影响开关与指纹
    c2 = experiments.resolve({"experiments": {"e1_bridge_shard": True, "locked": False, "params": {}}})
    assert c2.has("e1_bridge_shard")
    assert "locked" not in c2.fingerprint()


def test_empty_context_false_for_all():
    c = experiments.resolve(None)
    assert not c.enabled
    assert c.active_list() == []
    assert c.fingerprint() == ""
    for exp_id in experiments.EXPERIMENT_DEFS:
        assert c.has(exp_id) is False
    c2 = experiments.resolve({})
    assert not c2.enabled
    # 非 dict（如 boolean/str）也归一为空
    assert not experiments.resolve("x").enabled


def test_parse_on_and_params():
    c = experiments.resolve({"experiments": _ctx()})
    assert c.enabled
    assert c.has("e1_bridge_shard")
    assert not c.has("e2_memory_anchor")
    assert c.param("e1_bridge_shard", "滑窗token") == 34
    assert c.param("e1_bridge_shard", "重叠token") == 2
    assert c.param("e2_memory_anchor", "记忆帧数", 2) == 2   # 未开启项读参数给默认
    assert c.active_list() == ["e1_bridge_shard"]


def test_unknown_id_ignored():
    c = experiments.resolve({"experiments": {"ghost_exp": True, "params": {}}})
    assert not c.enabled
    assert c.active_list() == []
    assert not c.has("ghost_exp")


def test_params_not_shared_across_experiments():
    c = experiments.resolve({"experiments": _ctx()})
    # 关闭项 / 其它项的 params 字典互不污染
    assert c.param("e3_motion_gate", "运动z阈值", 2.0) == 2.0


def test_fingerprint_differs_by_combo_and_params():
    base = {"experiments": _ctx()}
    a = experiments.resolve(base)
    # 只开关不掉 -> 指纹变（触发整链重做）
    b_on2 = {"experiments": {**_ctx(), "e2_memory_anchor": True}}
    assert experiments.resolve(b_on2).fingerprint() != a.fingerprint()
    # 同开关但改参数 -> 指纹变
    c_param = {"experiments": _ctx()}
    c_param["experiments"]["params"] = {
        "e1_bridge_shard": {"滑窗token": 999, "子片帧数": 68, "重叠token": 2}}
    assert experiments.resolve(c_param).fingerprint() != a.fingerprint()
    # 同样的组合+参数 -> 指纹稳定一致
    assert experiments.resolve(base).fingerprint() == a.fingerprint()


def test_force_disabled_overrides_everything(monkeypatch):
    c = experiments.resolve({"experiments": _ctx()})
    assert c.enabled
    monkeypatch.setattr(experiments, "FORCE_DISABLED", True, raising=False)
    c2 = experiments.resolve({"experiments": _ctx()})
    assert not c2.enabled
    assert not c2.has("e1_bridge_shard")
    assert c2.fingerprint() == ""
    monkeypatch.undo()


def _no_diff_params(key="experiments", old_val="", new_val=""):
    old = {"width": 864, "length": 124}
    new = {"width": 864, "length": 124}
    if old_val is not None:
        old[key] = old_val
    if new_val is not None:
        new[key] = new_val
    return old, new


def test_assert_match_experiments_strict():
    # 关->开：旧档无 experiments（基线/previous 老链），本次开启 -> 判不一致（整链重做）
    old, new = _no_diff_params(old_val=None, new_val="e1[滑窗token=17]")
    try:
        checkpoint.assert_match(old, new)
        assert False, "应判 inconsistent"
    except ValueError as e:
        assert "experiments" in str(e)
    # 开->关：旧档开启、本次关闭 -> 判不一致
    old, new = _no_diff_params(old_val="e1[滑窗token=17]", new_val=None)
    try:
        checkpoint.assert_match(old, new)
        assert False, "应判 inconsistent"
    except ValueError:
        pass
    # 组合变化：开 e1 -> 开 e2 -> 判不一致
    old, new = _no_diff_params(old_val="e1", new_val="e2")
    try:
        checkpoint.assert_match(old, new)
        assert False, "应判 inconsistent"
    except ValueError:
        pass
    # 参数变化但不换组合 -> 判不一致
    old, new = _no_diff_params(old_val="e1[滑窗token=17]", new_val="e1[滑窗token=34]")
    try:
        checkpoint.assert_match(old, new)
        assert False, "应判 inconsistent"
    except ValueError:
        pass


def test_assert_match_experiments_consistent_or_absent():
    # 同组合同参数 -> 不报错
    checkpoint.assert_match(*_no_diff_params(old_val="e1", new_val="e1"))
    # 两边都无 experiments 键 -> 不报错（旧档续跑兼容）
    checkpoint.assert_match(*_no_diff_params(old_val=None, new_val=None))
    # 两边都不开启（显式 ""）-> 不报错
    checkpoint.assert_match(*_no_diff_params(old_val="", new_val=""))
    # 其它参数不一致仍捕获（不影响原有行为）
    try:
        checkpoint.assert_match({"width": 864}, {"width": 960})
        assert False, "应判 inconsistent"
    except ValueError:
        pass


def test_truncate_memory_anchor_partial_keeps():
    root = tempfile.mkdtemp()
    try:
        open(checkpoint.seg_path(root, 0), "wb").close()
        open(checkpoint.seg_path(root, 1), "wb").close()
        ma_file = os.path.join(root, "memory_anchor.pt")
        open(ma_file, "wb").close()
        m = {"done": 2, "seeds": [1, 2], "memory_anchor": "abc123"}
        out = checkpoint.truncate(root, m, 1)   # 只重做第 2 段 -> 首段保留，记忆锚沿用
        assert out["done"] == 1
        assert out["memory_anchor"] == "abc123"
        assert os.path.exists(ma_file)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_truncate_memory_anchor_full_clears():
    root = tempfile.mkdtemp()
    try:
        open(checkpoint.seg_path(root, 0), "wb").close()
        ma_file = os.path.join(root, "memory_anchor.pt")
        open(ma_file, "wb").close()
        m = {"done": 1, "seeds": [9], "memory_anchor": "xyz"}
        out = checkpoint.truncate(root, m, 0)   # 整链重做 -> 旧锚失效，记录与文件一起清
        assert out["done"] == 0
        assert "memory_anchor" not in out
        assert not os.path.exists(ma_file)
    finally:
        import shutil
        shutil.rmtree(root, ignore_errors=True)


def test_e1_windows_layout():
    # 退化：窗>=总 token / 窗<=0 -> 单窗 (= 现状零影响)
    assert experiments.e1_windows(17, 6, 2) == [(0, 6), (4, 10), (8, 14), (11, 17)]
    assert experiments.e1_windows(17, 17, 2) == [(0, 17)]
    assert experiments.e1_windows(17, 0, 2) == [(0, 17)]
    assert experiments.e1_windows(17, 40, 2) == [(0, 17)]
    assert experiments.e1_windows(0, 6, 2) == []
    # 完整覆盖 + 首窗从 0 + 末窗贴 T
    spans = experiments.e1_windows(32, 8, 3)
    assert spans[0][0] == 0
    assert spans[-1][1] == 32
    for a, b in spans:
        assert b > a and 0 <= a < b <= 32
    for (a, b), (c, d) in zip(spans, spans[1:]):
        assert c < b          # 相邻窗口确实重叠/相接
    assert len(spans) >= 2


class _FakeLatent:
    def __init__(self, t):
        self.shape = (1, 16, t, 8, 8)
    def dim(self):
        return 5
    def clone(self):
        return self
    def __getitem__(self, s):
        return self   # e1_window_kf 只做切片，测试只看结构与索引


def test_e1_window_kf_expansion_and_degrade():
    frames_of = lambda t: t * 4     # 简单单调 token->frame 映射（token 0 -> 帧 0）
    guide = {"resolved_frame_index": 0, "latent": _FakeLatent(17), "audio_latent": object()}
    out = experiments.e1_window_kf(guide, 6, 2, frames_of)
    assert len(out) > 1
    # 首窗在接缝 0，末窗贴到 T；音频只附首窗
    assert out[0]["resolved_frame_index"] == 0
    assert "audio_latent" in out[0]
    assert len([k for k in out if "audio_latent" in k]) == 1
    # 各窗起点帧序递增（重叠滑窗）
    idxs = [k["resolved_frame_index"] for k in out]
    assert idxs == sorted(idxs)
    # 退化：单 token 引导 / latency<3 维 -> 原样单元素
    g1 = {"resolved_frame_index": 0, "latent": _FakeLatent(1)}
    assert experiments.e1_window_kf(g1, 6, 2, frames_of) == [dict(g1)]
    assert experiments.e1_window_kf(None, 6, 2, frames_of) == [None]


def test_memory_tokens():
    f2t = lambda f, up=True: max(1, (f + 4) // 4)   # 单调伪映射：4 帧 ≈ 1 token
    assert experiments.memory_tokens(2, f2t) == 1
    assert experiments.memory_tokens(8, f2t) == 3
    assert experiments.memory_tokens(0, f2t) == 1    # <=0 兜底 1 token
    assert experiments.memory_tokens(-4, f2t) == 1
    # 映射给出 0 也兜底 1（首段恒有至少 1 token 可做 reference）
    assert experiments.memory_tokens(3, lambda f, up=True: 0) == 1


def test_memory_anchor_positions():
    # 段首模式：只锚接缝处（0 帧）
    assert experiments.memory_anchor_positions("段首", 124) == [0]
    # 全程模式：段首 + 段中
    assert experiments.memory_anchor_positions("全程", 124) == [0, 62]
    # 全程但采样帧数太少（<=2）不拆第二锚
    assert experiments.memory_anchor_positions("全程", 2) == [0]
    assert experiments.memory_anchor_positions("全程", 1) == [0]
    # 未知模式兜底段首
    assert experiments.memory_anchor_positions("乱写", 124) == [0]
    # 非整数帧数也安全（int 收敛）
    assert experiments.memory_anchor_positions("全程", 5) == [0, 2]


def test_e3_motion_trigger():
    # flow_z 超阈值 → 触发
    trig, act = experiments.e3_motion_trigger({"flow_z": 2.5, "cam_z": 0.3}, 2.0, "重摇")
    assert trig is True and act == "重摇"
    # cam_z 超阈值 → 触发，动作为重锚
    trig, act = experiments.e3_motion_trigger({"flow_z": 0.5, "cam_z": -3.1}, 2.0, "重锚")
    assert trig is True and act == "重锚"
    # 都不超 → 不触发
    trig, _ = experiments.e3_motion_trigger({"flow_z": 1.0, "cam_z": -1.5}, 2.0, "重摇")
    assert trig is False
    # 缺 flow_z / cam_z → 不触发（不因为 None 误触发）
    trig, _ = experiments.e3_motion_trigger({"flow_z": None, "cam_z": None}, 2.0, "重摇")
    assert trig is False
    trig, _ = experiments.e3_motion_trigger({"lpips_z": 3.0}, 2.0, "重摇")  # 只有无关维度
    assert trig is False
    # 空/非 dict → 不触发
    assert experiments.e3_motion_trigger(None, 2.0)[0] is False
    assert experiments.e3_motion_trigger({}, 2.0)[0] is False
    # 未知动作兜底重摇
    _, act = experiments.e3_motion_trigger({"flow_z": 2.5}, 2.0, "不知道")
    assert act == "重摇"
    # 阈值恰好：等于不触发（严格 >）
    trig, _ = experiments.e3_motion_trigger({"flow_z": 2.0}, 2.0)
    assert trig is False


_MAIN = {
    "test_defs_wellformed": test_defs_wellformed,
    "test_experiment_defs_payload": test_experiment_defs_payload,
    "test_nested_on_not_recognized": test_nested_on_not_recognized,
    "test_empty_context_false_for_all": test_empty_context_false_for_all,
    "test_parse_on_and_params": test_parse_on_and_params,
    "test_unknown_id_ignored": test_unknown_id_ignored,
    "test_params_not_shared_across_experiments": test_params_not_shared_across_experiments,
    "test_fingerprint_differs_by_combo_and_params": test_fingerprint_differs_by_combo_and_params,
    "test_force_disabled_overrides_everything": test_force_disabled_overrides_everything,
    "test_assert_match_experiments_strict": test_assert_match_experiments_strict,
    "test_assert_match_experiments_consistent_or_absent": test_assert_match_experiments_consistent_or_absent,
    "test_truncate_memory_anchor_partial_keeps": test_truncate_memory_anchor_partial_keeps,
    "test_truncate_memory_anchor_full_clears": test_truncate_memory_anchor_full_clears,
    "test_e1_windows_layout": test_e1_windows_layout,
    "test_e1_window_kf_expansion_and_degrade": test_e1_window_kf_expansion_and_degrade,
    "test_memory_tokens": test_memory_tokens,
    "test_memory_anchor_positions": test_memory_anchor_positions,
    "test_e3_motion_trigger": test_e3_motion_trigger,
}


def _run_plain():
    """单文件直跑：逐个调用（monkeypatch 用 try/finally 手动还原）。"""
    import types
    for name, fn in _MAIN.items():
        try:
            if fn.__code__.co_argcount:
                mp = types.SimpleNamespace(
                    setattr=lambda m, k, v, **kw: setattr(m, k, v),
                    undo=lambda: None,
                )
                fn(mp)
            else:
                fn()
            print(f"  PASS {name}")
        except Exception as e:
            print(f"  FAIL {name}: {e!r}")


if __name__ == "__main__":
    print("experiments 单测直跑：")
    _run_plain()
    print("OK（真回归请在 ComfyUI 环境用 pytest tests/）")