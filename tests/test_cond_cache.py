"""cond 文本编码缓存代理单测：python tests/test_cond_cache.py 或 pytest tests/test_cond_cache.py

覆盖：同提示词命中（encode 只前向一次、返回等值副本且不污染缓存）、不同
提示词各前向、属性透传原 clip、LRU 容量逐出后重前向、外部 tokens 直通
旁路（不缓存不错乱）、encode 额外参数参与缓存键、不可哈希参数安全旁路。
本测不依赖 torch/ComfyUI（代理为纯 Python）。
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ComfyUI_H3_SeamlessChain.cond_cache import CachedClipProxy


class _FakeClip:
    version = "v7"

    def __init__(self):
        self.enc_calls = []

    def tokenize(self, text, *args, **kwargs):
        return {"tok": text, "args": args, "kwargs": kwargs}

    def encode_from_tokens_scheduled(self, tokens, *args, **kwargs):
        self.enc_calls.append(tokens)
        return [(tokens["tok"], {"n": len(self.enc_calls)})]   # ComfyUI cond 惯例：[(张量, extra dict), ...]


def _encode(proxy, text):
    return proxy.encode_from_tokens_scheduled(proxy.tokenize(text))


def test_same_prompt_hits_cache():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    c1 = _encode(proxy, "一只猫")
    c2 = _encode(proxy, "一只猫")
    assert len(clip.enc_calls) == 1          # TE 只前向一次
    assert c2 == c1                          # 命中返回等值 cond
    assert c2 is not c1                      # 副本防原地写污染缓存
    assert proxy.hits == 1 and proxy.misses == 1
    assert proxy.last_encode_hit is True


def test_hit_copy_does_not_pollute_cache():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    c1 = _encode(proxy, "一只猫")
    c1[0][1]["width"] = 1920                 # 下游原地写（非常规）不进缓存
    c2 = _encode(proxy, "一只猫")
    assert len(clip.enc_calls) == 1
    assert "width" not in c2[0][1]


def test_different_prompts_each_encode():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    _encode(proxy, "一只猫")
    _encode(proxy, "一只狗")
    assert len(clip.enc_calls) == 2
    assert proxy.hits == 0 and proxy.misses == 2


def test_attribute_passthrough():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    assert proxy.version == "v7"
    assert proxy.enc_calls is clip.enc_calls
    assert proxy.tokenize is not clip.tokenize   # 仅编码链路被拦截


def test_lru_capacity_evicts_oldest():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip, capacity=2)
    _encode(proxy, "a")
    _encode(proxy, "b")
    _encode(proxy, "a")                       # 仍在容量内（LRU 触碰）
    assert proxy.hits == 1 and len(clip.enc_calls) == 2
    _encode(proxy, "c")                       # 容量 2：逐出最久未用的 "b"
    _encode(proxy, "b")                       # 被逐出 → 重新前向
    assert len(clip.enc_calls) == 4
    assert proxy.hits == 1


def test_foreign_tokens_bypass():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    _encode(proxy, "一只猫")
    out = proxy.encode_from_tokens_scheduled({"tok": "外部tokens"})   # 非代理 tokenize 产物
    assert len(clip.enc_calls) == 2          # 直通不查缓存
    assert proxy.hits == 0 and proxy.misses == 2
    assert out == [("外部tokens", {"n": 2})]


def test_encode_args_join_cache_key():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    t1 = proxy.tokenize("一只猫")
    proxy.encode_from_tokens_scheduled(t1, width=864)
    t2 = proxy.tokenize("一只猫")
    proxy.encode_from_tokens_scheduled(t2, width=1920)   # 参数不同 → 不命中
    t3 = proxy.tokenize("一只猫")
    proxy.encode_from_tokens_scheduled(t3, width=864)    # 参数相同 → 命中
    assert len(clip.enc_calls) == 2
    assert proxy.hits == 1


def test_unhashable_args_bypass_safely():
    clip = _FakeClip()
    proxy = CachedClipProxy(clip)
    t = proxy.tokenize("一只猫", {"不可哈希": object()})
    out = proxy.encode_from_tokens_scheduled(t)
    out2 = proxy.encode_from_tokens_scheduled(proxy.tokenize("一只猫", {"不可哈希": object()}))
    assert out is not None and out2 is not None
    assert proxy.hits == 0                   # 全程旁路，行为与无代理一致


def test_plain_encode_variant_cached_too():
    clip = _FakeClip()
    clip.encode_from_tokens = clip.encode_from_tokens_scheduled
    proxy = CachedClipProxy(clip)
    proxy.encode_from_tokens(proxy.tokenize("一只猫"))
    proxy.encode_from_tokens(proxy.tokenize("一只猫"))
    assert len(clip.enc_calls) == 1


if __name__ == "__main__":
    sys.exit(pytest.main([os.path.abspath(__file__), "-v"]))
