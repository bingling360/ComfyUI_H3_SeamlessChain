"""media.save_av_mp4 端到端回归（需要 av + torch，缺失自动 SKIP）。

历史 bug：临时文件名 *.part 使 PyAV 无法按扩展名推断封装格式，
av.open 直接抛 ValueError 被吞 -> 分段 mp4 与自动成片全部静默失败
（AutoDL 实测：存档目录里一个 mp4 都没有）。显式 format="mp4" 修复，
本测试守住该真实编码路径（无 av 环境跑不了，SKIP 可接受）。
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import av
    import torch
    # 合集运行时 test_node_structure 的假 torch stub 可能已在 sys.modules——视为缺失
    _HAS = hasattr(torch, "rand")
except Exception:
    _HAS = False


@unittest.skipUnless(_HAS, "需要 av + torch（ComfyUI 运行环境自带）")
class TestSaveAvMp4(unittest.TestCase):
    def test_part_tmpname_writes_mp4(self):
        from media import save_av_mp4
        frames = torch.rand(24, 64, 64, 3)
        wav = torch.randn(1, 44100) * 0.1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.mp4")
            self.assertTrue(save_av_mp4(path, frames, wav, 44100))
            self.assertGreater(os.path.getsize(path), 1000)
            self.assertFalse(os.path.exists(path + ".part"))   # 无半成品残留
            with av.open(path) as c:                            # 可重新打开、帧数守恒
                n = sum(1 for _ in c.decode(video=0))
            self.assertEqual(n, 24)

    def test_probe_video_size(self):
        """实测 (宽, 高) 且宽在前——try_final 拼接画幅改走探测口径的回归钉。"""
        from media import probe_video_size, save_av_mp4
        frames = torch.rand(6, 48, 80, 3)   # 高 48 × 宽 80 的横帧
        wav = torch.randn(1, 44100) * 0.1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "out.mp4")
            self.assertTrue(save_av_mp4(path, frames, wav, 44100))
            self.assertEqual(probe_video_size(path), (80, 48))
            self.assertIsNone(probe_video_size(os.path.join(d, "nope.mp4")))

    def test_hq_encode_options_and_dither(self):
        """HQ 编码档（抗糊 N5）：crf/preset/aq/dither 透传后可编码、帧数守恒。"""
        from media import save_av_mp4
        frames = torch.rand(12, 48, 64, 3)
        wav = torch.randn(1, 44100) * 0.1
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "hq.mp4")
            self.assertTrue(save_av_mp4(path, frames, wav, 44100, crf=16,
                                        preset="medium", aq_mode=3, dither=True))
            self.assertGreater(os.path.getsize(path), 1000)
            with av.open(path) as c:
                n = sum(1 for _ in c.decode(video=0))
            self.assertEqual(n, 12)


class TestDitherQuantize(unittest.TestCase):
    """Bayer 有序抖动（抗条纹核心）：纯 numpy，无需 av。"""

    def setUp(self):
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("需要 numpy")
        from media import dither_quantize
        self.dq = dither_quantize

    def test_range_and_dtype(self):
        import numpy as np
        out = self.dq(np.full((9, 9, 3), 2.0, dtype="float32"))    # 远超界
        self.assertEqual(out.dtype, np.uint8)
        self.assertEqual(int(out.max()), 255)
        out0 = self.dq(np.full((9, 9, 3), -1.0, dtype="float32"))
        self.assertEqual(int(out0.min()), 0)

    def test_determinism(self):
        import numpy as np
        x = np.random.RandomState(7).rand(37, 41, 3).astype("float32")
        np.testing.assert_array_equal(self.dq(x), self.dq(x.copy()))

    def test_uint8_input_preserves_levels(self):
        """防御误用：uint8 不得按 0..1 浮点重乘 255 后整片截白。"""
        import numpy as np
        x = np.arange(48, dtype="uint8").reshape(4, 4, 3) * 5
        np.testing.assert_array_equal(self.dq(x), x)

    def test_banding_breakup(self):
        """常量场量化误差被 16 档抖动打散（非整块同值）——banding 的反面。"""
        import numpy as np
        # 量化台阶中点：无抖动时整块 floor(127.5)=127 单值 → 有抖动出现 127/128 交替
        x = np.full((16, 16, 1), 127.5 / 255.0, dtype="float32")
        out = self.dq(x)
        vals = sorted(set(int(v) for v in out.flatten()))
        self.assertIn(127, vals)
        self.assertIn(128, vals)

    def test_unbiased_mean(self):
        """期望无偏：抖动后均值与原值×255 差 < 半个量化步。"""
        import numpy as np
        x = np.full((64, 64, 1), 0.501, dtype="float32")   # 127.755 中点附近
        out = self.dq(x).astype("float32")
        self.assertLess(abs(float(out.mean()) - 127.755), 0.5)

    def test_4x4_pattern_period(self):
        """抖动模式以 4×4 为周期平铺（非 4 倍数尺寸也保持平铺一致性）。"""
        import numpy as np
        x = np.full((7, 9, 1), 127.5 / 255.0, dtype="float32")
        out = self.dq(x)
        # (2,1) 处 bayer=11（阈值>0.5 → 128），(0,0) 处 bayer=0（→127）：
        # 同相不同值、跨周期同相同值——平铺正确性的正反两面
        self.assertEqual(int(out[2, 1, 0]), int(out[6, 5, 0]))
        self.assertNotEqual(int(out[0, 0, 0]), int(out[2, 1, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
