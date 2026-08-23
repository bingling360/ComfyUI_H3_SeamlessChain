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


if __name__ == "__main__":
    unittest.main(verbosity=2)
