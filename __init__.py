import traceback

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

try:
    from .nodes import H3SeamlessChainSampler
    NODE_CLASS_MAPPINGS["H3SeamlessChainSampler"] = H3SeamlessChainSampler
    NODE_DISPLAY_NAME_MAPPINGS["H3SeamlessChainSampler"] = "H3 Seamless Chain (段间引导续拍)"
except Exception:
    print("[ComfyUI_H3_SeamlessChain] 加载失败：本插件需要 ComfyUI v0.30.0 及以上（含官方 MiniMax H3 节点）。")
    print("[ComfyUI_H3_SeamlessChain] 请升级 ComfyUI 后重启。详细错误：")
    traceback.print_exc()
