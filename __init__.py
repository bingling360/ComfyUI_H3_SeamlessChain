import traceback

try:
    from comfy_api.latest import ComfyExtension

    from .nodes import H3SeamlessChainSampler

    class H3SeamlessChainExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SeamlessChainSampler]

    async def comfy_entrypoint() -> H3SeamlessChainExtension:
        return H3SeamlessChainExtension()
except Exception:
    print("[ComfyUI_H3_SeamlessChain] 加载失败：本插件需要 ComfyUI v0.30.0 及以上（含官方 MiniMax H3 节点与 comfy_api.latest）。")
    print("[ComfyUI_H3_SeamlessChain] 请升级 ComfyUI 后重启。详细错误：")
    traceback.print_exc()
