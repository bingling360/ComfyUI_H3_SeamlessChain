import traceback

WEB_DIRECTORY = "./web"  # 控制台/成片画廊 JS（h3chain_console.js / h3chain_saver.js）随插件分发

try:
    from comfy_api.latest import ComfyExtension

    from .nodes import H3SeamlessChainSampler
    from .storyboard import H3StoryboardChain
    from .console import H3ChainConsole
    from .saver import H3ChainSaver

    class H3SeamlessChainExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SeamlessChainSampler, H3StoryboardChain, H3ChainConsole, H3ChainSaver]

        async def add_routes(self, routes):
            try:
                from .routes import add_routes
                add_routes(routes)
            except Exception:
                print("[ComfyUI_H3_SeamlessChain] 路由注册失败（控制台降级：手输存档名 + 单链浏览）。详细错误：")
                traceback.print_exc()

    async def comfy_entrypoint() -> H3SeamlessChainExtension:
        return H3SeamlessChainExtension()
except Exception:
    print("[ComfyUI_H3_SeamlessChain] 加载失败：本插件需要 ComfyUI v0.30.0 及以上（含官方 MiniMax H3 节点与 comfy_api.latest）。")
    print("[ComfyUI_H3_SeamlessChain] 请升级 ComfyUI 后重启。详细错误：")
    traceback.print_exc()
