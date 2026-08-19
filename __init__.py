import traceback

WEB_DIRECTORY = "./web"  # 成片画廊 JS（h3chain_saver.js）随插件分发

try:
    from comfy_api.latest import ComfyExtension

    from .nodes import H3SeamlessChainSampler
    from .storyboard import H3StoryboardChain
    from .saver import H3ChainSaver
    from .seam_doctor import H3SeamDoctor

    class H3SeamlessChainExtension(ComfyExtension):
        async def get_node_list(self):
            return [H3SeamlessChainSampler, H3StoryboardChain, H3ChainSaver,
                    H3SeamDoctor]

        async def add_routes(self, routes):
            # 兼容未来版本可能存在的扩展钩子；现行版本靠下方 register() 直挂
            try:
                from .routes import register
                register()
            except Exception:
                print("[ComfyUI_H3_SeamlessChain] 路由注册失败（成片画廊降级：仅显示当前链）。详细错误：")
                traceback.print_exc()

    async def comfy_entrypoint() -> H3SeamlessChainExtension:
        return H3SeamlessChainExtension()

    # 路由注册主路径：PromptServer.instance.routes（custom nodes 加载时 instance 已就绪）。
    # ComfyExtension 基类在现行 ComfyUI 没有 add_routes 钩子，靠扩展钩子注册的旧写法
    # 导致路由从未挂上（前端删除请求 405）。
    try:
        from .routes import register as _register_routes
        _register_routes()
    except Exception:
        print("[ComfyUI_H3_SeamlessChain] 路由注册入口调用失败。详细错误：")
        traceback.print_exc()
except Exception:
    print("[ComfyUI_H3_SeamlessChain] 加载失败：本插件需要 ComfyUI v0.30.0 及以上（含官方 MiniMax H3 节点与 comfy_api.latest）。")
    print("[ComfyUI_H3_SeamlessChain] 请升级 ComfyUI 后重启。详细错误：")
    traceback.print_exc()
