# 把本项目完整同步到 ComfyUI 部署目录（镜像：删除目标多余文件，排除缓存与设计文档）
# 用法：双击「同步到ComfyUI.bat」或在本目录运行  powershell -File sync_to_comfyui.ps1
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
$dst = 'D:\Comfy-Desktop\ComfyUI-Shared\custom_nodes\ComfyUI_H3_SeamlessChain'

if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Force -Path $dst | Out-Null }

# 先清空目标（保留目录本身），避免旧版本残留
Get-ChildItem $dst -Recurse -Force | Remove-Item -Recurse -Force

$dirs = @('', 'web', 'tests', 'example_workflows')
foreach ($d in $dirs) {
    $s = Join-Path $src $d
    $t = Join-Path $dst $d
    if (-not (Test-Path $s)) { continue }
    New-Item -ItemType Directory -Force -Path $t | Out-Null
    $files = Get-ChildItem $s -File | Where-Object {
        $_.Name -notmatch '^__pycache__$' -and $_.Extension -notin ('.pyc',) -and
        $_.Name -ne '改造方案_逐段落盘与桥帧门控.md' -and $_.Name -notlike 'sync_to_comfyui*' -and
        $_.Name -notlike '同步到ComfyUI*'
    }
    $files | Copy-Item -Destination $t -Force
}

Write-Host ''
Write-Host ('已同步到 ' + $dst)
Write-Host '请重启 ComfyUI 使新控件（接缝混合 / 混合帧数）生效。'
