# Sync this project to the ComfyUI deploy dir (mirror: remove extras, skip caches & docs)
# Usage: double-click sync_to_comfyui.bat, or run:  powershell -File sync_to_comfyui.ps1
# NOTE: keep this file ASCII-only; Windows PowerShell 5.1 misreads BOM-less UTF-8 Chinese as GBK.
$ErrorActionPreference = 'Stop'
$src = $PSScriptRoot
# PRIMARY: the actually-running instance. Its main.py is started with
#   --base-directory D:\Comfy-Desktop\ComfyUI-Shared
# so custom_nodes are loaded from there, NOT from the install directory.
# SECONDARY: install-dir copy kept as a mirror for runs without --base-directory.
$dsts = @(
    'D:\Comfy-Desktop\ComfyUI-Shared\custom_nodes\ComfyUI_H3_SeamlessChain',
    'D:\本地部署文件集合\comfyui\comfy shili\ComfyUI\ComfyUI\custom_nodes\ComfyUI_H3_SeamlessChain'
)

foreach ($dst in $dsts) {

if (-not (Test-Path $dst)) { New-Item -ItemType Directory -Force -Path $dst | Out-Null }

# Clear the target first (keep the folder itself and its .git) to avoid stale old files
Get-ChildItem $dst -Force | Where-Object { $_.Name -ne '.git' } | Remove-Item -Recurse -Force

$dirs = @('', 'web', 'tests', 'example_workflows')
foreach ($d in $dirs) {
    $s = Join-Path $src $d
    $t = Join-Path $dst $d
    if (-not (Test-Path $s)) { continue }
    New-Item -ItemType Directory -Force -Path $t | Out-Null
    $files = Get-ChildItem $s -File | Where-Object {
        $_.Name -notmatch '^__pycache__$' -and $_.Extension -ne '.pyc' -and
        $_.Name -ne '改造方案_逐段落盘与桥帧门控.md' -and $_.Name -notlike 'sync_to_comfyui*' -and
        $_.Name -notlike '同步到ComfyUI*'
    }
    $files | Copy-Item -Destination $t -Force
}

Write-Host ''
Write-Host ('Synced to ' + $dst)

}

Write-Host ''
Write-Host 'Restart ComfyUI (or hard-refresh the browser) to load new nodes/web UI.'
