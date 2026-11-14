<#
.SYNOPSIS
  Start the sh-rag stack on Windows (PowerShell 5.1+ / 7+).
.EXAMPLE
  .\scripts\start.ps1                 # native Ollama on the host (recommended)
  .\scripts\start.ps1 -DockerOllama   # Ollama in Docker (CPU)
  .\scripts\start.ps1 -DockerOllama -Gpu   # Ollama in Docker with NVIDIA GPU
#>
param(
    [switch]$DockerOllama,
    [switch]$Gpu,
    [switch]$SkipPull
)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')

if (-not (Test-Path .env)) {
    Copy-Item .env.example .env
    Write-Host 'Created .env from .env.example'
}

$cfg = @{}
foreach ($line in Get-Content .env) {
    if ($line -match '^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$') { $cfg[$Matches[1]] = $Matches[2].Trim('"', "'") }
}
function Get-Cfg($key, $default) { if ($cfg[$key]) { $cfg[$key] } else { $default } }
$chatModel = Get-Cfg 'CHAT_MODEL' 'qwen2.5:7b'
$embedModel = Get-Cfg 'EMBED_MODEL' 'bge-m3'
$webuiPort = Get-Cfg 'WEBUI_PORT' '3000'

$compose = @('compose', '-f', 'docker-compose.yml')
if ($DockerOllama) {
    $env:OLLAMA_BASE_URL = 'http://ollama:11434'
    $compose += @('--profile', 'ollama')
    if ($Gpu) { $compose += @('-f', 'compose.gpu.yml') }
}

function Test-Ollama {
    try { Invoke-RestMethod -Uri 'http://localhost:11434/api/version' -TimeoutSec 3 | Out-Null; $true } catch { $false }
}

if ($DockerOllama) {
    & docker @compose up -d ollama
    if ($LASTEXITCODE -ne 0) { throw 'failed to start the ollama container' }
    Write-Host 'Waiting for Ollama container...'
    for ($i = 0; $i -lt 60 -and -not (Test-Ollama); $i++) { Start-Sleep -Seconds 1 }
}
if (-not (Test-Ollama)) {
    throw "Ollama is not reachable at http://localhost:11434.`nInstall it from https://ollama.com/download and start it, or rerun with -DockerOllama."
}

if (-not $SkipPull) {
    foreach ($m in @($embedModel, $chatModel)) {
        Write-Host "Pulling $m ..."
        if ($DockerOllama) { & docker @compose exec ollama ollama pull $m }
        elseif (Get-Command ollama -ErrorAction SilentlyContinue) { & ollama pull $m }
        else {
            $body = @{ model = $m; stream = $false } | ConvertTo-Json
            Invoke-RestMethod -Method Post -Uri 'http://localhost:11434/api/pull' -Body $body -ContentType 'application/json' -TimeoutSec 7200 | Out-Null
        }
        if ($LASTEXITCODE -and $LASTEXITCODE -ne 0) { throw "failed to pull $m" }
    }
}

New-Item -ItemType Directory -Force -Path (Get-Cfg 'DOCS_PATH' './data/docs'), './data/config', (Get-Cfg 'WORKSPACE_PATH' './data/workspace') | Out-Null
& docker @compose up -d --build
if ($LASTEXITCODE -ne 0) { throw 'docker compose up failed' }

Write-Host ''
Write-Host "Console   : http://localhost:$(Get-Cfg 'INGEST_PORT' '8001')   (documents, models, settings)"
Write-Host "OpenWebUI : http://localhost:$webuiPort   (pick a 'rag/...' model)"
Write-Host "Gateway   : http://localhost:$(Get-Cfg 'GATEWAY_PORT' '8090')/v1"
Write-Host "Ingest API: http://localhost:$(Get-Cfg 'INGEST_PORT' '8001')/docs"
Write-Host "Documents : drop files into $(Get-Cfg 'DOCS_PATH' './data/docs')"
