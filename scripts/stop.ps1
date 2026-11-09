# Stop the sh-rag stack (data volumes are kept). Pass -Purge to delete them too.
param([switch]$Purge)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$extra = @(); if ($Purge) { $extra += '-v' }
& docker compose -f docker-compose.yml --profile ollama down @extra
