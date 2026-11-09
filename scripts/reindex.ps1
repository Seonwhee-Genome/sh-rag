# Re-sync the documents folder into the index.
#   .\scripts\reindex.ps1             only changed files
#   .\scripts\reindex.ps1 -Force      re-embed everything
#   .\scripts\reindex.ps1 -Recreate   drop the collection first (after changing EMBED_MODEL)
param([switch]$Force, [switch]$Recreate)
$ErrorActionPreference = 'Stop'
Set-Location (Join-Path $PSScriptRoot '..')
$cmdArgs = @('scan'); if ($Force) { $cmdArgs += '--force' }; if ($Recreate) { $cmdArgs += '--recreate' }
& docker compose exec ingest python -m ingest @cmdArgs
