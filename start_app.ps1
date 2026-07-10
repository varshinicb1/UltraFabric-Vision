<#
.SYNOPSIS
    End-to-end launcher for UltraFabric-Vision (FastAPI AI backend + Vite React dashboard).

.DESCRIPTION
    Frees the target ports, starts the GPU inference backend, waits for it to become
    healthy, then starts the web dashboard (and optionally the Firebase remote-cam
    streamer). Each service runs in its own window so you can read its logs and stop
    it with Ctrl+C. Run Stop-UltraFabric (or .\start_app.ps1 -Stop) to kill everything.

.PARAMETER Stop
    Kill any process listening on the app ports and exit.

.PARAMETER NoFrontend
    Start only the backend AI engine.

.PARAMETER RemoteCam
    Also start the Firebase remote-cam streamer (web_app on 5173, remote_cam on 5174).

.PARAMETER Python
    Python launcher to run the backend with. Defaults to a local venv if present,
    otherwise 'py -3.12'.

.EXAMPLE
    .\start_app.ps1
.EXAMPLE
    .\start_app.ps1 -RemoteCam
.EXAMPLE
    .\start_app.ps1 -Stop
#>
[CmdletBinding()]
param(
    [switch]$Stop,
    [switch]$NoFrontend,
    [switch]$RemoteCam,
    [string]$Python
)

$ErrorActionPreference = 'Stop'
$Root         = $PSScriptRoot
$BackendPort  = 8000
$FrontendPort = 5173
$RemotePort   = 5174

function Write-Step($msg) { Write-Host "[UFV] $msg" -ForegroundColor Cyan }
function Write-Ok($msg)   { Write-Host "[UFV] $msg" -ForegroundColor Green }
function Write-Warn2($msg){ Write-Host "[UFV] $msg" -ForegroundColor Yellow }

function Stop-Port($port) {
    $conns = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue
    foreach ($c in $conns) {
        try {
            Stop-Process -Id $c.OwningProcess -Force -ErrorAction Stop
            Write-Warn2 "Freed port $port (killed PID $($c.OwningProcess))"
        } catch { }
    }
}

function Stop-All {
    Write-Step "Stopping UltraFabric-Vision services..."
    Stop-Port $BackendPort
    Stop-Port $FrontendPort
    Stop-Port $RemotePort
    Write-Ok "All app ports freed."
}

if ($Stop) { Stop-All; return }

# --- Resolve the Python launcher (venv > override > py -3.12 > python) ---
function Resolve-Python {
    param([string]$Override)
    $venv = Join-Path $Root 'venv\Scripts\python.exe'
    if (Test-Path $venv) { return @{ Exe = $venv; Args = @() } }
    if ($Override) {
        $parts = $Override.Split(' ', [StringSplitOptions]::RemoveEmptyEntries)
        return @{ Exe = $parts[0]; Args = @($parts[1..($parts.Length-1)]) }
    }
    if (Get-Command py -ErrorAction SilentlyContinue) { return @{ Exe = 'py'; Args = @('-3.12') } }
    return @{ Exe = 'python'; Args = @() }
}

$py = Resolve-Python -Override $Python
Write-Step "Backend interpreter: $($py.Exe) $($py.Args -join ' ')"

# --- Clean the ports before (re)starting ---
Stop-Port $BackendPort
if (-not $NoFrontend) { Stop-Port $FrontendPort; if ($RemoteCam) { Stop-Port $RemotePort } }

# --- Start backend in its own window ---
Write-Step "Starting FastAPI AI engine on :$BackendPort ..."
$backendArgs = @($py.Args + @('backend_api.py'))
$backend = Start-Process -FilePath $py.Exe -ArgumentList $backendArgs -WorkingDirectory $Root -PassThru
Write-Ok "Backend launched (PID $($backend.Id)). Loading models..."

# --- Wait for /api/health ---
$healthUrl = "http://localhost:$BackendPort/api/health"
$deadline  = (Get-Date).AddSeconds(180)
$healthy   = $false
while ((Get-Date) -lt $deadline) {
    if ($backend.HasExited) { Write-Warn2 "Backend process exited early (code $($backend.ExitCode)). Check its window for errors."; break }
    try {
        $r = Invoke-RestMethod -Uri $healthUrl -TimeoutSec 3 -ErrorAction Stop
        $healthy = $true; break
    } catch { Start-Sleep -Seconds 2 }
}
if ($healthy) { Write-Ok "Backend healthy at $healthUrl" }
else          { Write-Warn2 "Backend did not report healthy within timeout; continuing anyway." }

if ($NoFrontend) {
    Write-Ok "Backend running. Dashboard skipped (-NoFrontend)."
    Write-Host "  AI Backend:  http://localhost:$BackendPort" -ForegroundColor White
    return
}

# --- Ensure frontend deps ---
$webApp = Join-Path $Root 'web_app'
if (-not (Test-Path (Join-Path $webApp 'node_modules'))) {
    Write-Step "Installing web_app dependencies (first run)..."
    Push-Location $webApp; npm install; Pop-Location
}

# --- Start dashboard in its own window ---
Write-Step "Starting Vite dashboard on :$FrontendPort ..."
Start-Process -FilePath 'npm.cmd' -ArgumentList @('run','dev','--','--host') -WorkingDirectory $webApp | Out-Null

if ($RemoteCam) {
    $remoteDir = Join-Path $Root 'remote_cam'
    if (Test-Path $remoteDir) {
        if (-not (Test-Path (Join-Path $remoteDir 'node_modules'))) {
            Write-Step "Installing remote_cam dependencies (first run)..."
            Push-Location $remoteDir; npm install; Pop-Location
        }
        Write-Step "Starting Firebase remote-cam on :$RemotePort ..."
        Start-Process -FilePath 'npm.cmd' -ArgumentList @('run','dev','--','--host','--port',"$RemotePort") -WorkingDirectory $remoteDir | Out-Null
    } else { Write-Warn2 "remote_cam directory not found; skipping." }
}

Start-Sleep -Seconds 2
Write-Host ""
Write-Ok "========================================================"
Write-Ok " UltraFabric-Vision is running"
Write-Host "   Dashboard:  http://localhost:$FrontendPort" -ForegroundColor White
if ($RemoteCam) { Write-Host "   Remote cam: http://localhost:$RemotePort" -ForegroundColor White }
Write-Host "   AI Backend: http://localhost:$BackendPort  (health: /api/health)" -ForegroundColor White
Write-Ok "========================================================"
Write-Host "Stop everything with:  .\start_app.ps1 -Stop" -ForegroundColor DarkGray

try { Start-Process "http://localhost:$FrontendPort" } catch { }
