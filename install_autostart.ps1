<#
.SYNOPSIS
    Register G.I.D.E.O.N to start on its own.

.DESCRIPTION
    Installs two separate things, because they have different lifetimes:

      Engine (run_engine.py) -> Scheduled Task, at BOOT, 45s delayed
      Tray   (tray + hotkey) -> shortcut in the Startup folder, at LOGON

    They are split because a Windows service cannot show a tray icon or a
    window (Session 0 isolation), while the engine genuinely needs to run
    before anyone logs in — that is what makes reminders and Telegram survive
    a locked or logged-out machine.

    The 45-second delay is not superstition. Ollama has its own autostart, and
    ChromaDB, Kokoro and Piper all load during boot's busiest moment; starting
    immediately means losing the race and failing the first request.

.PARAMETER Uninstall
    Remove both the scheduled task and the Startup shortcut.

.PARAMETER EngineOnly
    Install the boot task but not the logon shortcut.

.PARAMETER TrayOnly
    Install the logon shortcut but not the boot task.

.PARAMETER DelaySeconds
    Override the post-boot delay before the engine starts.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install_autostart.ps1

.NOTES
    Registering a boot task needs an elevated shell. The Startup shortcut does
    not, so -TrayOnly works unelevated.

    Known limitation: Ollama's Windows app starts at logon, not at boot. Until
    you log in, the engine is up and reminders fire, but anything needing the
    LLM (a Telegram conversation) has no model to talk to. Install Ollama as a
    service if you want inference before login.
#>

[CmdletBinding()]
param(
    [switch]$Uninstall,
    [switch]$EngineOnly,
    [switch]$TrayOnly,
    [int]$DelaySeconds = 45
)

$ErrorActionPreference = 'Stop'

$TaskName     = 'GIDEON Engine'
$ShortcutName = 'G.I.D.E.O.N.lnk'
$ProjectDir   = $PSScriptRoot
$StartupDir   = [Environment]::GetFolderPath('Startup')
$ShortcutPath = Join-Path $StartupDir $ShortcutName

function Test-Elevated {
    $identity  = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Get-Pythonw {
    # pythonw.exe, not python.exe: the console window is the whole difference
    # between "starts silently" and "a black box sitting on your desktop".
    $candidate = Join-Path $ProjectDir '.venv\Scripts\pythonw.exe'
    if (Test-Path $candidate) { return $candidate }
    throw "Could not find $candidate. Create the venv first (uv sync)."
}

# ---------------------------------------------------------------------------
#  Uninstall
# ---------------------------------------------------------------------------
if ($Uninstall) {
    $removed = @()

    $existing = Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue
    if ($existing) {
        if (-not (Test-Elevated)) {
            throw "Removing '$TaskName' needs an elevated PowerShell."
        }
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        $removed += "scheduled task '$TaskName'"
    }

    if (Test-Path $ShortcutPath) {
        Remove-Item $ShortcutPath -Force
        $removed += "startup shortcut"
    }

    if ($removed.Count -eq 0) {
        Write-Host "Nothing to remove." -ForegroundColor Yellow
    } else {
        Write-Host ("Removed: " + ($removed -join ', ')) -ForegroundColor Green
    }
    return
}

$pythonw = Get-Pythonw

# ---------------------------------------------------------------------------
#  Engine: scheduled task at boot
# ---------------------------------------------------------------------------
if (-not $TrayOnly) {
    if (-not (Test-Elevated)) {
        throw "Registering the boot task needs an elevated PowerShell. Re-run as Administrator, or use -TrayOnly."
    }

    # run_engine.py rather than '-m uvicorn' on purpose. pythonw.exe has no
    # console, so sys.stderr is None and uvicorn dies on its first log line
    # with exit 1 — before binding, with no console for the traceback to
    # appear in. The symptom is simply "it does not start at boot". The
    # launcher binds logs\engine.log first, so failures are also readable.
    $engineScript = Join-Path $ProjectDir 'run_engine.py'
    if (-not (Test-Path $engineScript)) { throw "Missing $engineScript" }

    $action = New-ScheduledTaskAction `
        -Execute $pythonw `
        -Argument ('"' + $engineScript + '"') `
        -WorkingDirectory $ProjectDir

    $trigger = New-ScheduledTaskTrigger -AtStartup
    $trigger.Delay = "PT${DelaySeconds}S"

    # S4U runs as you, without a stored password and without requiring a
    # login session. Running as SYSTEM instead would use a different profile
    # and lose the user-scoped Ollama and environment entirely.
    $principal = New-ScheduledTaskPrincipal `
        -UserId "$env:USERDOMAIN\$env:USERNAME" `
        -LogonType S4U `
        -RunLevel Limited

    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries `
        -DontStopIfGoingOnBatteries `
        -StartWhenAvailable `
        -RestartCount 3 `
        -RestartInterval (New-TimeSpan -Minutes 1) `
        -ExecutionTimeLimit ([TimeSpan]::Zero)

    # StopIfGoingOnBatteries defaults on for laptops and would kill the engine
    # the moment you unplug — exactly when you are away from the desk and most
    # need reminders.
    $settings.DisallowStartIfOnBatteries = $false

    Register-ScheduledTask `
        -TaskName $TaskName `
        -Action $action `
        -Trigger $trigger `
        -Principal $principal `
        -Settings $settings `
        -Description 'G.I.D.E.O.N engine: reminders, Telegram and the local API.' `
        -Force | Out-Null

    Write-Host "Installed scheduled task '$TaskName' (at boot, +${DelaySeconds}s)." -ForegroundColor Green
}

# ---------------------------------------------------------------------------
#  Tray: shortcut in the Startup folder at logon
# ---------------------------------------------------------------------------
if (-not $EngineOnly) {
    $trayScript = Join-Path $ProjectDir 'gideon_tray.py'
    if (-not (Test-Path $trayScript)) { throw "Missing $trayScript" }

    $iconPath = Join-Path $ProjectDir 'static\gideon.ico'
    if (-not (Test-Path $iconPath)) {
        # The tray writes this on first run; until then fall back to the
        # interpreter's own icon so the shortcut is never broken.
        $iconPath = $pythonw
    }

    $shell    = New-Object -ComObject WScript.Shell
    $shortcut = $shell.CreateShortcut($ShortcutPath)
    $shortcut.TargetPath       = $pythonw
    $shortcut.Arguments        = '"' + $trayScript + '"'
    $shortcut.WorkingDirectory = $ProjectDir
    $shortcut.IconLocation     = $iconPath
    $shortcut.Description      = 'G.I.D.E.O.N — tray, global hotkey and quick ask'
    $shortcut.Save()

    Write-Host "Installed startup shortcut: $ShortcutPath" -ForegroundColor Green
    Write-Host "Right-click it and 'Pin to taskbar' for a click-to-summon icon." -ForegroundColor DarkGray
}

Write-Host ""
Write-Host "Done. Reboot to verify, or start now with:" -ForegroundColor Cyan
Write-Host "    Start-ScheduledTask -TaskName '$TaskName'"
Write-Host "    & '$pythonw' '$(Join-Path $ProjectDir 'gideon_tray.py')'"
