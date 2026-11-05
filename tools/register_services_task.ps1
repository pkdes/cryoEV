<#
.SYNOPSIS
  Register (or remove) the scheduled task "cryoev-services": at logon, run `tools\services.py start all`.

.DESCRIPTION
  One task, run as the current user (Interactive logon, limited rights: no admin needed), StartWhenAvailable,
  no time limit. Idempotent: running it again replaces the task. -Unregister removes it and leaves running
  services alone (stop them with `python tools\services.py stop all`).
  Registering starts nothing now; `Start-ScheduledTask cryoev-services` runs it once.
  The task uses pythonw.exe next to -Python when it exists (no console window at logon); services.py then
  logs its own output to %USERPROFILE%\.cryoev\logs\services.log.
  Run it from the main checkout, not a worktree: the task points at this script's repo.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\register_services_task.ps1
  powershell -ExecutionPolicy Bypass -File tools\register_services_task.ps1 -Python C:\path\to\python.exe
  powershell -ExecutionPolicy Bypass -File tools\register_services_task.ps1 -Unregister
#>
param(
    [string]$Python = '',
    [string]$Config = '',
    [string]$TaskName = 'cryoev-services',
    [int]$DelaySeconds = 30,
    [switch]$Unregister
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$script = Join-Path $PSScriptRoot 'services.py'

if ($Unregister) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "[OK] removed task $TaskName (running services left as they are)"
    } else {
        Write-Host "[OK] no task named $TaskName"
    }
    return
}

if (-not (Test-Path $script)) { throw "services.py not found at $script" }
if (-not $Python) {
    $cmd = Get-Command python -ErrorAction SilentlyContinue
    if (-not $cmd) { throw 'python is not on PATH; pass -Python <full path to python.exe>' }
    $Python = $cmd.Source
}
# resolve the real interpreter (PATH may hold the WindowsApps alias, which a logon task cannot rely on)
$Python = (& $Python -c 'import sys; print(sys.executable)').Trim()
if (-not (Test-Path $Python)) { throw "python not found: $Python" }
$exe = Join-Path (Split-Path -Parent $Python) 'pythonw.exe'
if (-not (Test-Path $exe)) { $exe = $Python }

$argList = '"' + $script + '"'
if ($Config) { $argList += ' --config "' + $Config + '"' }
$argList += ' start all'

$user = [System.Security.Principal.WindowsIdentity]::GetCurrent().Name
$action = New-ScheduledTaskAction -Execute $exe -Argument $argList -WorkingDirectory $repo
$trigger = New-ScheduledTaskTrigger -AtLogOn -User $user
if ($DelaySeconds -gt 0) { $trigger.Delay = 'PT' + $DelaySeconds + 'S' }  # let the network / Tailscale come up
$principal = New-ScheduledTaskPrincipal -UserId $user -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
    -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal `
    -Settings $settings -Description 'cryoEV: start the laptop services (tools\services.py start all) at logon' `
    -Force | Out-Null
Write-Host "[OK] task $TaskName registered for $user"
Write-Host "     $exe $argList"
Write-Host "     run once now: Start-ScheduledTask $TaskName ; check: python tools\services.py status"
