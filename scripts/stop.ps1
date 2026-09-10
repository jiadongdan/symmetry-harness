# Stop every local Symmetry Harness interface started by any tool or terminal.
# Usage:  & "<skill-directory>\scripts\stop.ps1"

$ErrorActionPreference = "Continue"

$processes = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*symmetry_harness.cli*launch*" }

if (-not $processes) {
    Write-Output "No running Symmetry Harness interface found."
    exit 0
}

foreach ($process in $processes) {
    $pidValue = $process.ProcessId
    Write-Output "Stopping PID $pidValue"
    Stop-Process -Id $pidValue -Force -ErrorAction SilentlyContinue
}

Start-Sleep -Seconds 2

$remaining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -like "*symmetry_harness.cli*launch*" }

if ($remaining) {
    Write-Output "WARNING: $($remaining.Count) process(es) still running."
    exit 1
}

Write-Output "All Symmetry Harness interfaces stopped."
exit 0
