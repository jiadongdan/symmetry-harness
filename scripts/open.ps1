Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Write-BlockedResult {
    param([string]$Issue)

    [ordered]@{
        status = "blocked"
        phase = "runtime"
        issues = @($Issue)
        recommendations = @("Run the symmetry-harness installer once.")
    } | ConvertTo-Json -Compress
}

function ConvertTo-NativeArgument {
    param([string]$Value)

    if ($Value.Length -gt 0 -and $Value -notmatch '[\s"]') {
        return $Value
    }
    $escaped = $Value -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

try {
    $stateDirectory = if ($env:SYMMETRY_HARNESS_HOME) {
        [System.IO.Path]::GetFullPath($env:SYMMETRY_HARNESS_HOME)
    }
    else {
        Join-Path ([Environment]::GetFolderPath("UserProfile")) ".symmetry-harness"
    }
    $pythonPathFile = Join-Path $stateDirectory "python-path.txt"
    $configPathFile = Join-Path $stateDirectory "config-path.txt"
    if (-not (Test-Path -LiteralPath $pythonPathFile -PathType Leaf) -or
        -not (Test-Path -LiteralPath $configPathFile -PathType Leaf)) {
        Write-BlockedResult "The local symmetry-harness runtime has not been installed."
        exit 2
    }

    $harnessPython = (Get-Content -Raw -LiteralPath $pythonPathFile).Trim()
    $configPath = (Get-Content -Raw -LiteralPath $configPathFile).Trim()
    if (-not (Test-Path -LiteralPath $harnessPython -PathType Leaf)) {
        Write-BlockedResult "The recorded symmetry-harness Python executable does not exist."
        exit 2
    }
    if (-not (Test-Path -LiteralPath $configPath -PathType Leaf)) {
        Write-BlockedResult "The recorded symmetry-harness configuration does not exist."
        exit 2
    }

    $logDirectory = Join-Path $stateDirectory "logs"
    New-Item -ItemType Directory -Force -Path $logDirectory | Out-Null
    $stdoutPath = Join-Path $logDirectory "fast-open.stdout.json"
    $stderrPath = Join-Path $logDirectory "fast-open.stderr.log"
    Remove-Item -LiteralPath $stdoutPath -Force -ErrorAction SilentlyContinue
    Remove-Item -LiteralPath $stderrPath -Force -ErrorAction SilentlyContinue

    $launchArguments = @(
        "-m",
        "symmetry_harness.cli",
        "launch",
        "--config",
        $configPath,
        "--server-port",
        "7860",
        "--no-inbrowser"
    ) + $args
    $argumentLine = ($launchArguments | ForEach-Object {
        ConvertTo-NativeArgument ([string]$_)
    }) -join " "
    $process = Start-Process `
        -FilePath $harnessPython `
        -ArgumentList $argumentLine `
        -WindowStyle Hidden `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru

    $ErrorActionPreference = "Continue"
    $PSNativeCommandUseErrorActionPreference = $false
    & $harnessPython -m symmetry_harness.cli wait `
        --server-port 7860 `
        --timeout-seconds 180 `
        --launch-pid $process.Id `
        --launch-output $stdoutPath `
        --launch-error $stderrPath
    exit $LASTEXITCODE
}
catch {
    Write-BlockedResult $_.Exception.Message
    exit 2
}
