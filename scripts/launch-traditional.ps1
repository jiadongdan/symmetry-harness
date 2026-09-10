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

    $launchArguments = @(
        "-m",
        "symmetry_harness.cli",
        "validate-traditional",
        "--config",
        $configPath,
        "--server-port",
        "0",
        "--no-inbrowser"
    ) + $args
    & $harnessPython @launchArguments
    exit $LASTEXITCODE
}
catch {
    Write-BlockedResult $_.Exception.Message
    exit 2
}
