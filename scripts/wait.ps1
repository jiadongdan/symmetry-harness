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
    if (-not (Test-Path -LiteralPath $pythonPathFile -PathType Leaf)) {
        Write-BlockedResult "The local symmetry-harness runtime has not been installed."
        exit 2
    }

    $harnessPython = (Get-Content -Raw -LiteralPath $pythonPathFile).Trim()
    if (-not (Test-Path -LiteralPath $harnessPython -PathType Leaf)) {
        Write-BlockedResult "The recorded symmetry-harness Python executable does not exist."
        exit 2
    }

    & $harnessPython -m symmetry_harness.cli wait --server-port 7860 @args
    exit $LASTEXITCODE
}
catch {
    Write-BlockedResult $_.Exception.Message
    exit 2
}
