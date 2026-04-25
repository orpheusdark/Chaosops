param(
    [Parameter(Mandatory = $false)]
    [string]$HFUsername = "",

    [Parameter(Mandatory = $false)]
    [string]$SpaceName = "chaosops"
)

$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path "$PSScriptRoot\.."

git -C "$repoRoot" rev-parse --is-inside-work-tree | Out-Null
if ($LASTEXITCODE -ne 0) {
    throw "Not a git repository: $repoRoot"
}

git -C "$repoRoot" config core.hooksPath .githooks

if (-not [string]::IsNullOrWhiteSpace($HFUsername)) {
    $repoUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"
    if ((git -C "$repoRoot" remote) -contains "hf") {
        git -C "$repoRoot" remote set-url hf $repoUrl
    } else {
        git -C "$repoRoot" remote add hf $repoUrl
    }
    Write-Host "HF remote configured: $repoUrl"
} elseif ((git -C "$repoRoot" remote) -contains "hf") {
    Write-Host "HF remote already exists: $(git -C "$repoRoot" remote get-url hf)"
} else {
    Write-Warning "HF remote not configured. Re-run with -HFUsername to enable HF auto-push."
}

if (-not ((git -C "$repoRoot" remote) -contains "origin")) {
    Write-Warning "GitHub remote origin is missing. Configure it to enable GitHub auto-push."
}

Write-Host "Auto push is enabled."
Write-Host "Hooks path: $(git -C "$repoRoot" config --get core.hooksPath)"
Write-Host "Remotes:"
git -C "$repoRoot" remote -v
