param(
    [Parameter(Mandatory = $true)]
    [string]$HFUsername,

    [Parameter(Mandatory = $false)]
    [string]$SpaceName = "chaosops",

    [Parameter(Mandatory = $false)]
    [string]$HFToken = ""
)

$ErrorActionPreference = "Stop"

Set-Location "$PSScriptRoot\.."

if (-not [string]::IsNullOrWhiteSpace($HFToken)) {
    hf auth login --token $HFToken
}

$who = hf auth whoami
if ($LASTEXITCODE -ne 0) {
    throw "Hugging Face login required. Run: hf auth login"
}

hf repo create "$SpaceName" --type space --space_sdk docker -y
if ($LASTEXITCODE -ne 0) {
    Write-Host "Repo may already exist. Continuing..."
}

$repoUrl = "https://huggingface.co/spaces/$HFUsername/$SpaceName"

git remote remove hf 2>$null

git remote add hf $repoUrl

git add .
$hasChanges = (git status --porcelain)
if (-not [string]::IsNullOrWhiteSpace($hasChanges)) {
    git commit -m "Prepare Hugging Face Space deployment"
}

git push hf main

Write-Host "Published to: $repoUrl"
