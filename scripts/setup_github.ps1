# One-time setup: put this project on GitHub.
# Run via setup_github.bat in the project root (double-click it).
param(
    [string]$RepoName = "credit-card-fraud-detection",
    [ValidateSet("public", "private")][string]$Visibility = "public"
)
$ErrorActionPreference = "Continue"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

function Fail($msg) { Write-Host $msg -ForegroundColor Red; exit 1 }
function Step($msg) { Write-Host "`n==> $msg" -ForegroundColor Cyan }

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    Fail "Git is not installed. Install it (winget install Git.Git, or https://git-scm.com/download/win), reopen this window and run again."
}

Step "Checking your git identity"
if (-not (git config --global user.name))  { git config --global user.name  (Read-Host "Your name for commits") }
if (-not (git config --global user.email)) { git config --global user.email (Read-Host "The email on your GitHub account") }

Step "Creating the local repository"
if (-not (Test-Path ".git")) { git init -b main | Out-Null }
git add -A

$big = git diff --cached --name-only | Where-Object { (Test-Path $_) -and ((Get-Item $_).Length -gt 50MB) }
if ($big) { git reset -q; Fail "These files are over 50 MB (GitHub rejects them). Add them to .gitignore first:`n$($big -join "`n")" }

git diff --cached --quiet
if ($LASTEXITCODE -ne 0) { git commit -q -m "Initial commit: credit card fraud detection project" }
git branch -M main

Step "Connecting to GitHub"
$remote = git remote get-url origin 2>&1
if ($LASTEXITCODE -ne 0) {
    if (Get-Command gh -ErrorAction SilentlyContinue) {
        gh auth status *> $null
        if ($LASTEXITCODE -ne 0) { gh auth login --web --git-protocol https }
        gh repo create $RepoName "--$Visibility" --source . --remote origin
        if ($LASTEXITCODE -ne 0) { Fail "Could not create the GitHub repo (does one with this name already exist?)." }
    } else {
        Write-Host "Opening GitHub: create an EMPTY $Visibility repo named '$RepoName'"
        Write-Host "(leave 'Add a README', '.gitignore' and 'license' all unticked)."
        Start-Process "https://github.com/new?name=$RepoName&visibility=$Visibility"
        $url = Read-Host "Paste the repo URL here (https://github.com/<your-username>/$RepoName.git)"
        git remote add origin $url.Trim()
    }
}

Step "Uploading (a browser window may ask you to sign in to GitHub once)"
git push -u origin main
if ($LASTEXITCODE -ne 0) { Fail "Push failed - see the message above." }

$url = (git remote get-url origin) -replace '\.git$', ''
Write-Host "`nDone. Your project is at $url" -ForegroundColor Green
Write-Host "From now on, commit and push from VS Code (Source Control panel) or with git commit / git push."
