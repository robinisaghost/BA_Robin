# Sync thesis/ folder to Overleaf
# Usage: .\sync_overleaf.ps1 -Token "your_token_here"
param(
    [Parameter(Mandatory=$true)]
    [string]$Token
)

$OVERLEAF_URL = "https://git:$Token@git.overleaf.com/6a181d6c1ad66b18cf71fe38"

# Make sure we are on main
$branch = git rev-parse --abbrev-ref HEAD
if ($branch -ne "main") {
    Write-Host "Switching to main first..."
    git checkout main
}

# Fetch Overleaf's current master
Write-Host "Fetching Overleaf state..."
git fetch $OVERLEAF_URL master

# Get tree hash of thesis/ from local HEAD
$thesis_tree = git rev-parse HEAD:thesis
if (-not $thesis_tree) {
    Write-Error "Could not find thesis/ folder in current commit."
    exit 1
}

# Create new commit on top of Overleaf's FETCH_HEAD with our thesis content
Write-Host "Creating sync commit..."
$new_commit = "sync thesis from local" | git commit-tree $thesis_tree -p FETCH_HEAD
if (-not $new_commit) {
    Write-Error "Failed to create commit."
    exit 1
}

# Push to Overleaf
Write-Host "Pushing to Overleaf..."
git push $OVERLEAF_URL "${new_commit}:master"

Write-Host "Done. Overleaf will recompile automatically."
