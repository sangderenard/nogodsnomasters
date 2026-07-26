[CmdletBinding(SupportsShouldProcess)]
param(
    [switch]$Commit,
    [switch]$Push,
    [switch]$IncludeRoot,
    [switch]$SetUpstream,
    [string]$Message = "Checkpoint workspace progress"
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$ExcludedNames = @(
    "_quarantine", "archive", "archives", "build", "node_modules",
    ".venv", "venv", "__pycache__"
)

function Invoke-Git {
    param(
        [Parameter(Mandatory)][string]$Repository,
        [Parameter(Mandatory)][string[]]$Arguments,
        [switch]$AllowFailure
    )
    $output = & git -C $Repository @Arguments 2>&1
    if ($LASTEXITCODE -ne 0 -and -not $AllowFailure) {
        throw "git -C `"$Repository`" $($Arguments -join ' ') failed:`n$output"
    }
    return @($output)
}

function Get-WorkspaceRepositories {
    $repositories = [System.Collections.Generic.List[string]]::new()
    if ($IncludeRoot -and (Test-Path -LiteralPath (Join-Path $WorkspaceRoot ".git"))) {
        $repositories.Add($WorkspaceRoot)
    }
    Get-ChildItem -LiteralPath $WorkspaceRoot -Directory -Force |
        Where-Object {
            $_.Name -notin $ExcludedNames -and
            (Test-Path -LiteralPath (Join-Path $_.FullName ".git"))
        } |
        Sort-Object Name |
        ForEach-Object { $repositories.Add($_.FullName) }
    return $repositories
}

$repositories = Get-WorkspaceRepositories
if ($repositories.Count -eq 0) {
    Write-Host "No repositories found."
    exit 0
}

foreach ($repository in $repositories) {
    $name = if ($repository -eq $WorkspaceRoot) { "<workspace-root>" } else {
        Split-Path -Leaf $repository
    }
    $branch = (Invoke-Git $repository @("branch", "--show-current")) -join ""
    $remoteNames = Invoke-Git $repository @("remote")
    $status = Invoke-Git $repository @("status", "--short")

    Write-Host ""
    Write-Host "[$name] branch=$branch"
    foreach ($remoteName in $remoteNames) {
        $url = (Invoke-Git $repository @("remote", "get-url", $remoteName)) -join ""
        Write-Host "  remote $remoteName -> $url"
    }
    if ($status.Count -eq 0) {
        Write-Host "  clean"
    } else {
        $status | ForEach-Object { Write-Host "  $_" }
    }

    if ($Commit -and $status.Count -gt 0) {
        if ($PSCmdlet.ShouldProcess($repository, "stage and commit all changes")) {
            Invoke-Git $repository @("add", "-A") | Out-Null
            $staged = Invoke-Git $repository @("diff", "--cached", "--name-only")
            if ($staged.Count -gt 0) {
                Invoke-Git $repository @("commit", "-m", $Message) |
                    ForEach-Object { Write-Host "  $_" }
            }
        }
    }

    if ($Push) {
        if ($remoteNames.Count -eq 0) {
            Write-Warning "[$name] has no remote; committed changes remain local."
            continue
        }
        if (-not $branch) {
            Write-Warning "[$name] is detached; not pushing."
            continue
        }
        if ($repository -eq $WorkspaceRoot) {
            $urls = foreach ($remoteName in $remoteNames) {
                (Invoke-Git $repository @("remote", "get-url", $remoteName)) -join ""
            }
            if ($urls -match "electrofluid") {
                throw "Refusing to push workspace root to electrofluid. Correct its remote first."
            }
        }
        if ($PSCmdlet.ShouldProcess($repository, "push branch $branch")) {
            $upstream = Invoke-Git $repository @(
                "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}"
            ) -AllowFailure
            if ($LASTEXITCODE -eq 0) {
                Invoke-Git $repository @("push") |
                    ForEach-Object { Write-Host "  $_" }
            } elseif ($SetUpstream) {
                $remote = if ($remoteNames -contains "origin") {
                    "origin"
                } else {
                    $remoteNames[0]
                }
                Invoke-Git $repository @("push", "--set-upstream", $remote, $branch) |
                    ForEach-Object { Write-Host "  $_" }
            } else {
                Write-Warning "[$name] has no upstream; rerun with -SetUpstream."
            }
        }
    }
}
