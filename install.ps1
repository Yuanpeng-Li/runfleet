# Install or update RunFleet in one line (Windows 10/11; Windows PowerShell 5.1 or PowerShell 7):
#
#   powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.ps1 | iex"
#
# Installed already: updated right here with `uv tool upgrade runfleet` (or `pipx reinstall runfleet`). Unlike
# install.sh this does not hand over to `runfleet update`, which on Windows only prints these steps (a running
# runfleet.exe cannot be replaced), so RunFleet must not be open: the script checks first.
# Otherwise: uv first when missing (Astral's PowerShell installer, which also puts %USERPROFILE%\.local\bin on the
# user PATH), then `uv tool install git+https://github.com/Yuanpeng-Li/runfleet`. uv fetches git URLs with the git
# CLI, so Git for Windows is a prerequisite. Nothing is installed on your clusters or workstations.
#
# Everything runs inside one script block: under `irm | iex` a top-level `exit` would close the user's window and
# variables would leak into their session, so errors `throw` and the normal end is `return`. ASCII only: Windows
# PowerShell 5.1 reads a script file without a BOM in the ANSI code page.
& {
    # Local to this block: a session that set 'Stop' would make Windows PowerShell 5.1 throw on the first line a
    # native program writes to a redirected stderr (2>$null). Failures are found by $LASTEXITCODE and end in Fail.
    $ErrorActionPreference = 'Continue'
    $Repo = 'https://github.com/Yuanpeng-Li/runfleet'
    $UvInstaller = 'https://astral.sh/uv/install.ps1'

    function Say([string]$Message) { Write-Host "runfleet-install: $Message" }
    function Fail([string]$Message) {
        Write-Host "runfleet-install: error: $Message" -ForegroundColor Red
        throw 'runfleet-install failed'
    }

    # A program on PATH (any PATHEXT) or in the usual install dirs (a fresh install is not on this session's PATH yet).
    function Find-Bin([string]$Name) {
        $cmd = Get-Command $Name -CommandType Application -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($cmd) { return $cmd.Source }
        foreach ($dir in @($env:UV_INSTALL_DIR, $env:XDG_BIN_HOME, (Join-Path $HOME '.local\bin'), (Join-Path $HOME '.cargo\bin'))) {
            if (-not $dir) { continue }
            $exe = Join-Path $dir "$Name.exe"
            if (Test-Path -LiteralPath $exe -PathType Leaf) { return $exe }
        }
        return $null
    }

    # Native commands: check $LASTEXITCODE (Windows PowerShell 5.1 has no $PSNativeCommandUseErrorActionPreference).
    function Invoke-Native([string]$Exe, [string]$Hint) {
        & $Exe @args
        if ($LASTEXITCODE -ne 0) {
            Fail "$([IO.Path]::GetFileName($Exe)) $($args -join ' ') exited with $LASTEXITCODE$Hint"
        }
    }

    if (-not (Get-Command ssh -CommandType Application -ErrorAction SilentlyContinue)) {
        Say 'note: no ssh client on PATH; RunFleet needs OpenSSH to reach your hosts (runfleet --demo works without).'
        Say '      Settings > System > Optional features > OpenSSH Client, or (admin): Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0'
    }

    $rf = Find-Bin 'runfleet'
    if ($rf) {
        if (Get-Process -Name runfleet -ErrorAction SilentlyContinue) {
            Fail 'RunFleet is running; quit it (q) everywhere, then run this again (a running runfleet.exe cannot be replaced)'
        }
        Say "updating $((& $rf --version 2>$null | Out-String).Trim())"
        $busy = ' (if RunFleet is open, quit it and run this again)'
        $uv = Find-Bin 'uv'
        if ($uv -and ((& $uv tool list 2>$null | Out-String) -match '(?m)^runfleet ')) {
            Invoke-Native $uv $busy tool upgrade runfleet
        } elseif ((Get-Command pipx -ErrorAction SilentlyContinue) -and ((& pipx list --short 2>$null | Out-String) -match '(?m)^runfleet ')) {
            Invoke-Native (Get-Command pipx).Source $busy reinstall runfleet
        } else {
            Fail "$rf was not installed by uv or pipx; update it the way you installed it"
        }
        Say "done: $((& $rf --version 2>$null | Out-String).Trim()). Start it with: runfleet"
        return
    }

    if (-not (Get-Command git -CommandType Application -ErrorAction SilentlyContinue)) {
        Fail ('uv needs Git to install from GitHub. Install Git for Windows (winget install --id Git.Git -e ' +
              '--source winget, or https://git-scm.com/download/win), open a new terminal and run this again.')
    }

    $uv = Find-Bin 'uv'
    if (-not $uv) {
        Say 'installing uv (Python package manager, https://docs.astral.sh/uv/) ...'
        # In a child shell: uv's installer defines functions and variables of its own and checks the execution policy.
        $shell = (Get-Process -Id $PID).Path
        & $shell -NoProfile -ExecutionPolicy Bypass -Command "irm $UvInstaller | iex"
        if ($LASTEXITCODE -ne 0) { Fail "the uv installer failed (exit $LASTEXITCODE)" }
        $uv = Find-Bin 'uv'
        if (-not $uv) { Fail 'uv was installed but not found; open a new terminal and run this again' }
    }

    Say "installing RunFleet from $Repo ..."
    Invoke-Native $uv '' tool install "git+$Repo"

    $binDir = (& $uv tool dir --bin 2>$null | Out-String).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $binDir) { $binDir = Join-Path $HOME '.local\bin' }
    if (Get-Command runfleet -CommandType Application -ErrorAction SilentlyContinue) {
        Say "done: $((& runfleet --version | Out-String).Trim()). Start it with: runfleet   (or try: runfleet --demo)"
    } else {
        & $uv tool update-shell *> $null      # adds the bin dir to the user PATH (registry); new terminals see it
        $env:Path = "$binDir;$env:Path"       # ... and this session too
        Say "done: $((& (Join-Path $binDir 'runfleet.exe') --version | Out-String).Trim())."
        Say 'Start it with: runfleet   (or try: runfleet --demo). Other open terminals need restarting to find it.'
    }
}
