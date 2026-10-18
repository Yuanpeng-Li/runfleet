#!/bin/sh
# Install or update RunFleet in one line (Linux, macOS, WSL):
#
#   curl -LsSf https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.sh | sh
#
# Installed already: `runfleet update` (older versions without it: `uv tool upgrade` / `pipx reinstall`).
# Otherwise: uv first when missing (Astral's installer, which also adds ~/.local/bin to your shell's PATH), then
# `uv tool install git+https://github.com/Yuanpeng-Li/runfleet`; uv downloads a Python 3.11+ itself when the system
# has none. Nothing is installed on your clusters or workstations. Run it again any time to update. Either way an
# agent skill installed with `runfleet skill install` is refreshed to the new version.
#
# Remove RunFleet (its agent skill first, then the tool; your config and state are kept and their paths printed):
#
#   curl -LsSf https://raw.githubusercontent.com/Yuanpeng-Li/runfleet/main/install.sh | sh -s -- --uninstall
set -eu

REPO="https://github.com/Yuanpeng-Li/runfleet"
UV_INSTALLER="https://astral.sh/uv/install.sh"

say() { printf 'runfleet-install: %s\n' "$*"; }
die() { say "error: $*" >&2; exit 1; }

download() {
    if command -v curl >/dev/null 2>&1; then
        curl -LsSf "$1"
    elif command -v wget >/dev/null 2>&1; then
        wget -qO- "$1"
    else
        die "need curl or wget to download $1"
    fi
}

# $1 = program; found on PATH or in the usual install dirs (a fresh install is not on this shell's PATH yet)
find_bin() {
    if command -v "$1" >/dev/null 2>&1; then
        command -v "$1"
        return 0
    fi
    for dir in "${UV_INSTALL_DIR:-}" "${XDG_BIN_HOME:-}" "$HOME/.local/bin" "$HOME/.cargo/bin"; do
        if [ -n "$dir" ] && [ -x "$dir/$1" ]; then
            printf '%s\n' "$dir/$1"
            return 0
        fi
    done
    return 1
}

# $1 = runfleet; refresh the agent skill if one is installed. The capability check: a version without
# `skill update` prints the main help or a usage error here, never --if-installed.
refresh_skill() {
    if "$1" skill update --help 2>/dev/null | grep -q -- '--if-installed'; then
        "$1" skill update --if-installed || true
    fi
}

uninstall() {
    rf=$(find_bin runfleet) || rf=""
    if [ -n "$rf" ] && "$rf" skill uninstall --help 2>/dev/null | grep -q -- '--dry-run'; then
        "$rf" skill uninstall || say "warning: \`runfleet skill uninstall\` failed (see above); RunFleet is removed anyway"
    fi
    if uv=$(find_bin uv) && "$uv" tool list 2>/dev/null | grep -q '^runfleet '; then
        say "removing RunFleet (uv tool uninstall runfleet) …"
        "$uv" tool uninstall runfleet
    elif command -v pipx >/dev/null 2>&1 && pipx list --short 2>/dev/null | grep -q '^runfleet '; then
        say "removing RunFleet (pipx uninstall runfleet) …"
        pipx uninstall runfleet
    elif [ -n "$rf" ]; then
        die "$rf was not installed by uv or pipx; remove it the way you installed it"
    else
        say "RunFleet is not installed (no runfleet found)"
    fi
    # where platformdirs keeps RunFleet's config, state and cache (runfleet.paths)
    if [ "$(uname -s 2>/dev/null || true)" = Darwin ]; then
        set -- "$HOME/Library/Application Support/runfleet" "$HOME/Library/Caches/runfleet"
    else
        set -- "${XDG_CONFIG_HOME:-$HOME/.config}/runfleet" "${XDG_STATE_HOME:-$HOME/.local/state}/runfleet" \
            "${XDG_CACHE_HOME:-$HOME/.cache}/runfleet"
    fi
    kept=""
    for dir in "$@"; do
        # quoted for the rm line: ' becomes '\''
        if [ -d "$dir" ]; then kept="$kept '$(printf '%s' "$dir" | sed "s/'/'\\\\''/g")'"; fi
    done
    if [ -n "$kept" ]; then
        say "kept your config and RunFleet's state (hosts, cache, log):$kept"
        say "to delete them too: rm -rf$kept"
    fi
    say "done. Your clusters and workstations were never changed."
}

if [ "${1:-}" = "--uninstall" ]; then
    uninstall
    exit 0
elif [ $# -gt 0 ]; then
    die "unknown option: $1 (only --uninstall)"
fi

command -v ssh >/dev/null 2>&1 ||
    say "note: no ssh client on PATH; RunFleet needs OpenSSH to reach your hosts (runfleet --demo works without)"

if rf=$(find_bin runfleet); then
    # `update --help` of a version without the subcommand is the main help (`update` taken for a host name)
    if "$rf" update --help 2>/dev/null | grep -q -- '--check'; then
        rc=0
        "$rf" update || rc=$?
        refresh_skill "$rf"
        exit "$rc"
    fi
    say "updating $("$rf" --version 2>/dev/null || echo RunFleet) (a version without \`runfleet update\`)"
    if uv=$(find_bin uv) && "$uv" tool list 2>/dev/null | grep -q '^runfleet '; then
        "$uv" tool upgrade runfleet
    elif command -v pipx >/dev/null 2>&1 && pipx list --short 2>/dev/null | grep -q '^runfleet '; then
        pipx reinstall runfleet
    else
        die "$rf was not installed by uv or pipx; update it the way you installed it"
    fi
    refresh_skill "$rf"
    say "done: $("$rf" --version). If RunFleet is open, quit it and start it again."
    exit 0
fi

if ! uv=$(find_bin uv); then
    say "installing uv (Python package manager, https://docs.astral.sh/uv/) …"
    download "$UV_INSTALLER" | sh
    uv=$(find_bin uv) || die "uv was installed but not found; open a new terminal and run this again"
fi

say "installing RunFleet from $REPO …"
"$uv" tool install "git+$REPO"

bin_dir=$("$uv" tool dir --bin 2>/dev/null || printf '%s' "$HOME/.local/bin")
if command -v runfleet >/dev/null 2>&1; then
    refresh_skill runfleet
    say "done: $(runfleet --version). Start it with: runfleet   (or try: runfleet --demo)"
else
    "$uv" tool update-shell >/dev/null 2>&1 || true
    refresh_skill "$bin_dir/runfleet"
    say "done: $("$bin_dir/runfleet" --version). Open a new terminal (or: export PATH=\"$bin_dir:\$PATH\"),"
    say "then start it with: runfleet   (or try: runfleet --demo)"
fi
