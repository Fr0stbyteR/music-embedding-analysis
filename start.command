#!/bin/bash
# Finder-friendly macOS launcher. Also works with: bash start.command --basic
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

on_exit() {
    status=$?
    if [ "$status" -ne 0 ] && [ "$status" -ne 130 ]; then
        printf '\nStartup failed (exit %s). Check the message above.\n' "$status"
        printf 'For analysis without model downloads: bash start.command --basic\n'
        if [ -t 0 ]; then
            read -r -p 'Press Return to close...' _ || true
        fi
    fi
}
trap on_exit EXIT

if [ "$(uname -s)" != Darwin ]; then
    printf 'This launcher is for macOS. See README.md for Windows setup.\n' >&2
    exit 1
fi
case "${1:-}" in
    ''|--basic) ;;
    --help|-h)
        printf 'Usage: bash start.command [--basic]\nDefault: CLAP audio/text analysis. --basic: librosa only, no model downloads.\n'
        exit 0 ;;
    *) printf 'Unknown option: %s\n' "$1" >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then
    printf 'Expected at most one option.\n' >&2
    exit 2
fi

if command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
elif [ -x "$PWD/.tools/uv/uv" ]; then
    uv_bin="$PWD/.tools/uv/uv"
else
    printf 'Installing uv locally (no Homebrew or administrator password needed)...\n'
    mkdir -p .tools/uv
    installer="$(mktemp "${TMPDIR:-/tmp}/music-uv.XXXXXX")"
    if ! curl --proto '=https' --tlsv1.2 -LsSf https://astral.sh/uv/install.sh -o "$installer"; then
        rm -f -- "$installer"
        printf 'Cannot download uv. Check your internet connection and retry.\n' >&2
        exit 1
    fi
    if ! UV_INSTALL_DIR="$PWD/.tools/uv" UV_NO_MODIFY_PATH=1 sh "$installer"; then
        rm -f -- "$installer"
        exit 1
    fi
    rm -f -- "$installer"
    uv_bin="$PWD/.tools/uv/uv"
fi

printf 'Preparing Python 3.11 and the backend environment...\n'
"$uv_bin" sync --locked --python 3.11 --inexact
"$uv_bin" run --no-sync python scripts/launch.py --uv "$uv_bin" "$@"
