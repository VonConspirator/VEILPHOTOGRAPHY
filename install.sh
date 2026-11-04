#!/bin/sh
# Veil installer: a small shell script that makes sure Python 3 exists, then runs the real installer (installer/cli.py).
# Run it from a copy of the Veil folder:   ./install.sh        (add --dry-run first if you want to see what it would do)
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
if ! command -v python3 >/dev/null 2>&1; then
  echo "Veil needs Python 3.10 or newer, and 'python3' was not found."
  if command -v apt-get >/dev/null 2>&1; then
    echo "On Debian, Ubuntu or Mint run:   sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip curl zstd"
  elif command -v dnf >/dev/null 2>&1; then
    echo "On Fedora run:   sudo dnf install -y python3 python3-pip curl zstd"
  elif command -v pacman >/dev/null 2>&1; then
    echo "On Arch run:   sudo pacman -S --needed python python-pip curl zstd"
  else
    echo "Install Python 3.10+ with your package manager."
  fi
  echo "Then run ./install.sh again."
  exit 2
fi
cd "$HERE"
exec python3 -I -c 'import sys; sys.path.insert(0, sys.argv[1]); from installer.cli import main; sys.exit(main(sys.argv[2:]))' "$HERE" "$@"
