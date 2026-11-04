#!/bin/sh
# Removes what the installer set up (never your photos). Same as:  ./install.sh --uninstall
exec "$(cd "$(dirname "$0")" && pwd)/install.sh" --uninstall "$@"
