#!/usr/bin/env bash
# Clones CaBRNet from its public GitHub repository, pinned to the exact commit
# that corresponds to the published 1.2 release (verified identical to the 1.2
# tag and to the PyPI 1.2 sdist), then installs it plus this repo's plugin
# package (beyond_l2_ext) in editable mode. Run from the repo root, ideally
# inside an activated virtualenv.
set -euo pipefail

CABRNET_COMMIT=1f3ac21df700c6fa98971458d6f672d0dee426c4
CABRNET_DIR="${1:-./cabrnet-src}"

py_ver=$(python3 -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')
case "$py_ver" in
  3.10|3.11|3.12) ;;
  *)
    echo "error: cabrnet 1.2 requires Python 3.10-3.12 (found $py_ver)" >&2
    exit 1
    ;;
esac

if [ -d "$CABRNET_DIR" ]; then
  echo "error: $CABRNET_DIR already exists, remove it first" >&2
  exit 1
fi

git clone https://github.com/aiser-team/cabrnet "$CABRNET_DIR"
git -C "$CABRNET_DIR" checkout "$CABRNET_COMMIT"
pip install -e "$CABRNET_DIR"
pip install -e .

echo "cabrnet pinned at $CABRNET_COMMIT ($CABRNET_DIR), beyond_l2_ext installed."
