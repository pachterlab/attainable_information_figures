#!/usr/bin/env bash
# Refresh the vendored snapshot of the attainable-information package from an upstream
# checkout (https://github.com/pachterlab/attainable_information) and record its commit.
#
#   scripts/sync_attainable_information.sh ~/path/to/attainable_information
set -euo pipefail
SRC="${1:?usage: $0 <path to attainable_information checkout>}"
DST="$(cd "$(dirname "$0")/.." && pwd)"

rsync -a --delete --exclude '__pycache__' "$SRC/attainable_information/" "$DST/attainable_information/"
mkdir -p "$DST/tests/attainable_information"
rsync -a --delete --exclude '__pycache__' "$SRC/tests/" "$DST/tests/attainable_information/"
cp "$SRC/LICENSE" "$DST/attainable_information/LICENSE"
{
  echo "Pinned snapshot of the attainable-information package."
  echo "Upstream: https://github.com/pachterlab/attainable_information"
  echo "Commit:   $(git -C "$SRC" rev-parse HEAD)"
  echo "Version:  $(grep -m1 '^version' "$SRC/pyproject.toml" | cut -d'"' -f2)"
  echo "Refresh with scripts/sync_attainable_information.sh <path-to-upstream-checkout>"
} > "$DST/attainable_information/UPSTREAM.txt"
cat "$DST/attainable_information/UPSTREAM.txt"
