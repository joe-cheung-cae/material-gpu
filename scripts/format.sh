#!/usr/bin/env bash
set -euo pipefail

# Find clang-format
if command -v clang-format >/dev/null 2>&1; then
  CF=clang-format
elif command -v clang-format-18 >/dev/null 2>&1; then
  CF=clang-format-18
else
  echo "[format] clang-format not found. Please install clang-format." >&2
  exit 1
fi

# Default to repository root
ROOT_DIR="$(cd "$(dirname "$0")"/.. && pwd)"
cd "$ROOT_DIR"

# Files to format (try git first, else fallback to find)
if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  mapfile -t FILES < <(git ls-files '**/*.[ch]pp' '**/*.[ch]xx' '**/*.[ch]' '**/*.cu' '**/*.cuh' | grep -v '^build/')
else
  mapfile -t FILES < <(find . -path ./build -prune -o \
    -type f \( \
      -name '*.c' -o -name '*.cc' -o -name '*.cpp' -o -name '*.cxx' -o \
      -name '*.h' -o -name '*.hpp' -o -name '*.hxx' -o \
      -name '*.cu' -o -name '*.cuh' \
    \) -print | sed 's#^\./##')
fi

if [ ${#FILES[@]} -eq 0 ]; then
  echo "[format] No files to format"
  exit 0
fi

"$CF" -i "${FILES[@]}"

echo "[format] Formatted ${#FILES[@]} files."
