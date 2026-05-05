#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
OUT_DIR="$ROOT_DIR/dist/arxiv"
BUNDLE_DIR="$OUT_DIR/coherence-delay-tradeoff-arxiv"
ARCHIVE_PATH="$OUT_DIR/coherence-delay-tradeoff-arxiv.tar.gz"

mkdir -p "$OUT_DIR"
rm -rf "$BUNDLE_DIR"
rm -f "$ARCHIVE_PATH"

printf 'Building manuscript with tectonic...\n'
cd "$ROOT_DIR"
tectonic --keep-intermediates main.tex

if [[ ! -f "$ROOT_DIR/main.bbl" ]]; then
  printf 'error: main.bbl was not produced; arXiv bundle requires a compiled bibliography.\n' >&2
  exit 1
fi

mkdir -p "$BUNDLE_DIR"

cp "$ROOT_DIR/main.tex" "$BUNDLE_DIR/"
cp "$ROOT_DIR/main.bbl" "$BUNDLE_DIR/"
cp "$ROOT_DIR/bibliography.bib" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/appendices" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/config" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/discussion" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/frontmatter" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/theory" "$BUNDLE_DIR/"
cp -R "$ROOT_DIR/figures" "$BUNDLE_DIR/"

tar -C "$OUT_DIR" -czf "$ARCHIVE_PATH" "$(basename "$BUNDLE_DIR")"

printf 'Created %s\n' "$ARCHIVE_PATH"
