#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="$ROOT_DIR/data/kuairand"
ARCHIVE_PATH="$DATA_DIR/KuaiRand-Pure.tar.gz"
URL="https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"

mkdir -p "$DATA_DIR"

if [[ ! -f "$ARCHIVE_PATH" ]]; then
  printf 'Downloading KuaiRand-Pure dataset...\n'
  wget -O "$ARCHIVE_PATH" "$URL"
fi

printf 'Extracting KuaiRand-Pure dataset...\n'
tar -xzvf "$ARCHIVE_PATH" -C "$DATA_DIR"
printf 'Done. Data is under %s/KuaiRand-Pure/data\n' "$DATA_DIR"
