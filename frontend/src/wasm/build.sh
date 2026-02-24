#!/usr/bin/env bash
# Compile matvec.c → matvec.wasm (WASM SIMD128, imported memory)
set -euo pipefail
DIR="$(cd "$(dirname "$0")" && pwd)"
clang --target=wasm32 -O3 -msimd128 -nostdlib \
  -Wl,--no-entry -Wl,--import-memory \
  -Wl,--stack-first -Wl,-z,stack-size=4096 \
  -o "$DIR/../../public/matvec.wasm" \
  "$DIR/matvec.c"
echo "Built matvec.wasm ($(wc -c < "$DIR/../../public/matvec.wasm") bytes)"
