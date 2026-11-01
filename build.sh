#!/usr/bin/env bash
# Builds this plugin. The library lands in dist/, where the centre repo's
# collect.sh looks for a plugin cargo knows nothing about.
#
# It needs cmake and a C++20 compiler. Without cmake it says so and stops
# without failing, so collect.sh carries on and a fresh clone of the centre
# repo stays green.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cmake >/dev/null 2>&1; then
    echo "    not built: cmake is not installed"
    exit 0
fi

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build build --parallel
