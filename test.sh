#!/usr/bin/env bash
# Runs this plugin's tests.
#
# It needs cmake and a C++20 compiler. Without cmake there is nothing to test
# rather than something broken, so this says so and passes.
set -euo pipefail

cd "$(dirname "$0")"

if ! command -v cmake >/dev/null 2>&1; then
    echo "no tests for mstu-computer-vision: cmake is not installed"
    echo "    on debian and ubuntu: apt install cmake"
    exit 0
fi

# Unformatted code fails too, the same as a failing test would.
./fmt.sh --check

cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug >/dev/null
cmake --build build --parallel --target tests >/dev/null

ctest --test-dir build --output-on-failure "$@"
