#!/usr/bin/env bash
# Formats this plugin, which is written in C++ and so is no business of
# rustfmt's. The centre repo's fmt.sh calls this instead.
#
#   ./fmt.sh            format in place
#   ./fmt.sh --check    say what is unformatted, change nothing
#
# Without clang-format there is nothing to format rather than something
# broken, so this says so and passes: the centre stays green on a machine
# that has not got it.
set -euo pipefail

cd "$(dirname "$0")"

tool=""

for name in clang-format clang-format-19 clang-format-18 clang-format-17; do
    if command -v "$name" >/dev/null 2>&1; then
        tool="$name"
        break
    fi
done

if [ -z "$tool" ]; then
    echo "mstu-computer-vision: clang-format is not installed, nothing formatted"
    exit 0
fi

files=$(find src tests -name '*.cpp' -o -name '*.hpp')

if [ "${1:-}" = "--check" ]; then
    # --dry-run says what it would change; -Werror turns that into a failure.
    $tool --dry-run -Werror $files
    exit $?
fi

$tool -i $files
