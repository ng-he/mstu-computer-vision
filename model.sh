#!/usr/bin/env bash
# Fetches the model this detects with, into models/.
#
# YOLOX-tiny, which is Apache-2.0. Ultralytics' YOLOv8 and YOLOv5 weights are
# AGPL-3.0, and a default that dragged that licence into everything built here
# would be a decision taken on the user's behalf; point the `model` setting at
# one and it is their decision instead.
#
# Not called by build.sh: a build that needs the network is a build that fails
# on a train. Run this once.
set -euo pipefail

cd "$(dirname "$0")"

RELEASE=https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0
NAME=yolox_tiny.onnx

mkdir -p models

if [ -f "models/$NAME" ]; then
    echo "already here: $(pwd)/models/$NAME"
    exit 0
fi

echo "fetching $NAME"
curl -fL --progress-bar "$RELEASE/$NAME" -o "models/$NAME.part"
mv "models/$NAME.part" "models/$NAME"

echo "$(pwd)/models/$NAME"
