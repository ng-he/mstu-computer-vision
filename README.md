# mstu-computer-vision

An [mstu](https://github.com/ng-he/mstu) plugin: images in, and what is in
them out. Written in C++, which makes it the first part of mstu that is not
Rust.

```sh
./build.sh
./test.sh
```

Which is what the centre repository calls to build and test this one. It needs
cmake and a compiler that takes C++20; both scripts say so and pass when cmake
is missing, so a fresh clone of the centre repository stays green.

## What it does

Nothing yet. This is the step before any detecting: an image goes through
untouched and the list of objects goes out empty, so the whole of the engine's
side is in place and running before a model is loaded into it.

It takes images the way the
[image decoder](https://github.com/ng-he/mstu-image-decoder) gives them — the
same four fields in the same order — so it drops into a chain that already
carries images, and gives those four back with the detections behind them:

```
in     start_time · width · height · bytes (a jpeg)
out    start_time · width · height · bytes (the same jpeg) · objects
```

One object is a record: the class and the model's own number for it, how sure,
and where, in the pixels of the image the same message carries. Pixels rather
than a fraction of the width, because a box is only ever read against that one
image, and because a whole number reads better than a float in a log or on a
page.

**The image goes out whether anything was found or not.** A detection without
its image is nearly useless downstream — an overlay has nothing to draw on, a
dumper nothing to write, an alert nothing to crop — and the engine shares one
payload between every consumer of a node rather than copying it, so carrying
the image on costs a pointer. `only_when_found` turns it off for the alerting
case, at the price of the node no longer being transparent.

The list of records is **one leaf** when nodes are wired field to field, so
whatever takes the detections has to declare the same record, field for field.
That shape is a contract on both ends of the wire.

## Why C++

Because the C header is the ABI's other half, and until this plugin nothing had
ever built a whole plugin from it. Writing this one found two holes in it on the
first afternoon: the header had no `extern "C"`, so a C++ plugin exported a
mangled `plugin_descriptor` the engine could not find, and it carried no type
constants, so the schema half of the ABI could not be written from C at all
without rebuilding every `struct Type` by hand.

Both are fixed in mstu-sdk, and `mstu-sdk/tests/plugin.c` keeps them fixed.

Python was considered and dropped: it cannot export `plugin_descriptor` at all,
so it always needs a shim, and then either the GIL serialises every Python node
in the engine or a sidecar process has to be managed — for no gain over C++,
which the detection libraries speak anyway.

## How it is put together

| | |
| --- | --- |
| `src/schema.cpp` | The tables the engine reads: what it takes, gives, answers and reports. |
| `src/plugin.cpp` | The descriptor, and the one class behind it. |
| `src/host.hpp` | The log, the events and the live readings, as one thing to hold. |

Nothing links against the SDK: the generated header is the whole of it, and a
message is read with `mstu_value` rather than through any Rust of ours.
`CMakeLists.txt` finds the header beside this repository inside the centre
checkout, and takes `-DMSTU_SDK_ABI=<folder>` for a clone on its own.

Two things worth knowing before changing it.

The header puts its names in the global namespace — `Value`, `Type`, `Field`,
`Message` — so a name of ours shadows one of its, and the compiler's complaint
is about braces rather than about the name. `MessageField` is called that for
exactly this reason.

**An event must not be raised from inside a command.** The engine holds its
event lock while a subscriber runs, so raising one from a handler it delivered
deadlocks. `codec.change` sets a flag and the next round does the raising.

## It will need a model

Detection comes next, through OpenCV's `dnn`: `imdecode` takes the jpeg,
`readNet` the model, `NMSBoxes` the overlapping boxes, and `imencode` puts a
preview back together with the boxes drawn on it. One dependency for the whole
pipeline, and a seam of its own so ONNX Runtime or TensorRT can take its place.

The default model will be **YOLOX**, which is Apache-2.0. Ultralytics' YOLOv8
and YOLOv5 weights are AGPL-3.0, and a build script that downloads one by
default would make that licence part of what everyone ships; pointing the
`model` setting at one is a choice, and the choice should be the user's.
