# mstu-computer-vision

An [mstu](https://github.com/ng-he/mstu) plugin: images in, and what is in
them out. Written in C++, which makes it the first part of mstu that is not
Rust.

```sh
./build.sh
./test.sh
./model.sh      # once, to fetch the model it detects with
```

The first two are what the centre repository calls. It needs cmake and a
compiler that takes C++20, and OpenCV to detect with; every script says so and
passes without what it wants, so a fresh clone of the centre repository stays
green. Built without OpenCV the node still loads, still describes itself and
still hands images on, and its own reading says why it finds nothing.

## What it does

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

## What it detects with

OpenCV's `dnn`, which carries the whole way from a jpeg to a list of boxes:
`imdecode` reads the image, `readNet` the model, `NMSBoxes` drops the weaker of
two boxes over the same thing -- one class at a time, because two different
things in the same place are both there.

What it expects is a **YOLOX** export: one output of `[1, boxes, 5 + classes]`,
with the boxes still in grid units, since the ONNX releases are exported
without the decode. That decode -- strides of 8, 16 and 32, `exp` on the width
and height, the objectness gating the class scan -- is the only part of the
backend particular to a family of models, and it is about forty lines. A class
number is named from the COCO eighty, or from a `.names` file beside the model
for anything trained on something else.

`./model.sh` fetches **YOLOX-tiny**, which is Apache-2.0. Ultralytics' YOLOv8
and YOLOv5 weights are AGPL-3.0, and a default that dragged that licence into
everything built here would be a decision taken on the user's behalf; pointing
the `model` setting at one is their decision instead. `build.sh` does not call
it: a build that needs the network is a build that fails on a train.

Detection takes tens of milliseconds and `process` runs on one of the engine's
own threads, so the looking happens on a thread of this plugin's. An image
offered while one is being looked at **replaces** what was waiting and the
dropped count rises -- a detection three seconds late is of use to nobody.
Throttling for real belongs upstream, in the image decoder's `rate` and
`min_change`.

The image is copied once on the way in, because the engine's own bytes are
gone by the time the looking is done and the image has to leave with the boxes
found in it. A jpeg is tens of kilobytes; this is the price of not blocking
the engine.

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

## How it is tested

`tests/tests.cpp` **draws a red octagon with STOP written on it** and asks the
model what it is. A model trained on COCO says `stop sign`, with the box where
the octagon was drawn -- and the box is what makes the test worth having, since
a grid decode out by one cell of the coarsest head still finds the sign and
puts it thirty-two pixels away. Drawn rather than photographed, so there is no
image committed beside it, no network, and the same pixels every run.

It skips, rather than fails, without OpenCV or without the model.

## Its own page

`ui/` is the page the desktop app mounts inside the node: the settings on the
left, and the latest image with its boxes drawn on beside them. The node asks
for 1224 pixels across when previews are on and folds back to 320 when they
are off, so a node showing nothing takes no room for it -- a box drawn on a
frame cannot be read in a column a few hundred pixels wide.

The box is asked for again when the picture's shape changes, because a
picture has no height until it has loaded: ask once at mount and the node is
sized for a page with no picture in it, and the one that arrives afterwards is
cut off. Not on every load, since the picture changes several times a second
and asking for a box lays the whole canvas out again.

The picture goes through a **file**. A live reading carries the length of a
bytes field and not the bytes, and the app already serves a local file to a
plugin's page, so the plugin keeps the latest jpeg in
`$XDG_RUNTIME_DIR/mstu-preview-<id>.jpg` and the reading names it. Written
beside and renamed over, so a page never catches half an image, and at most
four times a second, which is about as often as a page redraws.

The boxes are drawn by the backend rather than beside it, because drawing is
pixel work and pixels are what a backend owns -- one that keeps them on a GPU
draws there and hands back only the jpeg. They are drawn in the colour the
canvas draws a transform node in, which is what this node is.

Turning the preview off stops the decode and the encode as well as the file:
nothing downstream wants the drawn copy, so it is only made when a page is
there to look at it.

## Still to come

`dnn`'s CUDA backend, or ONNX Runtime, behind the same `Detector` seam.
