#include "schema.hpp"

namespace mstu::vision {

static const Field INPUT_FIELDS[] = {
    {MSTU_STR("start_time"), &MSTU_TYPE_U64, MSTU_STR("Time of the image in microseconds.")},
    {MSTU_STR("width"), &MSTU_TYPE_U32, MSTU_STR("Width of the image in pixels.")},
    {MSTU_STR("height"), &MSTU_TYPE_U32, MSTU_STR("Height of the image in pixels.")},
    {MSTU_STR("bytes"), &MSTU_TYPE_BYTES, MSTU_STR("The image, as a jpeg file.")},
};

const Schema INPUT_SCHEMA = {MSTU_SLICE(INPUT_FIELDS)};

// One object found: what it is, how sure, and where in this image's own
// pixels. Read against the width and height the same message carries.
static const Field OBJECT_FIELDS[] = {
    {MSTU_STR("class"), &MSTU_TYPE_STRING, MSTU_STR("What was found, as the model names it.")},
    {MSTU_STR("class_id"), &MSTU_TYPE_U32, MSTU_STR("The model's own number for that class.")},
    {MSTU_STR("confidence"), &MSTU_TYPE_F64, MSTU_STR("How sure, 0 to 1.")},
    {MSTU_STR("x"), &MSTU_TYPE_U32, MSTU_STR("Left edge in pixels.")},
    {MSTU_STR("y"), &MSTU_TYPE_U32, MSTU_STR("Top edge in pixels.")},
    {MSTU_STR("width"), &MSTU_TYPE_U32, MSTU_STR("Width of the box in pixels.")},
    {MSTU_STR("height"), &MSTU_TYPE_U32, MSTU_STR("Height of the box in pixels.")},
};

static const Schema OBJECT_SCHEMA = {MSTU_SLICE(OBJECT_FIELDS)};

static const Type OBJECT_TYPE = MSTU_TYPE_RECORD(&OBJECT_SCHEMA);
static const Type OBJECTS_TYPE = MSTU_TYPE_LIST(&OBJECT_TYPE);

static const Field OUTPUT_FIELDS[] = {
    {MSTU_STR("start_time"), &MSTU_TYPE_U64, MSTU_STR("Time of the image this came from.")},
    {MSTU_STR("width"), &MSTU_TYPE_U32, MSTU_STR("Width of the image in pixels.")},
    {MSTU_STR("height"), &MSTU_TYPE_U32, MSTU_STR("Height of the image in pixels.")},
    {MSTU_STR("bytes"), &MSTU_TYPE_BYTES, MSTU_STR("The image, handed on unchanged.")},
    {MSTU_STR("objects"), &OBJECTS_TYPE,
     MSTU_STR("What was found in it, empty when nothing was. The image goes out either way, so "
              "this node can sit anywhere images already flow.")},
};

const Schema OUTPUT_SCHEMA = {MSTU_SLICE(OUTPUT_FIELDS)};

static const Field SETTINGS_FIELDS[] = {
    {MSTU_STR("model"), &MSTU_TYPE_STRING,
     MSTU_STR("Path of the model to detect with. Nothing is detected until one is named.")},
    {MSTU_STR("confidence"), &MSTU_TYPE_F64,
     MSTU_STR("How sure the model must be before an object is reported, 0 to 1.")},
    {MSTU_STR("classes"), &MSTU_TYPE_STRING,
     MSTU_STR("Only report these classes, named as the model names them and separated by "
              "commas. Empty reports every class the model knows.")},
    {MSTU_STR("max_objects"), &MSTU_TYPE_U32,
     MSTU_STR("How many objects one image may report, highest confidence first. 0 is no limit.")},
    {MSTU_STR("only_when_found"), &MSTU_TYPE_BOOL,
     MSTU_STR("Hands on nothing when an image held no object, instead of the image with an "
              "empty list. Saves the work downstream and costs this node its transparency.")},
    {MSTU_STR("preview"), &MSTU_TYPE_BOOL,
     MSTU_STR("Keeps the latest image, with the boxes drawn on it, in a file for this plugin's "
              "own page. A decode and an encode per picture, a few a second; turn it off and "
              "the page shows nothing.")},
};

const Schema SETTINGS_SCHEMA = {MSTU_SLICE(SETTINGS_FIELDS)};

static const Field LIVE_FIELDS[] = {
    {MSTU_STR("state"), &MSTU_TYPE_STRING,
     MSTU_STR("What it is doing: idle, detecting or failed.")},
    {MSTU_STR("images"), &MSTU_TYPE_U64, MSTU_STR("Images taken in.")},
    {MSTU_STR("objects"), &MSTU_TYPE_U64, MSTU_STR("Objects found in all of them.")},
    {MSTU_STR("dropped"), &MSTU_TYPE_U64,
     MSTU_STR("Images let go without looking, because one was still being looked at. Detection "
              "is slower than a camera, so this rises rather than the queue growing.")},
    {MSTU_STR("fps"), &MSTU_TYPE_U64, MSTU_STR("Images a second, as they are going through now.")},
    {MSTU_STR("backend"), &MSTU_TYPE_STRING, MSTU_STR("What is doing the looking.")},
    {MSTU_STR("note"), &MSTU_TYPE_STRING,
     MSTU_STR("Why it is in the state it is, when it is worth saying.")},
    {MSTU_STR("objects_now"), &MSTU_TYPE_STRING,
     MSTU_STR("The classes in view at the moment, each named once and separated by commas.")},
    {MSTU_STR("preview"), &MSTU_TYPE_STRING,
     MSTU_STR("Path of the file holding the latest image with its boxes, for a page to show. "
              "Empty when previews are off or nothing has been looked at.")},
};

const Schema LIVE_SCHEMA = {MSTU_SLICE(LIVE_FIELDS)};

// The one codec this reads, named as the rest of mstu names it. An enum
// rather than a string so the app offers it rather than asking for it typed,
// and so what the image decoder announces and what this is asked are the same
// kind of thing.
static const EnumVariant CODEC_VARIANTS[] = {
    {MSTU_STR("JPEG"), {ValueString, {.string_ = MSTU_STR("jpeg")}}},
};

static const EnumSchema CODEC_ENUM = {MSTU_SLICE(CODEC_VARIANTS)};

static const Type CODEC_TYPE = MSTU_TYPE_ENUM(&CODEC_ENUM);

static const Field OBJECTS_FOUND_FIELDS[] = {
    {MSTU_STR("count"), &MSTU_TYPE_U32, MSTU_STR("How many objects are in the image.")},
    {MSTU_STR("classes"), &MSTU_TYPE_STRING,
     MSTU_STR("Which classes are in it, separated by commas and each named once.")},
};

static const Schema OBJECTS_FOUND_SCHEMA = {MSTU_SLICE(OBJECTS_FOUND_FIELDS)};

// The same three fields, in the same order, as a source's file.changed and
// the image decoder's image.changed: one wiring downstream serves any of them.
static const Field IMAGE_CHANGED_FIELDS[] = {
    {MSTU_STR("from"), &CODEC_TYPE, MSTU_STR("What the images were before this node.")},
    {MSTU_STR("codec"), &CODEC_TYPE, MSTU_STR("What they are leaving it, which is always jpeg.")},
    {MSTU_STR("extras"), &MSTU_TYPE_BYTES,
     MSTU_STR("Empty: a jpeg carries everything a reader of it needs.")},
};

static const Schema IMAGE_CHANGED_SCHEMA = {MSTU_SLICE(IMAGE_CHANGED_FIELDS)};

static const EventDescriptor EVENT_LIST[] = {
    {MSTU_STR("image.changed"), &IMAGE_CHANGED_SCHEMA,
     MSTU_STR("Raised when images start flowing, so whatever takes them knows what they are.")},
    {MSTU_STR("objects.found"), &OBJECTS_FOUND_SCHEMA,
     MSTU_STR("Raised when what is in view changes: an object of a class that was not there "
              "before, or the last of a class leaving.")},
};

const Slice_EventDescriptor EVENTS = MSTU_SLICE(EVENT_LIST);

// The image decoder's ingest shape, so its image.changed wires straight in.
static const Field CODEC_CHANGE_FIELDS[] = {
    {MSTU_STR("codec"), &CODEC_TYPE,
     MSTU_STR("What the images that follow are. Only jpeg is read.")},
    {MSTU_STR("extras"), &MSTU_TYPE_BYTES, MSTU_STR("Ignored: a jpeg needs none.")},
};

static const Schema CODEC_CHANGE_SCHEMA = {MSTU_SLICE(CODEC_CHANGE_FIELDS)};

static const CommandDescriptor COMMAND_LIST[] = {
    {MSTU_STR("codec.change"), &CODEC_CHANGE_SCHEMA,
     MSTU_STR("Says what the images that follow are, and starts looking at them.")},
};

const Slice_CommandDescriptor COMMANDS = MSTU_SLICE(COMMAND_LIST);

} // namespace mstu::vision
