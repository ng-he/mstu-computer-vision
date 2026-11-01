// What this plugin takes, gives, answers and reports, as the engine reads it.
#pragma once

#include "mstu-sdk.h"

namespace mstu::vision {

extern const Schema INPUT_SCHEMA;
extern const Schema OUTPUT_SCHEMA;
extern const Schema SETTINGS_SCHEMA;
extern const Schema LIVE_SCHEMA;

extern const Slice_EventDescriptor EVENTS;
extern const Slice_CommandDescriptor COMMANDS;

/// Where each setting is, since set_parameter addresses one by index.
enum Setting : uintptr_t {
    SettingModel,
    SettingConfidence,
    SettingClasses,
    SettingMaxObjects,
    SettingOnlyWhenFound,
};

/// Where each event is, in the order EVENTS names them.
///
/// image.changed is first, as it is on a source and on the image decoder: a
/// subscription that says what the data is now addresses event 0 wherever it
/// is wired from.
enum Event : uintptr_t {
    EventImageChanged,
    EventObjectsFound,
};

/// Where each command is, in the order COMMANDS names them.
enum Command : uintptr_t {
    CommandCodecChange,
};

/// The fields of a message in, and of one out: the first four are the image
/// decoder's own, in its order, so this plugs in where images already go.
///
/// Not named `Field`, which is the SDK's own type for a schema's field: the
/// header puts its names in the global namespace, so a plugin's shadow it.
enum MessageField : uintptr_t {
    FieldStartTime,
    FieldWidth,
    FieldHeight,
    FieldBytes,
    FieldObjects,
};

} // namespace mstu::vision
