// A node that looks at images and says what is in them.
//
// This is the step before any detecting: the image goes through untouched and
// the list of objects goes out empty, so the whole of the engine's side -- the
// schemas, the settings, the command, the events, the readings -- is in place
// and running before a model is loaded into it.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <string>

#include "host.hpp"
#include "schema.hpp"

namespace mstu::vision {
namespace {

const Metadata METADATA = {MSTU_STR("Computer vision"), MSTU_STR("0.1.0")};

/// How often readings go out while images are flowing.
constexpr auto LIVE_INTERVAL = std::chrono::milliseconds(250);

using Clock = std::chrono::steady_clock;

class Vision {
  public:
    Vision(const HostContext *ctx, Str id) : host_(ctx, std::string(view(id))) {}

    const std::string &id() const { return host_.id(); }

    bool settings(Writer *out) const {
        const Value values[] = {
            mstu_str(model_.data(), model_.size()),
            mstu_float(confidence_),
            mstu_str(classes_.data(), classes_.size()),
            mstu_uint(max_objects_),
            mstu_bool(only_when_found_),
        };

        return mstu_fill(out, values, std::size(values));
    }

    bool set(uintptr_t field, Value value) {
        switch (field) {
        case SettingModel:
            if (value.kind != ValueString) {
                return false;
            }

            model_ = std::string(view(value.data.string_));
            note_.clear();

            return true;

        case SettingConfidence:
            if (value.kind != ValueFloat) {
                return false;
            }

            confidence_ = value.data.float_;

            return true;

        case SettingClasses:
            if (value.kind != ValueString) {
                return false;
            }

            classes_ = std::string(view(value.data.string_));

            return true;

        case SettingMaxObjects:
            if (value.kind != ValueUint) {
                return false;
            }

            max_objects_ = static_cast<uint32_t>(value.data.uint_);

            return true;

        case SettingOnlyWhenFound:
            if (value.kind != ValueBool) {
                return false;
            }

            only_when_found_ = value.data.bool_;

            return true;

        default:
            return false;
        }
    }

    /// Says what the images that follow are. Nothing is published here: the
    /// engine holds its event lock while a subscriber runs, so raising an
    /// event from inside a command it delivered deadlocks. The next round
    /// does it.
    bool invoke(uintptr_t command, const Message *input) {
        if (command != CommandCodecChange || input == nullptr) {
            return false;
        }

        const Value *codec = mstu_value(input, 0);

        if (codec == nullptr || codec->kind != ValueString) {
            return false;
        }

        const std::string_view named = view(codec->data.string_);

        if (named != "jpeg") {
            note_ = std::string(named) + " is not an image this reads";
            log(Warn, note_);

            return false;
        }

        // Said again for every image, the way the decoder says it, so the
        // announcement goes out again and whatever takes the images starts
        // afresh on each. Only a real change is worth a log line.
        if (codec_ != named) {
            log(Info, "taking jpeg images");
        }

        codec_ = named;
        announce_ = true;
        note_.clear();

        return true;
    }

    bool process(ProcessContext *ctx) {
        if (announce_) {
            announce_ = false;
            changed();
        }

        const Value *time = mstu_value(ctx->input, FieldStartTime);
        const Value *width = mstu_value(ctx->input, FieldWidth);
        const Value *height = mstu_value(ctx->input, FieldHeight);
        const Value *image = mstu_value(ctx->input, FieldBytes);

        // A round with nothing in it is how a source says it is done.
        if (time == nullptr || image == nullptr) {
            return false;
        }

        if (image->kind != ValueBytes) {
            return false;
        }

        images_ += 1;

        // Nothing detects yet, so every image holds nothing as far as this
        // knows: asked for only what was found, it hands on nothing at all.
        // That is the setting working, not a fault.
        if (only_when_found_) {
            report();

            return true;
        }

        const uintptr_t len = image->data.bytes_.len;

        uint8_t *payload = mstu_reserve(ctx->output, len);

        if (payload == nullptr) {
            return false;
        }

        if (len > 0) {
            std::memcpy(payload, image->data.bytes_.ptr, len);
        }

        const Value values[] = {
            mstu_uint(time->data.uint_),
            mstu_uint(width != nullptr ? width->data.uint_ : 0),
            mstu_uint(height != nullptr ? height->data.uint_ : 0),
            mstu_bytes(payload, len),
            mstu_list(nullptr, 0),
        };

        report();

        return mstu_fill(ctx->output, values, std::size(values));
    }

  private:
    void log(LogLevel level, std::string_view message) const { host_.log(level, message); }

    /// Tells whoever takes the images what they are, which is what they were:
    /// this hands a jpeg on as a jpeg.
    void changed() {
        const Value values[] = {
            mstu_str(codec_.data(), codec_.size()),
            mstu_str("jpeg", 4),
            mstu_bytes(nullptr, 0),
        };

        if (!host_.raise(EventImageChanged, values, std::size(values))) {
            log(Warn, "the host took no image.changed to fill");
        }
    }

    /// The readings the plugin's own page shows, at most four a second.
    void report() {
        const auto now = Clock::now();
        const bool first = beat_ == Clock::time_point{};

        if (!first && now - beat_ < LIVE_INTERVAL) {
            return;
        }

        const auto elapsed =
            std::chrono::duration_cast<std::chrono::milliseconds>(now - beat_).count();

        uint64_t fps = 0;

        if (!first && elapsed > 0) {
            fps = (images_ - counted_) * 1000 / static_cast<uint64_t>(elapsed);
        }

        const std::string state = codec_.empty() ? "idle" : "detecting";

        const Value values[] = {
            mstu_str(state.data(), state.size()),
            mstu_uint(images_),
            mstu_uint(objects_),
            mstu_uint(dropped_),
            mstu_uint(fps),
            mstu_str("none", 4),
            mstu_str(note_.data(), note_.size()),
        };

        host_.live(values, std::size(values));

        beat_ = now;
        counted_ = images_;
    }

    Host host_;

    std::string model_;
    double confidence_ = 0.5;
    std::string classes_;
    uint32_t max_objects_ = 0;
    bool only_when_found_ = false;

    /// What the images are, as the command said. Empty until it does.
    std::string codec_;

    /// Set by the command, acted on by the next round.
    bool announce_ = false;

    uint64_t images_ = 0;
    uint64_t objects_ = 0;
    uint64_t dropped_ = 0;
    std::string note_;

    Clock::time_point beat_{};
    uint64_t counted_ = 0;
};

const Metadata *metadata() {
    return &METADATA;
}

PluginHandle create(const HostContext *ctx, Str id) {
    return new Vision(ctx, id);
}

Str plugin_id(PluginHandle handle) {
    return str(static_cast<Vision *>(handle)->id());
}

void release(PluginHandle handle) {
    delete static_cast<Vision *>(handle);
}

const Schema *input_schema() {
    return &INPUT_SCHEMA;
}

const Schema *output_schema() {
    return &OUTPUT_SCHEMA;
}

const Schema *settings_schema() {
    return &SETTINGS_SCHEMA;
}

const Schema *live_schema() {
    return &LIVE_SCHEMA;
}

bool settings(PluginHandle handle, Writer *out) {
    return static_cast<Vision *>(handle)->settings(out);
}

bool set_parameter(PluginHandle handle, uintptr_t field, Value value) {
    return static_cast<Vision *>(handle)->set(field, value);
}

/// No page of its own yet, so the engine draws the node and nothing more.
Str ui() {
    return Str{nullptr, 0};
}

bool process(PluginHandle handle, ProcessContext *ctx) {
    return static_cast<Vision *>(handle)->process(ctx);
}

Slice_EventDescriptor events() {
    return EVENTS;
}

Slice_CommandDescriptor commands() {
    return COMMANDS;
}

bool invoke(PluginHandle handle, uintptr_t command, const Message *input) {
    return static_cast<Vision *>(handle)->invoke(command, input);
}

const PluginDescriptor DESCRIPTOR = {
    metadata,      create,          plugin_id, release,       input_schema,
    output_schema, settings_schema, settings,  set_parameter, ui,
    process,       events,          commands,  invoke,        live_schema,
};

} // namespace
} // namespace mstu::vision

// Declared by the header inside its extern "C", which is what keeps this
// symbol findable from C++.
const PluginDescriptor *plugin_descriptor(void) {
    return &mstu::vision::DESCRIPTOR;
}
