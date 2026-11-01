// A node that looks at images and says what is in them.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include "host.hpp"
#include "preview.hpp"
#include "schema.hpp"
#include "worker.hpp"

namespace mstu::vision {
namespace {

const Metadata METADATA = {MSTU_STR("Computer vision"), MSTU_STR("0.1.0")};

/// How often readings go out while images are flowing.
constexpr auto LIVE_INTERVAL = std::chrono::milliseconds(250);

/// How many numbers one detection is written as.
constexpr size_t OBJECT_FIELDS = 7;

using Clock = std::chrono::steady_clock;

/// The classes in one lot of detections, each named once, in one string.
std::string classes_of(const std::vector<Detection> &found) {
    std::set<std::string> named;

    for (const Detection &one : found) {
        named.insert(one.name);
    }

    std::string all;

    for (const std::string &name : named) {
        if (!all.empty()) {
            all += ",";
        }

        all += name;
    }

    return all;
}

/// Splits a comma separated setting, trimming the spaces people leave.
std::vector<std::string> split(std::string_view text) {
    std::vector<std::string> parts;
    size_t at = 0;

    while (at <= text.size()) {
        const size_t comma = text.find(',', at);
        const size_t end = comma == std::string_view::npos ? text.size() : comma;

        std::string_view part = text.substr(at, end - at);

        while (!part.empty() && part.front() == ' ') {
            part.remove_prefix(1);
        }

        while (!part.empty() && part.back() == ' ') {
            part.remove_suffix(1);
        }

        if (!part.empty()) {
            parts.emplace_back(part);
        }

        if (comma == std::string_view::npos) {
            break;
        }

        at = comma + 1;
    }

    return parts;
}

class Vision {
  public:
    Vision(const HostContext *ctx, Str id)
        : host_(ctx, std::string(view(id))), preview_(std::string(view(id))) {
        worker_.preview(preview_.wanted());
    }

    const std::string &id() const { return host_.id(); }

    bool settings(Writer *out) const {
        const Value values[] = {
            mstu_str(model_.data(), model_.size()),
            mstu_float(wanted_.confidence),
            mstu_str(classes_.data(), classes_.size()),
            mstu_uint(wanted_.most),
            mstu_bool(only_when_found_),
            mstu_bool(preview_.wanted()),
        };

        return mstu_fill(out, values, std::size(values));
    }

    bool set(uintptr_t field, Value value) {
        switch (field) {
        case SettingModel: {
            if (value.kind != ValueString) {
                return false;
            }

            model_ = std::string(view(value.data.string_));

            // Reading a model is the thread's first job, so this returns
            // before tens of megabytes have been read.
            if (model_.empty()) {
                worker_.stop();
                log(Info, "no model named, images pass through untouched");
            } else {
                worker_.want(wanted_);
                worker_.start(model_);
                log(Info, "reading model " + model_);
            }

            return true;
        }

        case SettingConfidence:
            if (value.kind != ValueFloat) {
                return false;
            }

            wanted_.confidence = value.data.float_;
            worker_.want(wanted_);

            return true;

        case SettingClasses:
            if (value.kind != ValueString) {
                return false;
            }

            classes_ = std::string(view(value.data.string_));
            wanted_.classes = split(classes_);
            worker_.want(wanted_);

            return true;

        case SettingMaxObjects:
            if (value.kind != ValueUint) {
                return false;
            }

            wanted_.most = static_cast<uint32_t>(value.data.uint_);
            worker_.want(wanted_);

            return true;

        case SettingOnlyWhenFound:
            if (value.kind != ValueBool) {
                return false;
            }

            only_when_found_ = value.data.bool_;

            return true;

        case SettingPreview:
            if (value.kind != ValueBool) {
                return false;
            }

            preview_.set(value.data.bool_);
            worker_.preview(value.data.bool_);

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
            trouble_ = std::string(named) + " is not an image this reads";
            log(Warn, trouble_);

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
        trouble_.clear();

        return true;
    }

    bool process(ProcessContext *ctx) {
        if (announce_) {
            announce_ = false;
            changed();
        }

        Looked image;

        if (!taken(ctx->input, image)) {
            return false;
        }

        images_ += 1;

        // With a detector running, the looking happens on its thread and what
        // comes back is whatever it has finished by now. Without one -- no
        // model named, or a model that would not read -- the image goes
        // straight on with nothing found in it, so a pipeline drawn around
        // this node keeps working and the reading says why.
        if (worker_.state() == State::Looking) {
            worker_.offer(std::move(image));

            if (!worker_.take(image)) {
                report();

                return true;
            }
        }

        objects_ += image.found.size();

        // The drawn copy when there is one, the image itself when there is
        // not: a page showing the plain picture beats a page showing nothing.
        preview_.keep(image.drawn.empty() ? image.jpeg : image.drawn);

        showing_ = classes_of(image.found);

        report();

        if (image.found.empty() && only_when_found_) {
            return true;
        }

        found(image.found);

        return written(ctx->output, image);
    }

  private:
    void log(LogLevel level, std::string_view message) const { host_.log(level, message); }

    /// Reads one round's input into a job of its own.
    bool taken(const Message *input, Looked &job) const {
        const Value *time = mstu_value(input, FieldStartTime);
        const Value *width = mstu_value(input, FieldWidth);
        const Value *height = mstu_value(input, FieldHeight);
        const Value *image = mstu_value(input, FieldBytes);

        // A round with nothing in it is how a source says it is done.
        if (time == nullptr || image == nullptr || image->kind != ValueBytes) {
            return false;
        }

        job.start_time = time->data.uint_;
        job.width = width != nullptr ? static_cast<uint32_t>(width->data.uint_) : 0;
        job.height = height != nullptr ? static_cast<uint32_t>(height->data.uint_) : 0;
        job.found.clear();

        job.jpeg.assign(image->data.bytes_.ptr, image->data.bytes_.ptr + image->data.bytes_.len);

        return true;
    }

    /// Writes the image back out with what was found in it beside it.
    bool written(Writer *out, const Looked &image) {
        const size_t count = image.found.size();

        // Reserved up front: a record points into this, so it must not move
        // while the message is being filled.
        std::vector<Value> cells(count * OBJECT_FIELDS);
        std::vector<Value> records;

        records.reserve(count);

        for (size_t at = 0; at < count; at += 1) {
            const Detection &one = image.found[at];
            Value *cell = cells.data() + at * OBJECT_FIELDS;

            cell[0] = mstu_str(one.name.data(), one.name.size());
            cell[1] = mstu_uint(one.class_id);
            cell[2] = mstu_float(one.confidence);
            cell[3] = mstu_uint(one.x);
            cell[4] = mstu_uint(one.y);
            cell[5] = mstu_uint(one.width);
            cell[6] = mstu_uint(one.height);

            records.push_back(mstu_record(cell, OBJECT_FIELDS));
        }

        uint8_t *payload = mstu_reserve(out, image.jpeg.size());

        if (payload == nullptr) {
            return false;
        }

        if (!image.jpeg.empty()) {
            std::memcpy(payload, image.jpeg.data(), image.jpeg.size());
        }

        const Value values[] = {
            mstu_uint(image.start_time),
            mstu_uint(image.width),
            mstu_uint(image.height),
            mstu_bytes(payload, image.jpeg.size()),
            mstu_list(records.data(), records.size()),
        };

        return mstu_fill(out, values, std::size(values));
    }

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

    /// Raised when what is in view changes rather than on every image, since
    /// a camera watching a road would otherwise raise it thirty times a
    /// second and say nothing new each time.
    void found(const std::vector<Detection> &all) {
        if (showing_ == raised_) {
            return;
        }

        raised_ = showing_;

        const Value values[] = {
            mstu_uint(all.size()),
            mstu_str(raised_.data(), raised_.size()),
        };

        host_.raise(EventObjectsFound, values, std::size(values));
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

        const std::string state = named(worker_.state());
        const std::string backend = worker_.backend();
        const std::string note = trouble_.empty() ? worker_.note() : trouble_;
        const std::string shown = preview_.path();

        const Value values[] = {
            mstu_str(state.data(), state.size()),
            mstu_uint(images_),
            mstu_uint(objects_),
            mstu_uint(worker_.dropped()),
            mstu_uint(fps),
            mstu_str(backend.data(), backend.size()),
            mstu_str(note.data(), note.size()),
            mstu_str(showing_.data(), showing_.size()),
            mstu_str(shown.data(), shown.size()),
        };

        host_.live(values, std::size(values));

        beat_ = now;
        counted_ = images_;
    }

    static const char *named(State state) {
        switch (state) {
        case State::Loading:
            return "loading";
        case State::Looking:
            return "detecting";
        case State::Failed:
            return "failed";
        case State::Idle:
            break;
        }

        return "idle";
    }

    Host host_;
    Worker worker_;
    Preview preview_;

    std::string model_;
    std::string classes_;
    Wanted wanted_;
    bool only_when_found_ = false;

    /// What the images are, as the command said. Empty until it does.
    std::string codec_;

    /// Set by the command, acted on by the next round.
    bool announce_ = false;

    /// The classes in view now, and the ones objects.found last said, so it
    /// is raised on a change rather than on every image.
    std::string showing_;
    std::string raised_;

    uint64_t images_ = 0;
    uint64_t objects_ = 0;

    /// Something this plugin itself has to say, which outranks the thread's.
    std::string trouble_;

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

/// The folder beside the library, which the host serves to the page.
///
/// Named as collect.sh leaves it, not as it is called here: the plugins go
/// into one flat folder, so every ui folder carries its plugin's name.
Str ui() {
    return MSTU_STR("mstu-computer-vision-ui");
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
