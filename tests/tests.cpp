// This plugin's own tests, run by test.sh through ctest.
//
// They stand in for the engine: a writer of their own, so what the plugin
// wrote can be read back, and no host at all, so nothing here needs one
// running.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "host.hpp"
#include "schema.hpp"

namespace {

int failures = 0;

void check(bool held, const char *what, const char *file, int line) {
    if (held) {
        return;
    }

    std::printf("  FAILED %s\n    at %s:%d\n", what, file, line);
    failures += 1;
}

#define CHECK(held) check((held), #held, __FILE__, __LINE__)

/// What the plugin wrote, as the engine would hold it.
///
/// The spine of a list or a record is copied, because a plugin builds those
/// on its own stack and they are gone when the call returns. Leaves are left
/// alone: they point at static data or at memory reserve handed out.
struct Engine {
    std::vector<Value> values;
    std::vector<uint8_t> reserved;
    std::vector<std::vector<Value>> kept;

    Writer writer() {
        Writer w{};

        w._engine_data = this;
        w.fill = &Engine::took;
        w.reserve = &Engine::lend;

        return w;
    }

    static bool took(Writer *w, Message message) {
        auto *self = static_cast<Engine *>(w->_engine_data);

        self->values = self->keep(message.values);

        return true;
    }

    static uint8_t *lend(Writer *w, uintptr_t len) {
        auto *self = static_cast<Engine *>(w->_engine_data);

        self->reserved.assign(len, 0);

        return self->reserved.data();
    }

    std::vector<Value> keep(Slice_Value slice) {
        std::vector<Value> copied;

        for (uintptr_t at = 0; at < slice.len; at += 1) {
            Value value = slice.ptr[at];

            if (value.kind == ValueList || value.kind == ValueRecord) {
                kept.push_back(keep(value.data.list_));

                const std::vector<Value> &inner = kept.back();

                value.data.list_ = Slice_Value{inner.data(), inner.size()};
            }

            copied.push_back(value);
        }

        return copied;
    }
};

const PluginDescriptor *plugin() {
    return plugin_descriptor();
}

std::string text(Str from) {
    return std::string(mstu::vision::view(from));
}

void the_schemas_are_the_contract() {
    const Schema *out = plugin()->process_output_schema();

    CHECK(out->fields.len == 5);

    // The first four are the image decoder's own output, in its order, so
    // this node drops into a chain that already carries images.
    const Schema *in = plugin()->process_input_schema();

    CHECK(in->fields.len == 4);

    for (uintptr_t at = 0; at < in->fields.len; at += 1) {
        CHECK(text(in->fields.ptr[at].name) == text(out->fields.ptr[at].name));
        CHECK(in->fields.ptr[at].ty->kind == out->fields.ptr[at].ty->kind);
    }

    // And the fifth is the detections: a list of records, which is one leaf
    // when wired, so its shape is a contract on both ends.
    const Field &objects = out->fields.ptr[4];

    CHECK(text(objects.name) == "objects");
    CHECK(objects.ty->kind == TypeList);

    const Type *element = objects.ty->schema.list_.element;

    CHECK(element->kind == TypeRecord);
    CHECK(element->schema.record_.schema->fields.len == 7);

    const Field *box = element->schema.record_.schema->fields.ptr;

    CHECK(text(box[0].name) == "class");
    CHECK(box[0].ty->kind == TypeString);
    CHECK(text(box[3].name) == "x");
    CHECK(box[3].ty->kind == TypeInt);
    CHECK(box[3].ty->schema.int_.bits == 32);
}

void settings_are_read_back_as_they_were_written() {
    PluginHandle handle = plugin()->create(nullptr, mstu::vision::str("vision-1"));

    CHECK(text(plugin()->id(handle)) == "vision-1");

    CHECK(plugin()->set_parameter(handle, mstu::vision::SettingModel, mstu_str("yolox.onnx", 10)));
    CHECK(plugin()->set_parameter(handle, mstu::vision::SettingConfidence, mstu_float(0.4)));
    CHECK(plugin()->set_parameter(handle, mstu::vision::SettingMaxObjects, mstu_uint(8)));

    // A setting of the wrong kind is refused rather than taken as zero.
    CHECK(!plugin()->set_parameter(handle, mstu::vision::SettingConfidence, mstu_uint(1)));

    // And one it does not have at all.
    CHECK(!plugin()->set_parameter(handle, 99, mstu_bool(true)));

    Engine engine;
    Writer writer = engine.writer();

    CHECK(plugin()->settings(handle, &writer));
    CHECK(engine.values.size() == 5);
    CHECK(text(engine.values[0].data.string_) == "yolox.onnx");
    CHECK(engine.values[1].data.float_ == 0.4);
    CHECK(engine.values[3].data.uint_ == 8);

    plugin()->release(handle);
}

/// Nothing detects yet, so what this proves is that the image goes through
/// and the list goes out empty rather than missing.
void an_image_goes_through_with_an_empty_list() {
    PluginHandle handle = plugin()->create(nullptr, mstu::vision::str("vision-2"));

    const uint8_t image[] = {0xff, 0xd8, 0xff, 0xe0, 0x11};

    const Value input[] = {
        mstu_uint(1234),
        mstu_uint(640),
        mstu_uint(480),
        mstu_bytes(image, sizeof(image)),
    };

    Message message = {{input, std::size(input)}};

    Engine engine;
    Writer writer = engine.writer();
    ProcessContext ctx = {&message, &writer};

    CHECK(plugin()->process(handle, &ctx));

    CHECK(engine.values.size() == 5);
    CHECK(engine.values[0].data.uint_ == 1234);
    CHECK(engine.values[1].data.uint_ == 640);
    CHECK(engine.values[2].data.uint_ == 480);

    CHECK(engine.values[3].kind == ValueBytes);
    CHECK(engine.values[3].data.bytes_.len == sizeof(image));
    CHECK(std::memcmp(engine.values[3].data.bytes_.ptr, image, sizeof(image)) == 0);

    // The payload is the memory the writer reserved, not one of the plugin's
    // own: the engine owns every byte a message carries.
    CHECK(engine.values[3].data.bytes_.ptr == engine.reserved.data());

    CHECK(engine.values[4].kind == ValueList);
    CHECK(engine.values[4].data.list_.len == 0);

    plugin()->release(handle);
}

/// Asked for only what was found, and nothing is found yet, it hands on
/// nothing rather than an image with an empty list.
void only_when_found_holds_the_image_back() {
    PluginHandle handle = plugin()->create(nullptr, mstu::vision::str("vision-5"));

    CHECK(plugin()->set_parameter(handle, mstu::vision::SettingOnlyWhenFound, mstu_bool(true)));

    const uint8_t image[] = {0xff, 0xd8, 0xff};

    const Value input[] = {
        mstu_uint(1),
        mstu_uint(16),
        mstu_uint(16),
        mstu_bytes(image, sizeof(image)),
    };

    Message message = {{input, std::size(input)}};

    Engine engine;
    Writer writer = engine.writer();
    ProcessContext ctx = {&message, &writer};

    // True, because nothing went wrong: a round that produces nothing is how
    // a node says it has nothing to say this time.
    CHECK(plugin()->process(handle, &ctx));
    CHECK(engine.values.empty());

    plugin()->release(handle);
}

void a_round_with_nothing_in_it_is_refused() {
    PluginHandle handle = plugin()->create(nullptr, mstu::vision::str("vision-3"));

    Message empty = {{nullptr, 0}};

    Engine engine;
    Writer writer = engine.writer();
    ProcessContext ctx = {&empty, &writer};

    CHECK(!plugin()->process(handle, &ctx));
    CHECK(engine.values.empty());

    plugin()->release(handle);
}

void only_jpeg_is_taken() {
    PluginHandle handle = plugin()->create(nullptr, mstu::vision::str("vision-4"));

    const Value jpeg[] = {mstu_str("jpeg", 4), mstu_bytes(nullptr, 0)};
    Message saying = {{jpeg, std::size(jpeg)}};

    CHECK(plugin()->invoke(handle, mstu::vision::CommandCodecChange, &saying));

    // A codec this cannot read is refused, not taken and then failed on every
    // frame after.
    const Value h264[] = {mstu_str("h264", 4), mstu_bytes(nullptr, 0)};
    Message wrong = {{h264, std::size(h264)}};

    CHECK(!plugin()->invoke(handle, mstu::vision::CommandCodecChange, &wrong));

    // And a command it does not have.
    CHECK(!plugin()->invoke(handle, 7, &saying));

    plugin()->release(handle);
}

void the_descriptor_is_filled_in() {
    CHECK(text(plugin()->metadata()->name) == "Computer vision");
    CHECK(plugin()->events().len == 2);
    // image.changed is first, as it is on a source and on the image decoder,
    // so one subscription reaches any of them.
    CHECK(text(plugin()->events().ptr[0].event_name) == "image.changed");
    CHECK(text(plugin()->events().ptr[1].event_name) == "objects.found");
    CHECK(plugin()->commands().len == 1);
    CHECK(text(plugin()->commands().ptr[0].command_name) == "codec.change");
    CHECK(plugin()->live_schema()->fields.len == 7);
    CHECK(plugin()->settings_schema()->fields.len == 5);
}

struct Test {
    const char *name;
    void (*run)();
};

const Test TESTS[] = {
    {"the_descriptor_is_filled_in", the_descriptor_is_filled_in},
    {"the_schemas_are_the_contract", the_schemas_are_the_contract},
    {"settings_are_read_back_as_they_were_written", settings_are_read_back_as_they_were_written},
    {"an_image_goes_through_with_an_empty_list", an_image_goes_through_with_an_empty_list},
    {"only_when_found_holds_the_image_back", only_when_found_holds_the_image_back},
    {"a_round_with_nothing_in_it_is_refused", a_round_with_nothing_in_it_is_refused},
    {"only_jpeg_is_taken", only_jpeg_is_taken},
};

} // namespace

int main() {
    for (const Test &test : TESTS) {
        const int before = failures;

        test.run();

        std::printf("test %s ... %s\n", test.name, failures == before ? "ok" : "FAILED");
    }

    std::printf("\n%zu test(s), %d failure(s)\n", std::size(TESTS), failures);

    return failures == 0 ? 0 : 1;
}
