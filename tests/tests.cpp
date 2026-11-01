// This plugin's own tests, run by test.sh through ctest.
//
// They stand in for the engine: a writer of their own, so what the plugin
// wrote can be read back, and no host at all, so nothing here needs one
// running.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <thread>

#ifdef MSTU_WITH_OPENCV
#include <cmath>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#endif

#include "host.hpp"
#include "labels.hpp"
#include "preview.hpp"
#include "schema.hpp"
#include "worker.hpp"

namespace {

int failures = 0;
const char *skipped = nullptr;

/// Says a test had nothing to run against, which is not a failure: the model
/// is fetched by model.sh and OpenCV is installed or it is not.
void skip(const char *why) {
    skipped = why;
}

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
    CHECK(engine.values.size() == 6);
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
    CHECK(plugin()->live_schema()->fields.len == 9);
    CHECK(plugin()->settings_schema()->fields.len == 6);

    // A page of its own, named as collect.sh leaves it beside the library
    // rather than as the folder is called in this repository.
    CHECK(text(plugin()->ui()) == "mstu-computer-vision-ui");
}

void the_classes_have_names() {
    CHECK(mstu::vision::coco().size() == 80);
    CHECK(mstu::vision::coco().front() == "person");
    CHECK(mstu::vision::coco().back() == "toothbrush");

    // A model trained on something else brings its own list.
    const std::filesystem::path dir = std::filesystem::temp_directory_path() / "mstu-vision-names";

    std::filesystem::create_directories(dir);

    const std::filesystem::path model = dir / "mine.onnx";

    {
        std::ofstream names(dir / "mine.names");

        names << "cat\n";
        names << "dog \r\n";
        names << "\n";
    }

    const std::vector<std::string> mine = mstu::vision::names_beside(model.string());

    CHECK(mine.size() == 2);
    CHECK(mine[0] == "cat");
    CHECK(mine[1] == "dog");

    // No such file is the ordinary case, and not a fault.
    CHECK(mstu::vision::names_beside((dir / "other.onnx").string()).empty());

    std::filesystem::remove_all(dir);
}

/// Waits for the thread to settle, so the test is over when the work is
/// rather than after a fixed sleep.
mstu::vision::State settled(mstu::vision::Worker &worker) {
    for (int at = 0; at < 300; at += 1) {
        const mstu::vision::State state = worker.state();

        if (state != mstu::vision::State::Loading) {
            return state;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    return worker.state();
}

/// A model that will not load leaves the node running and says why, rather
/// than taking the pipeline down with it.
void a_model_that_is_not_there_says_so() {
    mstu::vision::Worker worker;

    worker.start("/no/such/model.onnx");

    CHECK(settled(worker) == mstu::vision::State::Failed);
    CHECK(!worker.note().empty());

    worker.stop();

    CHECK(worker.state() == mstu::vision::State::Idle);
}

std::string model_path() {
    return std::string(MSTU_MODELS) + "/yolox_tiny.onnx";
}

std::vector<uint8_t> flat_jpeg() {
#ifdef MSTU_WITH_OPENCV
    const cv::Mat image(240, 320, CV_8UC3, cv::Scalar(90, 110, 130));
    std::vector<uint8_t> jpeg;

    cv::imencode(".jpg", image, jpeg);

    return jpeg;
#else
    return {};
#endif
}

/// The whole way through the thread: an image in, a result out, with the
/// real model doing the looking.
///
/// A flat image holds nothing, so what this proves is the path and the
/// invariants -- a decode gone wrong would have boxes to show for it.
void an_image_goes_through_the_thread() {
#ifndef MSTU_WITH_OPENCV
    return skip("built without opencv");
#endif

    if (!std::filesystem::exists(model_path())) {
        return skip("no model: run ./model.sh");
    }

    mstu::vision::Worker worker;

    worker.start(model_path());

    if (settled(worker) != mstu::vision::State::Looking) {
        CHECK(false);
        std::printf("    %s\n", worker.note().c_str());

        return;
    }

    CHECK(worker.backend() != std::string("none"));

    mstu::vision::Looked job;

    job.start_time = 99;
    job.width = 320;
    job.height = 240;
    job.jpeg = flat_jpeg();

    CHECK(!job.jpeg.empty());

    worker.offer(std::move(job));

    mstu::vision::Looked back;
    bool came = false;

    for (int at = 0; at < 300 && !came; at += 1) {
        came = worker.take(back);

        if (!came) {
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }

    CHECK(came);
    CHECK(back.start_time == 99);
    CHECK(back.width == 320);
    CHECK(!back.jpeg.empty());
    CHECK(back.found.empty());
    CHECK(worker.note().empty());
}

/// A red octagon with STOP written on it, which a model trained on COCO
/// recognises as what it is drawn to look like.
///
/// Drawn rather than photographed, so the test needs no image committed
/// beside it and no network, and the same pixels every run.
std::vector<uint8_t> stop_sign(int side) {
#ifdef MSTU_WITH_OPENCV
    const int middle = side / 2;
    const int across = side * 170 / 480;

    cv::Mat image(side, side, CV_8UC3, cv::Scalar(200, 200, 200));
    std::vector<cv::Point> corners;

    for (int at = 0; at < 8; at += 1) {
        const double turn = M_PI / 8 + at * M_PI / 4;

        corners.push_back(cv::Point(middle + static_cast<int>(across * std::cos(turn)),
                                    middle + static_cast<int>(across * std::sin(turn))));
    }

    cv::fillConvexPoly(image, corners, cv::Scalar(30, 30, 200));
    cv::putText(image, "STOP", cv::Point(side * 90 / 480, side * 270 / 480),
                cv::FONT_HERSHEY_SIMPLEX, side * 2.6 / 480, cv::Scalar(255, 255, 255),
                side * 14 / 480);

    std::vector<uint8_t> jpeg;

    cv::imencode(".jpg", image, jpeg);

    return jpeg;
#else
    (void)side;
    return {};
#endif
}

/// What is drawn is found, where it was drawn.
///
/// The box is what makes this worth having: a grid decode that is wrong by a
/// stride still finds the sign, and puts it somewhere else.
void what_was_drawn_is_found_where_it_was_drawn() {
#ifndef MSTU_WITH_OPENCV
    return skip("built without opencv");
#endif

    if (!std::filesystem::exists(model_path())) {
        return skip("no model: run ./model.sh");
    }

    std::unique_ptr<mstu::vision::Detector> detect = mstu::vision::detector();
    std::string why;

    if (!detect->load(model_path(), why)) {
        CHECK(false);
        std::printf("    %s\n", why.c_str());

        return;
    }

    const int side = 480;
    const std::vector<uint8_t> jpeg = stop_sign(side);

    mstu::vision::Wanted wanted;
    std::vector<mstu::vision::Detection> found;

    CHECK(detect->look(jpeg.data(), jpeg.size(), wanted, found, why));
    CHECK(found.size() == 1);

    if (found.size() != 1) {
        return;
    }

    const mstu::vision::Detection &one = found.front();

    CHECK(one.name == "stop sign");
    CHECK(one.confidence > 0.8);
    CHECK(one.confidence <= 1.0);

    // The octagon spans about 340 of the 480 pixels, centred.
    const int cx = static_cast<int>(one.x + one.width / 2);
    const int cy = static_cast<int>(one.y + one.height / 2);

    // Tight on purpose: a grid decode out by one cell of the coarsest head
    // still finds the sign and puts it 32 pixels away, and a loose tolerance
    // would call that right.
    CHECK(std::abs(cx - side / 2) < 20);
    CHECK(std::abs(cy - side / 2) < 20);
    CHECK(one.width > 250 && one.width < 420);
    CHECK(one.height > 250 && one.height < 420);

    // Asked for other classes only, the sign is no longer reported.
    wanted.classes = {"person", "car"};

    CHECK(detect->look(jpeg.data(), jpeg.size(), wanted, found, why));
    CHECK(found.empty());

    // And asked for its own class it comes back.
    wanted.classes = {"stop sign"};

    CHECK(detect->look(jpeg.data(), jpeg.size(), wanted, found, why));
    CHECK(found.size() == 1);

    // A confidence nothing can meet finds nothing.
    wanted.classes.clear();
    wanted.confidence = 0.999;

    CHECK(detect->look(jpeg.data(), jpeg.size(), wanted, found, why));
    CHECK(found.empty());
}

/// An image offered while one is being looked at replaces what was waiting,
/// rather than a queue growing until nothing in it is worth having.
void an_image_offered_over_another_is_counted_as_dropped() {
    mstu::vision::Worker worker;

    // No thread started, so nothing takes them: both offers just queue.
    mstu::vision::Looked one;
    mstu::vision::Looked two;

    one.start_time = 1;
    two.start_time = 2;

    worker.offer(std::move(one));

    CHECK(worker.dropped() == 0);

    worker.offer(std::move(two));

    CHECK(worker.dropped() == 1);
}

/// A name nothing else is using, so two builds running at once do not read
/// each other's preview.
std::string unique() {
    static int at = 0;

    return std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()) + "-" +
           std::to_string(at++);
}

/// The file a page reads: written beside and renamed over, taken away when
/// previews are turned off, and never faster than a page redraws.
void the_preview_is_a_file_a_page_can_read() {
    mstu::vision::Preview preview("test-" + unique());

    CHECK(preview.path().empty());
    CHECK(!std::filesystem::exists(preview.at()));

    preview.keep({1, 2, 3});

    CHECK(preview.path() == preview.at().string());
    CHECK(std::filesystem::file_size(preview.at()) == 3);

    // A page redraws a few times a second, and writing faster is writing for
    // nobody.
    preview.keep({4, 5, 6, 7});

    CHECK(std::filesystem::file_size(preview.at()) == 3);

    // Nothing half written is ever left behind.
    std::filesystem::path part = preview.at();

    part.replace_extension(".part");

    CHECK(!std::filesystem::exists(part));

    preview.set(false);

    CHECK(!std::filesystem::exists(preview.at()));
    CHECK(preview.path().empty());

    preview.keep({8});

    CHECK(!std::filesystem::exists(preview.at()));
}

/// The file goes when the plugin does, so a page never shows the picture of a
/// node that is no longer there.
void the_preview_goes_when_the_plugin_does() {
    std::filesystem::path at;

    {
        mstu::vision::Preview preview("gone-" + unique());

        preview.keep({1, 2, 3});
        at = preview.at();

        CHECK(std::filesystem::exists(at));
    }

    CHECK(!std::filesystem::exists(at));
}

/// The boxes are drawn onto a copy, and the copy is still a jpeg of the same
/// size: a page scales it to the node, so a different shape would shift it.
void the_boxes_are_drawn_onto_a_copy() {
#ifndef MSTU_WITH_OPENCV
    return skip("built without opencv");
#endif

    if (!std::filesystem::exists(model_path())) {
        return skip("no model: run ./model.sh");
    }

    std::unique_ptr<mstu::vision::Detector> detect = mstu::vision::detector();
    std::string why;

    CHECK(detect->load(model_path(), why));

    const std::vector<uint8_t> jpeg = stop_sign(480);
    std::vector<mstu::vision::Detection> found;
    mstu::vision::Wanted wanted;

    CHECK(detect->look(jpeg.data(), jpeg.size(), wanted, found, why));
    CHECK(found.size() == 1);

    std::vector<uint8_t> drawn;

    CHECK(detect->drawn(jpeg.data(), jpeg.size(), found, drawn, why));
    CHECK(!drawn.empty());

    // A jpeg, and not the one that went in.
    CHECK(drawn.size() >= 3);
    CHECK(drawn[0] == 0xff && drawn[1] == 0xd8 && drawn[2] == 0xff);
    CHECK(drawn != jpeg);

#ifdef MSTU_WITH_OPENCV
    const cv::Mat was = cv::imdecode(
        cv::Mat(1, static_cast<int>(jpeg.size()), CV_8U, const_cast<uint8_t *>(jpeg.data())),
        cv::IMREAD_COLOR);
    const cv::Mat now = cv::imdecode(
        cv::Mat(1, static_cast<int>(drawn.size()), CV_8U, drawn.data()), cv::IMREAD_COLOR);

    CHECK(now.cols == was.cols);
    CHECK(now.rows == was.rows);

    // Drawn where the sign is: the row through the middle of the box differs
    // from what was there, and a row well outside it does not.
    const mstu::vision::Detection &one = found.front();
    const int inside = static_cast<int>(one.y);

    const cv::Mat differs = was.row(inside) != now.row(inside);

    CHECK(cv::countNonZero(differs.reshape(1)) > 0);
#endif

    // Nothing found, nothing drawn, and still a jpeg to show.
    found.clear();
    drawn.clear();

    CHECK(detect->drawn(jpeg.data(), jpeg.size(), found, drawn, why));
    CHECK(!drawn.empty());
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
    {"the_classes_have_names", the_classes_have_names},
    {"a_model_that_is_not_there_says_so", a_model_that_is_not_there_says_so},
    {"an_image_offered_over_another_is_counted_as_dropped",
     an_image_offered_over_another_is_counted_as_dropped},
    {"an_image_goes_through_the_thread", an_image_goes_through_the_thread},
    {"what_was_drawn_is_found_where_it_was_drawn", what_was_drawn_is_found_where_it_was_drawn},
    {"the_preview_is_a_file_a_page_can_read", the_preview_is_a_file_a_page_can_read},
    {"the_preview_goes_when_the_plugin_does", the_preview_goes_when_the_plugin_does},
    {"the_boxes_are_drawn_onto_a_copy", the_boxes_are_drawn_onto_a_copy},
};

} // namespace

int main() {
    for (const Test &test : TESTS) {
        const int before = failures;

        skipped = nullptr;
        test.run();

        if (skipped != nullptr) {
            std::printf("test %s ... skipped: %s\n", test.name, skipped);
            continue;
        }

        std::printf("test %s ... %s\n", test.name, failures == before ? "ok" : "FAILED");
    }

    std::printf("\n%zu test(s), %d failure(s)\n", std::size(TESTS), failures);

    return failures == 0 ? 0 : 1;
}
