// Detection on a thread of its own.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "detector.hpp"

namespace mstu::vision {

/// One image on its way to be looked at, and back again with what was in it.
///
/// It carries a copy of the jpeg, because the engine's own bytes are gone by
/// the time the looking is done and the image has to leave with the boxes
/// that were found in it.
struct Looked {
    uint64_t start_time = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    std::vector<uint8_t> jpeg;
    std::vector<Detection> found;

    /// The same image with the boxes drawn on it, for the page. Empty unless
    /// previews were asked for, since drawing one costs a decode and an
    /// encode that nothing downstream wants.
    std::vector<uint8_t> drawn;
};

enum class State {
    Idle,
    Loading,
    Looking,
    Failed,
};

/// What the plugin hands images to.
///
/// Inference takes tens of milliseconds and `process` runs on one of the
/// engine's own threads, so nothing out there ever waits for a model. An
/// image offered while one is being looked at replaces whatever was waiting:
/// a detection three seconds late is of use to nobody, so the count of
/// dropped images rises rather than a queue growing.
class Worker {
  public:
    Worker() = default;
    ~Worker();

    Worker(const Worker &) = delete;
    Worker &operator=(const Worker &) = delete;

    /// Takes the model on, and reads it on the thread: a model is tens of
    /// megabytes and this is called from the app's own request.
    void start(const std::string &model);

    void stop();

    State state() const { return state_.load(); }

    /// What is doing the work, or what there is instead of a backend.
    std::string backend() const;

    /// Why it is in the state it is, when that is worth saying.
    std::string note() const;

    void want(const Wanted &wanted);

    /// Whether to draw the boxes onto a copy for the page. The drawing is
    /// done here because it is a decode and an encode, which belong off the
    /// engine's thread with the rest of the pixel work.
    void preview(bool on);

    /// Hands one image over, letting go of whatever was still waiting.
    void offer(Looked job);

    /// Takes one finished image, if there is one by now.
    bool take(Looked &out);

    uint64_t dropped() const { return dropped_.load(); }

  private:
    void run(std::string model);

    void said(const std::string &why);

    std::thread thread_;
    std::atomic<State> state_{State::Idle};
    std::atomic<bool> stopping_{false};
    std::atomic<uint64_t> dropped_{0};

    mutable std::mutex lock_;
    std::condition_variable wake_;

    std::optional<Looked> waiting_;
    std::vector<Looked> done_;
    Wanted wanted_;
    bool preview_ = false;
    std::string note_;
    std::string backend_;
};

/// How many finished images may wait to be picked up.
///
/// One a round is taken, and a round comes with every image, so this only
/// has to cover a hiccup rather than a backlog.
constexpr size_t KEEP_DONE = 4;

} // namespace mstu::vision
