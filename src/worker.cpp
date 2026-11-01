#include "worker.hpp"

namespace mstu::vision {

Worker::~Worker() {
    stop();
}

void Worker::start(const std::string &model) {
    stop();

    stopping_.store(false);
    dropped_.store(0);
    state_.store(State::Loading);

    {
        std::lock_guard<std::mutex> held(lock_);

        waiting_.reset();
        done_.clear();
        note_.clear();
        backend_.clear();
    }

    thread_ = std::thread([this, model] { run(model); });
}

void Worker::stop() {
    if (!thread_.joinable()) {
        return;
    }

    stopping_.store(true);
    wake_.notify_all();
    thread_.join();

    state_.store(State::Idle);
}

std::string Worker::backend() const {
    std::lock_guard<std::mutex> held(lock_);

    return backend_.empty() ? "none" : backend_;
}

std::string Worker::note() const {
    std::lock_guard<std::mutex> held(lock_);

    return note_;
}

void Worker::want(const Wanted &wanted) {
    std::lock_guard<std::mutex> held(lock_);

    wanted_ = wanted;
}

void Worker::preview(bool on) {
    std::lock_guard<std::mutex> held(lock_);

    preview_ = on;
}

void Worker::offer(Looked job) {
    {
        std::lock_guard<std::mutex> held(lock_);

        if (waiting_.has_value()) {
            dropped_.fetch_add(1);
        }

        waiting_ = std::move(job);
    }

    wake_.notify_one();
}

bool Worker::take(Looked &out) {
    std::lock_guard<std::mutex> held(lock_);

    if (done_.empty()) {
        return false;
    }

    out = std::move(done_.front());
    done_.erase(done_.begin());

    return true;
}

void Worker::said(const std::string &why) {
    std::lock_guard<std::mutex> held(lock_);

    note_ = why;
}

void Worker::run(std::string model) {
    std::unique_ptr<Detector> detect = detector();

    if (detect == nullptr) {
        said("built without a detector: opencv was not found when this was compiled");
        state_.store(State::Failed);

        return;
    }

    {
        std::lock_guard<std::mutex> held(lock_);

        backend_ = detect->backend();
    }

    std::string why;

    if (!detect->load(model, why)) {
        said(why.empty() ? "the model would not load" : why);
        state_.store(State::Failed);

        return;
    }

    state_.store(State::Looking);

    while (!stopping_.load()) {
        Looked job;

        {
            std::unique_lock<std::mutex> held(lock_);

            wake_.wait(held, [this] { return stopping_.load() || waiting_.has_value(); });

            if (stopping_.load()) {
                return;
            }

            job = std::move(*waiting_);
            waiting_.reset();
        }

        Wanted wanted;
        bool preview = false;

        {
            std::lock_guard<std::mutex> held(lock_);

            wanted = wanted_;
            preview = preview_;
        }

        if (!detect->look(job.jpeg.data(), job.jpeg.size(), wanted, job.found, why)) {
            said(why.empty() ? "the image could not be looked at" : why);

            // One image that would not read is not a reason to stop: the next
            // one may well be fine, and the note says what happened.
            continue;
        }

        said("");

        // Only worth a decode and an encode when a page is there to look at
        // it, and only worth drawing at all when something was found.
        if (preview && !job.found.empty()) {
            std::string ignored;

            if (!detect->drawn(job.jpeg.data(), job.jpeg.size(), job.found, job.drawn, ignored)) {
                job.drawn.clear();
            }
        }

        std::lock_guard<std::mutex> held(lock_);

        // The oldest goes first if nothing is picking them up, for the same
        // reason an image waiting to be looked at does.
        if (done_.size() >= KEEP_DONE) {
            done_.erase(done_.begin());
            dropped_.fetch_add(1);
        }

        done_.push_back(std::move(job));
    }
}

} // namespace mstu::vision
