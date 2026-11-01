// The calls the engine lends a plugin, with this plugin's name on them.
#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "mstu-sdk.h"

namespace mstu::vision {

/// A `Str` over text the caller goes on owning.
inline Str str(std::string_view text) {
    return Str{reinterpret_cast<const uint8_t *>(text.data()), text.size()};
}

inline std::string_view view(Str text) {
    if (text.ptr == nullptr) {
        return {};
    }

    return std::string_view{reinterpret_cast<const char *>(text.ptr), text.len};
}

/// The log, the events and the live readings, as one thing to hold.
///
/// The tests build one over a null context and every call then does nothing,
/// rather than each caller asking whether there is a host.
class Host {
  public:
    Host() = default;

    Host(const HostContext *ctx, std::string id) : ctx_(ctx), id_(std::move(id)) {}

    const std::string &id() const { return id_; }

    bool logs(LogLevel level) const { return ctx_ != nullptr && ctx_->log_enabled(level); }

    void log(LogLevel level, std::string_view message) const {
        if (ctx_ == nullptr) {
            return;
        }

        // Targets read as `Computer vision (a1b2c3)`, so instances stay apart.
        const std::string target = "Computer vision (" + id_ + ")";

        ctx_->logger(level, str(target), str(message));
    }

    /// Fills one event and publishes it on its own.
    ///
    /// Events are queued and flushed, so a batch of one is the simple case:
    /// nothing here raises two of the same event in a round.
    bool raise(uintptr_t event, const Value *values, uintptr_t len) const {
        if (ctx_ == nullptr) {
            return false;
        }

        Writer *writer = const_cast<Writer *>(ctx_->new_event(str(id_), event));

        if (writer == nullptr) {
            return false;
        }

        if (!mstu_fill(writer, values, len)) {
            return false;
        }

        ctx_->publish_events(str(id_), event);

        return true;
    }

    void live(const Value *values, uintptr_t len) const {
        if (ctx_ == nullptr) {
            return;
        }

        mstu_publish_live(ctx_, str(id_), values, len);
    }

  private:
    const HostContext *ctx_ = nullptr;
    std::string id_;
};

} // namespace mstu::vision
