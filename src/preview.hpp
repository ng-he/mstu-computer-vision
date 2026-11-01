// The latest image, in a file for this plugin's own page.
//
// Live readings carry sizes rather than payloads -- the engine keeps the
// length of a bytes field and not the bytes -- and the desktop already serves
// a local file to a plugin's page. So the picture goes through a file, and the
// reading names it.
#pragma once

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace mstu::vision {

/// How often the file is rewritten. A page redraws at about this rate, and
/// anything faster is writing for nobody.
constexpr auto PREVIEW_EVERY = std::chrono::milliseconds(250);

class Preview {
  public:
    /// One file per plugin instance, beside the engine's own socket where
    /// there is a runtime directory for it.
    explicit Preview(const std::string &id);

    ~Preview();

    Preview(const Preview &) = delete;
    Preview &operator=(const Preview &) = delete;

    /// Whether previews are wanted at all.
    void set(bool on);

    bool wanted() const { return on_; }

    /// The file to show, or nothing when none has been written.
    std::string path() const;

    /// Where the file would be, whether or not one is there.
    const std::filesystem::path &at() const { return path_; }

    /// Keeps `jpeg` for the page, at most as often as PREVIEW_EVERY.
    void keep(const std::vector<uint8_t> &jpeg);

  private:
    bool due() const;

    /// Takes the file away, which the destructor does and a setting may ask
    /// for: a page showing a stale picture is worse than one showing none.
    void forget();

    std::filesystem::path path_;
    std::chrono::steady_clock::time_point wrote_{};
    bool written_ = false;
    bool on_ = true;
};

} // namespace mstu::vision
