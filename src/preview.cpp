#include "preview.hpp"

#include <cstdlib>
#include <fstream>

namespace mstu::vision {

Preview::Preview(const std::string &id) {
    const char *runtime = std::getenv("XDG_RUNTIME_DIR");

    const std::filesystem::path dir = runtime != nullptr ? std::filesystem::path(runtime)
                                                         : std::filesystem::temp_directory_path();

    path_ = dir / ("mstu-preview-" + id + ".jpg");
}

Preview::~Preview() {
    forget();
}

void Preview::set(bool on) {
    on_ = on;

    if (!on) {
        forget();
    }
}

std::string Preview::path() const {
    return written_ ? path_.string() : std::string();
}

void Preview::keep(const std::vector<uint8_t> &jpeg) {
    if (!on_ || jpeg.empty() || !due()) {
        return;
    }

    std::filesystem::path part = path_;

    part.replace_extension(".part");

    {
        std::ofstream file(part, std::ios::binary | std::ios::trunc);

        if (!file) {
            return;
        }

        file.write(reinterpret_cast<const char *>(jpeg.data()),
                   static_cast<std::streamsize>(jpeg.size()));

        if (!file) {
            return;
        }
    }

    // Written beside and renamed over, so a page reading it never catches
    // half an image.
    std::error_code failed;

    std::filesystem::rename(part, path_, failed);

    if (failed) {
        std::filesystem::remove(part, failed);

        return;
    }

    wrote_ = std::chrono::steady_clock::now();
    written_ = true;
}

bool Preview::due() const {
    if (!written_) {
        return true;
    }

    return std::chrono::steady_clock::now() - wrote_ >= PREVIEW_EVERY;
}

void Preview::forget() {
    std::error_code failed;

    if (written_) {
        std::filesystem::remove(path_, failed);
        written_ = false;
    }

    std::filesystem::path part = path_;

    part.replace_extension(".part");
    std::filesystem::remove(part, failed);
}

} // namespace mstu::vision
