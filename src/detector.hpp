// The seam: a jpeg in, what is in it out.
//
// The decode, the inference and the pruning of overlapping boxes all live
// behind this one call rather than being three stages with a raster frame
// between them -- so a backend that keeps its pixels on a GPU can take its
// place without anything out here agreeing on where pixels live.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace mstu::vision {

/// One object found, in the pixels of the image it was found in.
struct Detection {
    std::string name;
    uint32_t class_id = 0;
    double confidence = 0;
    uint32_t x = 0;
    uint32_t y = 0;
    uint32_t width = 0;
    uint32_t height = 0;
};

/// What the settings ask of one look.
struct Wanted {
    /// How sure the model must be before an object is worth reporting.
    double confidence = 0.5;

    /// How much two boxes may overlap before the weaker one is dropped.
    double overlap = 0.45;

    /// How many objects one image may report, strongest first. 0 is no limit.
    uint32_t most = 0;

    /// Only these classes, by the model's own names. Empty is every class.
    std::vector<std::string> classes;
};

class Detector {
  public:
    virtual ~Detector() = default;

    /// What is doing the work, for the live readings.
    virtual const char *backend() const = 0;

    /// Reads the model at `path`. False leaves `why` worth showing.
    virtual bool load(const std::string &path, std::string &why) = 0;

    /// Looks at one jpeg. False leaves `why` worth showing; an image with
    /// nothing in it is true and an empty `found`.
    virtual bool look(const uint8_t *jpeg, size_t len, const Wanted &wanted,
                      std::vector<Detection> &found, std::string &why) = 0;

    /// The same image with the boxes drawn on it, for the plugin's own page.
    ///
    /// Here rather than beside the preview because drawing is pixel work, and
    /// pixels are what a backend owns: one that keeps them on a GPU draws
    /// there and hands back only the jpeg.
    virtual bool drawn(const uint8_t *jpeg, size_t len, const std::vector<Detection> &found,
                       std::vector<uint8_t> &out, std::string &why) = 0;
};

/// The backend this was built with, or nothing when it was built without one.
std::unique_ptr<Detector> detector();

} // namespace mstu::vision
