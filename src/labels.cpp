#include "labels.hpp"

#include <filesystem>
#include <fstream>

namespace mstu::vision {

const std::vector<std::string> &coco() {
    static const std::vector<std::string> NAMES = {
        "person",        "bicycle",      "car",
        "motorcycle",    "airplane",     "bus",
        "train",         "truck",        "boat",
        "traffic light", "fire hydrant", "stop sign",
        "parking meter", "bench",        "bird",
        "cat",           "dog",          "horse",
        "sheep",         "cow",          "elephant",
        "bear",          "zebra",        "giraffe",
        "backpack",      "umbrella",     "handbag",
        "tie",           "suitcase",     "frisbee",
        "skis",          "snowboard",    "sports ball",
        "kite",          "baseball bat", "baseball glove",
        "skateboard",    "surfboard",    "tennis racket",
        "bottle",        "wine glass",   "cup",
        "fork",          "knife",        "spoon",
        "bowl",          "banana",       "apple",
        "sandwich",      "orange",       "broccoli",
        "carrot",        "hot dog",      "pizza",
        "donut",         "cake",         "chair",
        "couch",         "potted plant", "bed",
        "dining table",  "toilet",       "tv",
        "laptop",        "mouse",        "remote",
        "keyboard",      "cell phone",   "microwave",
        "oven",          "toaster",      "sink",
        "refrigerator",  "book",         "clock",
        "vase",          "scissors",     "teddy bear",
        "hair drier",    "toothbrush",
    };

    return NAMES;
}

std::vector<std::string> names_beside(const std::string &path) {
    std::filesystem::path beside(path);

    beside.replace_extension(".names");

    std::ifstream file(beside);
    std::vector<std::string> names;

    if (!file) {
        return names;
    }

    std::string line;

    while (std::getline(file, line)) {
        // A trailing carriage return from a file written on Windows.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) {
            line.pop_back();
        }

        if (!line.empty()) {
            names.push_back(line);
        }
    }

    return names;
}

} // namespace mstu::vision
