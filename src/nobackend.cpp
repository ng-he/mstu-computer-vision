// Built without OpenCV, so there is nothing to detect with.
//
// The node still loads, still describes itself and still hands images on: a
// pipeline drawn around it keeps working, and the live reading says why
// nothing is being found.

#include "detector.hpp"

namespace mstu::vision {

std::unique_ptr<Detector> detector() {
    return nullptr;
}

} // namespace mstu::vision
