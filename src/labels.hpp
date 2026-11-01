// What a class number is called.
#pragma once

#include <string>
#include <vector>

namespace mstu::vision {

/// The eighty COCO classes, in the order every model trained on it numbers
/// them. A model trained on something else is read from a `.names` file.
const std::vector<std::string> &coco();

/// The names beside the model at `path`: the same name with `.names` in place
/// of its extension, one class per line. Empty when there is no such file,
/// which is the ordinary case and not a fault.
std::vector<std::string> names_beside(const std::string &path);

} // namespace mstu::vision
