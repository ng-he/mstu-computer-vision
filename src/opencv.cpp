// The one backend there is: OpenCV's dnn, which carries the whole way from a
// jpeg to a list of boxes. imdecode reads the image, readNet the model,
// NMSBoxes prunes what overlaps.
//
// What it expects is a YOLOX export: one output of [1, boxes, 5 + classes],
// with the boxes still in grid units because the ONNX releases are exported
// without the decode. The decode is below, and it is the only part of this
// that is particular to a family of models.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <map>

#include <opencv2/dnn.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "detector.hpp"
#include "labels.hpp"

namespace mstu::vision {
namespace {

/// The strides a YOLOX head predicts at, smallest boxes first.
constexpr int STRIDES[] = {8, 16, 32};

/// How many numbers come before the class scores: the box and the objectness.
constexpr int LEADING = 5;

/// One point of the grid a prediction is relative to.
struct Cell {
    float x;
    float y;
    float stride;
};

/// The grid the output rows run through, in the order the model writes them.
std::vector<Cell> grid(int side) {
    std::vector<Cell> cells;

    for (int stride : STRIDES) {
        const int span = side / stride;

        for (int y = 0; y < span; y += 1) {
            for (int x = 0; x < span; x += 1) {
                cells.push_back(
                    Cell{static_cast<float>(x), static_cast<float>(y), static_cast<float>(stride)});
            }
        }
    }

    return cells;
}

class OpenCv final : public Detector {
  public:
    const char *backend() const override { return "opencv dnn, cpu"; }

    bool load(const std::string &path, std::string &why) override {
        try {
            net_ = cv::dnn::readNet(path);
        } catch (const cv::Exception &error) {
            why = error.msg;
            return false;
        }

        if (net_.empty()) {
            why = "the model read as nothing";
            return false;
        }

        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);

        outputs_ = net_.getUnconnectedOutLayersNames();

        // A model trained on something other than COCO brings its own names.
        names_ = names_beside(path);

        if (names_.empty()) {
            names_ = coco();
        }

        // What the model takes is read from its first forward pass, since the
        // export does not have to say.
        side_ = 0;

        return true;
    }

    bool look(const uint8_t *jpeg, size_t len, const Wanted &wanted, std::vector<Detection> &found,
              std::string &why) override {
        found.clear();

        if (net_.empty()) {
            why = "no model";
            return false;
        }

        const cv::Mat encoded(1, static_cast<int>(len), CV_8U, const_cast<uint8_t *>(jpeg));
        const cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);

        if (image.empty()) {
            why = "the image did not read as a jpeg";
            return false;
        }

        try {
            return looked(image, wanted, found, why);
        } catch (const cv::Exception &error) {
            why = error.msg;
            return false;
        }
    }

    bool drawn(const uint8_t *jpeg, size_t len, const std::vector<Detection> &found,
               std::vector<uint8_t> &out, std::string &why) override {
        const cv::Mat encoded(1, static_cast<int>(len), CV_8U, const_cast<uint8_t *>(jpeg));
        cv::Mat image = cv::imdecode(encoded, cv::IMREAD_COLOR);

        if (image.empty()) {
            why = "the image did not read as a jpeg";
            return false;
        }

        // The colour the canvas draws a transform node in, which is what this
        // node is, so a page and the picture on it agree.
        const cv::Scalar ink(255, 112, 169);

        // Thin lines vanish on a large image and swamp a small one.
        const int pen = std::max(1, std::min(image.cols, image.rows) / 240);

        for (const Detection &one : found) {
            const cv::Rect box(static_cast<int>(one.x), static_cast<int>(one.y),
                               static_cast<int>(one.width), static_cast<int>(one.height));

            cv::rectangle(image, box, ink, pen);

            char said[128];

            std::snprintf(said, sizeof(said), "%s %.2f", one.name.c_str(), one.confidence);

            const double size = std::max(0.35, std::min(image.cols, image.rows) / 900.0);
            int base = 0;
            const cv::Size said_at =
                cv::getTextSize(said, cv::FONT_HERSHEY_SIMPLEX, size, pen, &base);

            // Above the box where there is room, inside it where there is
            // not, and never off the edge: a label wider than its own box
            // would otherwise run off the picture and lose its name.
            const int wide = said_at.width + pen * 2;
            const int tall = said_at.height + base;
            const int top = box.y - tall > 0 ? box.y - tall : box.y;
            const int left = std::max(0, std::min(box.x, image.cols - wide));

            cv::rectangle(image, cv::Rect(left, top, std::min(wide, image.cols - left), tall), ink,
                          cv::FILLED);

            cv::putText(image, said, cv::Point(left + pen, top + said_at.height),
                        cv::FONT_HERSHEY_SIMPLEX, size, cv::Scalar(20, 20, 24), pen);
        }

        try {
            if (!cv::imencode(".jpg", image, out)) {
                why = "the picture would not encode";
                return false;
            }
        } catch (const cv::Exception &error) {
            why = error.msg;
            return false;
        }

        return true;
    }

  private:
    bool looked(const cv::Mat &image, const Wanted &wanted, std::vector<Detection> &found,
                std::string &why) {
        if (side_ == 0) {
            side_ = 416;
        }

        // YOLOX takes the image fitted into a square and the rest left grey,
        // top left aligned, at its own scale: no mean, no division, and BGR
        // as the decode gave it.
        const double scale = std::min(side_ / static_cast<double>(image.rows),
                                      side_ / static_cast<double>(image.cols));

        const int fitted_w = static_cast<int>(image.cols * scale);
        const int fitted_h = static_cast<int>(image.rows * scale);

        cv::Mat canvas(side_, side_, CV_8UC3, cv::Scalar(114, 114, 114));
        cv::Mat fitted;

        cv::resize(image, fitted, cv::Size(fitted_w, fitted_h), 0, 0, cv::INTER_LINEAR);
        fitted.copyTo(canvas(cv::Rect(0, 0, fitted_w, fitted_h)));

        const cv::Mat blob = cv::dnn::blobFromImage(canvas, 1.0, cv::Size(side_, side_),
                                                    cv::Scalar(), false, false, CV_32F);

        net_.setInput(blob);

        std::vector<cv::Mat> out;
        net_.forward(out, outputs_);

        if (out.empty() || out[0].dims != 3) {
            why = "the model gave something this does not read";
            return false;
        }

        const int rows = out[0].size[1];
        const int stride = out[0].size[2];
        const int classes = stride - LEADING;

        if (classes <= 0) {
            why = "the model names no classes";
            return false;
        }

        const std::vector<Cell> cells = grid(side_);

        if (static_cast<int>(cells.size()) != rows) {
            why = "the model predicts at sizes this does not expect";
            return false;
        }

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> which;

        const float *row = out[0].ptr<float>();

        for (int at = 0; at < rows; at += 1, row += stride) {
            const Cell &cell = cells[static_cast<size_t>(at)];

            // The objectness gates the whole row, so the class scan is only
            // worth doing for a row that got past it.
            const float objectness = row[4];

            if (objectness < wanted.confidence) {
                continue;
            }

            const float *first = row + LEADING;
            const int best = static_cast<int>(std::max_element(first, first + classes) - first);
            const float score = objectness * first[best];

            if (score < wanted.confidence) {
                continue;
            }

            if (!asked_for(best, wanted)) {
                continue;
            }

            // Grid units into pixels of the square, then back out through the
            // scale the image was fitted by.
            const double cx = (row[0] + cell.x) * cell.stride / scale;
            const double cy = (row[1] + cell.y) * cell.stride / scale;
            const double w = std::exp(row[2]) * cell.stride / scale;
            const double h = std::exp(row[3]) * cell.stride / scale;

            boxes.push_back(cv::Rect(static_cast<int>(cx - w / 2), static_cast<int>(cy - h / 2),
                                     static_cast<int>(w), static_cast<int>(h)));
            scores.push_back(score);
            which.push_back(best);
        }

        prune(boxes, scores, which, wanted, image.size(), found);

        return true;
    }

    bool asked_for(int class_id, const Wanted &wanted) const {
        if (wanted.classes.empty()) {
            return true;
        }

        const std::string &name = named(class_id);

        return std::find(wanted.classes.begin(), wanted.classes.end(), name) !=
               wanted.classes.end();
    }

    const std::string &named(int class_id) const {
        static const std::string unknown = "unknown";

        if (class_id < 0 || static_cast<size_t>(class_id) >= names_.size()) {
            return unknown;
        }

        return names_[static_cast<size_t>(class_id)];
    }

    /// Drops the weaker of two boxes that cover the same thing, one class at
    /// a time: two different things in the same place are both there.
    void prune(const std::vector<cv::Rect> &boxes, const std::vector<float> &scores,
               const std::vector<int> &which, const Wanted &wanted, const cv::Size &bounds,
               std::vector<Detection> &found) const {
        std::map<int, std::vector<int>> by_class;

        for (size_t at = 0; at < which.size(); at += 1) {
            by_class[which[at]].push_back(static_cast<int>(at));
        }

        for (const auto &[class_id, rows] : by_class) {
            std::vector<cv::Rect> mine;
            std::vector<float> theirs;

            for (int at : rows) {
                mine.push_back(boxes[static_cast<size_t>(at)]);
                theirs.push_back(scores[static_cast<size_t>(at)]);
            }

            std::vector<int> kept;

            cv::dnn::NMSBoxes(mine, theirs, static_cast<float>(wanted.confidence),
                              static_cast<float>(wanted.overlap), kept);

            for (int at : kept) {
                const cv::Rect box = mine[static_cast<size_t>(at)] & cv::Rect(cv::Point(), bounds);

                if (box.width <= 0 || box.height <= 0) {
                    continue;
                }

                Detection one;

                one.name = named(class_id);
                one.class_id = static_cast<uint32_t>(class_id);
                one.confidence = theirs[static_cast<size_t>(at)];
                one.x = static_cast<uint32_t>(box.x);
                one.y = static_cast<uint32_t>(box.y);
                one.width = static_cast<uint32_t>(box.width);
                one.height = static_cast<uint32_t>(box.height);

                found.push_back(one);
            }
        }

        // Strongest first, so a limit keeps what matters most.
        std::sort(found.begin(), found.end(), [](const Detection &a, const Detection &b) {
            return a.confidence > b.confidence;
        });

        if (wanted.most > 0 && found.size() > wanted.most) {
            found.resize(wanted.most);
        }
    }

    cv::dnn::Net net_;
    std::vector<std::string> outputs_;
    std::vector<std::string> names_;
    int side_ = 0;
};

} // namespace

std::unique_ptr<Detector> detector() {
    return std::make_unique<OpenCv>();
}

} // namespace mstu::vision
