#include "edgesubpix.h"

constexpr int MIN_AREA  = 256;
constexpr int CANDIDATE = 5;

struct Pose {
    float x;
    float y;
    float angle;
    float score;
};

struct Candidate {
    double      score;
    float       angle;
    cv::Point2f pos;

    Candidate(double _score, float _angle, cv::Point2f _pos)
        : score(_score)
        , angle(_angle)
        , pos(_pos) {}

    bool operator<(const Candidate &rhs) const {
        return this->score > rhs.score;
    }
};

struct Template {
    float                    angleStep;
    float                    radius;
    std::vector<cv::Point2f> edges;
    std::vector<float>       angles;
};

enum Metric {
    USE_POLARITY,
    IGNORE_LOAL_POLARITY,
    IGNORE_GLOBAL_POLARITY,
};

enum Reduce { NONE, LOW, MEDIUM, HIGH, AUTO };

struct EdgeParam {
    float sigma;
    uchar low;
    uchar high;
    int   minLength;
};

struct Model {
    EdgeParam edgeParam;
    uchar     minMag;
    Metric    metric;
    Reduce    reduce;
    float     radius;

    cv::Mat               source;
    std::vector<Template> templates;
};

// inline double sizeAngleStep(const cv::Size &size) {
//     return atan(2. / std::max(size.width, size.height)) * 180. / CV_PI;
// }

inline double sizeAngleStep(const cv::Size &size) {
    const auto diameter = sqrt(size.width * size.width + size.height * size.height);
    return atan(2. / diameter);
}

int computeLayers(const int width, const int height, const int minArea) {
    assert(width > 0 && height > 0 && minArea > 0);

    auto area  = width * height;
    int  layer = 0;
    while (area > minArea) {
        area /= 4;
        layer++;
    }

    return layer;
}

void nextMaxLoc(cv::Mat         &score,
                const cv::Point &pos,
                float            radius,
                float            maxOverlap,
                double          &maxScore,
                cv::Point       &maxPos) {
    const auto alone       = 1.f - maxOverlap;
    const auto clearRadius = alone * radius;

    // clear neighbor
    cv::circle(score, pos, (int)clearRadius, 0, -1);

    cv::minMaxLoc(score, nullptr, &maxScore, nullptr, &maxPos);
}

Template buildTemplate(const cv::Mat     &src,
                       const EdgeParam   &edgeParam,
                       const cv::Point2f &offset,
                       cv::InputArray     mask = cv::noArray()) {
    std::vector<std::vector<cv::Point2f>> edges;
    std::vector<std::vector<cv::Vec2f>>   dirs;
    EdgePoint(src, edges, dirs, edgeParam.sigma, edgeParam.low, edgeParam.high, mask);

    std::vector<cv::Point2f> points;
    std::vector<float>       angles;
    for (std::size_t i = 0; i < edges.size(); i++) {
        const auto &edge   = edges[ i ];
        const auto &dir    = dirs[ i ];
        const auto  length = static_cast<int>(edge.size());
        if (edgeParam.minLength > length) {
            continue;
        }

        std::vector<float> subAngles(dir.size());
        std::transform(dir.begin(), dir.end(), subAngles.begin(), [](const cv::Vec2f &vec) {
            return atan2f(vec[ 1 ], vec[ 0 ]);
        });

        points.insert(points.end(), edge.begin(), edge.end());
        angles.insert(angles.begin(), subAngles.begin(), subAngles.end());
    }

    std::for_each(points.begin(), points.end(), [ & ](cv::Point2f &point) { point -= offset; });
    return {0, 0, std::move(points), std::move(angles)};
}

cv::Mat matchTemplate(const cv::Mat &angle, const Template &temp, float rotation) {
    cv::Mat score(angle.size(), CV_32FC1);

    auto alpha = std::cos(rotation);
    auto beta  = std::sin(rotation);
    auto size  = temp.edges.size();

    for (int y = 0; y < angle.rows; y++) {
        for (int x = 0; x < angle.cols; x++) {
            float tmpScore = 0;
            for (std::size_t i = 0; i < size; i++) {
                const auto &point = temp.edges[ i ];
                auto        rx    = point.x * alpha - point.y * beta + x;
                auto        ry    = point.x * beta + point.y * alpha + y;

                cv::Point pos(cvRound(rx), cvRound(ry));
                if (pos.x < 0 || pos.y < 0 || pos.x >= angle.cols || pos.y >= angle.rows) {
                    continue;
                }

                auto ra   = temp.angles[ i ] + rotation - angle.at<float>(pos);
                tmpScore += cos(ra);
            }

            score.at<float>(y, x) = tmpScore / size;
        }
    }

    return score;
}

std::vector<cv::Mat> buildPyramid(const cv::Mat &src, int numLevels) {
    auto        srcWidth      = static_cast<std::size_t>(src.cols);
    auto        srcHeight     = static_cast<std::size_t>(src.rows);
    std::size_t step          = 1 << (numLevels - 1);
    auto        alignedWidth  = cv::alignSize(srcWidth, (int)step);
    auto        alignedHeight = cv::alignSize(srcHeight, (int)step);

    std::size_t paddWidth  = alignedWidth - srcWidth;
    std::size_t paddHeight = alignedHeight - srcHeight;

    // build pyramids
    std::vector<cv::Mat> pyramids;
    cv::Mat              templateImg = src;
    if (0 != paddHeight || 0 != paddWidth) {
        cv::copyMakeBorder(src,
                           templateImg,
                           0,
                           (int)paddWidth,
                           0,
                           (int)paddHeight,
                           cv::BORDER_REFLECT);
    }

    pyramids.emplace_back(std::move(templateImg));
    for (std::size_t i = 0; i < numLevels - 1; i++) {
        const auto &last = pyramids[ i ];
        cv::Mat     tmp;
        cv::resize(last, tmp, last.size() / 2, 0, 0, cv::INTER_AREA);

        pyramids.emplace_back(std::move(tmp));
    }

    return pyramids;
}

std::vector<Candidate> matchTopLayel(const cv::Mat &dstTop,
                                     float          startAngle,
                                     float          spanAngle,
                                     float          maxOverlap,
                                     float          minScore,
                                     int            maxCount,
                                     const Model   &model,
                                     int            numLevels) {
    std::vector<Candidate> candidates;

    const auto &templateTop       = model.templates[ numLevels - 1 ];
    const auto  topScoreThreshold = minScore * pow(0.9, numLevels - 1);
    const auto  angleStep         = templateTop.angleStep;
    const auto  count             = static_cast<int>(spanAngle / angleStep) + 1;

    cv::Mat blur;
    cv::GaussianBlur(dstTop, blur, cv::Size{5, 5}, 0);

    cv::Mat dx;
    cv::Mat dy;
    cv::spatialGradient(blur, dx, dy);

    cv::Mat angle(dx.size(), CV_32FC1);
    cv::Mat mag(dx.size(), CV_32FC1);
    angle.forEach<float>([ & ](float &pixel, const int *pos) {
        auto x = dx.at<short>(pos[ 0 ], pos[ 1 ]);
        auto y = dy.at<short>(pos[ 0 ], pos[ 1 ]);

        pixel = atan2f(y, x);

        mag.at<float>(pos[ 0 ], pos[ 1 ]) = sqrt(x * x + y * y) / 4.f;
    });

    for (int i = 0; i < count; i++) {
        const auto rotation = startAngle + angleStep * i;

        auto result = matchTemplate(angle, templateTop, rotation);

        double    maxScore;
        cv::Point maxPos;
        cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);
        if (maxScore < topScoreThreshold) {
            continue;
        }

        candidates.emplace_back(maxScore, rotation, cv::Point2f(maxPos));
        for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
            nextMaxLoc(result, maxPos, templateTop.radius, maxOverlap, maxScore, maxPos);
            if (maxScore < topScoreThreshold) {
                break;
            }

            candidates.emplace_back(maxScore, rotation, cv::Point2f(maxPos));
        }
    }

    std::sort(candidates.begin(), candidates.end());

    return candidates;
}

std::vector<Candidate> matchDownLayel(const std::vector<cv::Mat>   &pyramids,
                                      const std::vector<Candidate> &candidates,
                                      double                        minScore,
                                      int                           subpixel,
                                      const Model                  &model,
                                      int                           numLevels) {
    std::vector<Candidate> levelMatched;
    // auto                   count = static_cast<int>(candidates.size());
    //
    // for (int index = 0; index < count; index++) {
    //     auto pose    = candidates[ index ];
    //     bool matched = true;
    //
    //     for (int currentLevel = numLevels - 1; currentLevel >= 0; currentLevel--) {
    //         const auto &currentLayel   = model.templates[ currentLevel ];
    //         const auto  scoreThreshold = minScore * pow(0.9, currentLevel);
    //         const auto  angleStep      = currentLayel.angleStep;
    //
    //         for (int i = -1; i <= 1; i++) {
    //             auto rotation = pose.angle + i * angleStep;
    //
    //             auto result = matchTemplate(angle, currentLayel, rotation);
    //         }
    //     }
    // }

    return levelMatched;
}

Model trainModel(const cv::Mat &src,
                 int            numLevels,
                 Reduce         reduce,
                 Metric         metric,
                 EdgeParam      edgeParam,
                 uchar          minMag) {
    if (src.empty() || src.channels() != 1) {
        return {};
    }

    if (numLevels <= 0) {
        // level must greater than 0
        numLevels = computeLayers(src.size().width, src.size().height, MIN_AREA);
    }

    const auto scale   = 1 << (numLevels - 1);
    const auto topArea = src.size().area() / (scale * scale);
    if (MIN_AREA > topArea) {
        // top area must greater than MIN_AREA
        return {};
    }

    auto pyramids = buildPyramid(src, numLevels);

    // build bottom template
    cv::Point2f center(0, 0);
    float       radius       = 0;
    auto        baseTemplate = buildTemplate(pyramids.front(), edgeParam, center);
    cv::minEnclosingCircle(baseTemplate.edges, center, radius);
    std::for_each(baseTemplate.edges.begin(), baseTemplate.edges.end(), [ & ](cv::Point2f &point) {
        point -= center;
    });
    auto angleStep         = atan(1 / radius);
    baseTemplate.radius    = radius;
    baseTemplate.angleStep = angleStep;

    Model model{edgeParam, minMag, metric, reduce, radius, src, {}};
    model.templates.emplace_back(std::move(baseTemplate));
    for (std::size_t i = 1; i < pyramids.size(); i++) {
        center /= 2.f;

        auto &temImg = pyramids[ i ];
        model.templates.emplace_back(buildTemplate(temImg, edgeParam, center));
        model.templates.back().radius = radius /= 2.f;
        model.templates.back().angleStep = angleStep *= 2.f;
    }

    return model;
}

std::vector<Pose> matchModel(const cv::Mat &dst,
                             const Model   &model,
                             float          angleStart,
                             float          angleExtent,
                             float          angleStep,
                             float          minScore,
                             int            numMatches,
                             float          maxOverlap,
                             bool           subpixel,
                             int            numLevels,
                             float          greediness) {
    if (dst.empty() || model.templates.empty()) {
        return {};
    }

    const auto templateLevel = static_cast<int>(model.templates.size());
    if (numLevels < 0 || numLevels > templateLevel) {
        numLevels = templateLevel;
    }

    auto pyramids = buildPyramid(dst, numLevels);

    // compute top
    const std::vector<Candidate> candidates = matchTopLayel(pyramids.back(),
                                                            angleStart,
                                                            angleExtent,
                                                            maxOverlap,
                                                            minScore,
                                                            numMatches,
                                                            model,
                                                            numLevels);

    // match candidate each layel
    std::vector<Candidate> matched =
        matchDownLayel(pyramids, candidates, minScore, subpixel, model, numLevels);
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        throw std::runtime_error("too few args");
    }

    auto src = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(argv[ 2 ], cv::IMREAD_GRAYSCALE);

    auto model = trainModel(src, -1, NONE, USE_POLARITY, {0.5, 20, 40, 5}, 10);

    auto result = matchModel(dst, model, 0, CV_2PI, -1, 0.6, 2, 0.5, false, -1, 0.9);

    return {};
}