#include "edgesubpix.h"

constexpr int   MIN_AREA  = 256;
constexpr int   CANDIDATE = 5;
constexpr float INVALID   = -1.;

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

    Candidate()
        : score(0)
        , angle(0) {}

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

cv::Mat matchTemplate(const cv::Mat  &angle,
                      const Template &temp,
                      float           rotation,
                      const cv::Rect &rect) {
    cv::Mat score(rect.size(), CV_32FC1);

    auto alpha = std::cos(rotation);
    auto beta  = std::sin(rotation);
    auto size  = temp.edges.size();

    for (int py = 0; py < rect.height; py++) {
        for (int px = 0; px < rect.width; px++) {
            float tmpScore = 0;
            int   x        = rect.x + px;
            int   y        = rect.y + py;
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

            score.at<float>(py, px) = tmpScore / size;
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

void buildEdge(const cv::Mat &src, cv::Mat &angle, cv::Mat &mag) {
    cv::Mat blur;
    cv::GaussianBlur(src, blur, cv::Size{5, 5}, 0);

    cv::Mat dx;
    cv::Mat dy;
    cv::spatialGradient(blur, dx, dy);

    angle = cv::Mat(dx.size(), CV_32FC1);
    mag   = cv::Mat(dx.size(), CV_32FC1);
    angle.forEach<float>([ & ](float &pixel, const int *pos) {
        auto x = dx.at<short>(pos[ 0 ], pos[ 1 ]);
        auto y = dy.at<short>(pos[ 0 ], pos[ 1 ]);

        pixel = atan2f(y, x);

        mag.at<float>(pos[ 0 ], pos[ 1 ]) = sqrt(x * x + y * y) / 4.f;
    });
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

    cv::Mat angle;
    cv::Mat mag;
    buildEdge(dstTop, angle, mag);

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
    std::vector<cv::Mat>   angles(numLevels - 1);
    std::vector<cv::Mat>   mags(numLevels - 1);

    for (int i = 0; i < numLevels - 1; i++) {
        cv::Mat angle;
        cv::Mat mag;
        buildEdge(pyramids[ i ], angle, mag);

        angles[ i ] = std::move(angle);
        mags[ i ]   = std::move(mag);
    }

    auto count = static_cast<int>(candidates.size());

    for (int index = 0; index < count; index++) {
        auto pose    = candidates[ index ];
        bool matched = true;

        for (int currentLevel = numLevels - 2; currentLevel >= 0; currentLevel--) {
            const auto    &currentTemp    = model.templates[ currentLevel ];
            const auto     scoreThreshold = minScore * pow(0.9, currentLevel);
            const auto     angleStep      = currentTemp.angleStep;
            const auto     center         = pose.pos * 2.f;
            const cv::Rect rect(center.x - 3, center.y - 3, 7, 7);

            Candidate newCandidate;
            for (int i = -1; i <= 1; i++) {
                auto rotation = pose.angle + i * angleStep;
                auto result   = matchTemplate(angles[ currentLevel ], currentTemp, rotation, rect);

                double    maxScore;
                cv::Point maxPos;
                cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);

                if (newCandidate.score >= maxScore || maxScore < scoreThreshold) {
                    continue;
                }

                newCandidate = {maxScore, rotation, maxPos + rect.tl()};
            }

            if (newCandidate.score < scoreThreshold) {
                matched = false;
                break;
            }

            pose = newCandidate;
        }

        if (!matched) {
            continue;
        }

        levelMatched.push_back(pose);
    }
    std::sort(levelMatched.begin(), levelMatched.end());

    return levelMatched;
}

void filterOverlap(std::vector<Candidate> &candidates, float maxOverlap, float radius) {
    float      minDist = radius * radius * maxOverlap * maxOverlap;
    const auto size    = candidates.size();
    for (std::size_t i = 0; i < size; i++) {
        auto &candidate = candidates[ i ];
        if (candidate.score < 0) {
            continue;
        }
        for (std::size_t j = i + 1; j < size; j++) {
            auto &refCandidate = candidates[ j ];

            if (refCandidate.score < 0) {
                continue;
            }

            auto delta = candidate.pos - refCandidate.pos;
            auto dist  = delta.dot(delta);
            if (dist > minDist) {
                continue;
            }

            (candidate.score > refCandidate.score ? refCandidate.score : candidate.score) = INVALID;
        }
    }
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

    filterOverlap(matched, maxOverlap, model.templates.front().radius);

    std::vector<Pose> result;
    {
        const auto count = matched.size();
        for (std::size_t i = 0; i < count; i++) {
            const auto &candidate = matched[ i ];

            if (candidate.score < 0) {
                continue;
            }

            result.emplace_back(
                Pose{candidate.pos.x, candidate.pos.y, candidate.angle, (float)candidate.score});
        }

        std::sort(result.begin(), result.end(), [](const Pose &a, const Pose &b) {
            return a.score > b.score;
        });
    }

    return result;
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        throw std::runtime_error("too few args");
    }

    auto src = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(argv[ 2 ], cv::IMREAD_GRAYSCALE);

    auto model = trainModel(src, -1, NONE, USE_POLARITY, {1, 10, 29, 5}, 10);

    auto result = matchModel(dst, model, 0, CV_2PI, -1, 0.9, 2, 0.5, false, -1, 0.9);

    cv::Mat color;
    cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for (int i = 0; i < result.size(); i++) {
        auto           &pose = result[ i ];
        cv::RotatedRect rect(cv::Point2f(pose.x, pose.y), src.size(), -pose.angle);

        cv::Point2f pts[ 4 ];
        rect.points(pts);

        cv::line(color, pts[ 0 ], pts[ 1 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 1 ], pts[ 2 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 2 ], pts[ 3 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::line(color, pts[ 3 ], pts[ 0 ], cv::Scalar(255, 0, 0), 1, cv::LINE_AA);

        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }

    cv::imshow("img", color);
    cv::waitKey();
}