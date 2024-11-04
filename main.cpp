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
    float       score;
    float       angle;
    cv::Point2f pos;

    bool operator<(const Candidate &rhs) const {
        return this->score > rhs.score;
    }
};

struct Template {
    std::vector<cv::Point2f> edges;
    std::vector<float>       angles;
};

struct Layer {
    float    angleStep;
    Template templates;
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

    cv::Mat            source;
    std::vector<Layer> layels;
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
                const cv::Size   templateSize,
                const double     maxOverlap,
                double          &maxScore,
                cv::Point       &maxPos) {
    const auto      alone = 1. - maxOverlap;
    const cv::Point offset(static_cast<int>(templateSize.width * alone),
                           static_cast<int>(templateSize.height * alone));
    const cv::Size  size(static_cast<int>(2 * templateSize.width * alone),
                        static_cast<int>(2 * templateSize.height * alone));
    const cv::Rect  rectIgnore(pos - offset, size);

    // clear neighbor
    cv::rectangle(score, rectIgnore, cv::Scalar(-1), cv::FILLED);

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
    return {points, angles};
}

cv::Mat matchTemplate(const cv::Mat &angle, const Template &temp, float rotation) {}

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

    const auto &templateTop       = model.layels[ numLevels - 1 ];
    const auto  topScoreThreshold = minScore * pow(0.9, numLevels - 1);
    const auto  angleStep         = templateTop.angleStep;
    const auto  count             = static_cast<int>(spanAngle / angleStep) + 1;

    cv::Mat dx;
    cv::Mat dy;
    cv::spatialGradient(dstTop, dx, dy);

    dx.convertTo(dx, CV_32FC1);
    dy.convertTo(dy, CV_32FC1);

    cv::Mat mag;
    cv::Mat angle;
    cv::cartToPolar(dx, dy, mag, angle);

    for (int i = 0; i < count; i++) {
        const auto rotation = startAngle + angleStep * i;

        auto result = matchTemplate(angle, templateTop.templates, rotation);

        double    maxScore;
        cv::Point maxPos;
        cv::minMaxLoc(result, nullptr, &maxScore, nullptr, &maxPos);
        if (maxScore < topScoreThreshold) {
            continue;
        }

        candidates.emplace_back(maxScore, angle, cv::Point2f(maxPos));
        for (int j = 0; j < maxCount + CANDIDATE - 1; j++) {
            nextMaxLoc(result, maxPos, templateTop.size(), maxOverlap, maxScore, maxPos);
            if (maxScore < topScoreThreshold) {
                break;
            }

            candidates.emplace_back(maxScore, angle, cv::Point2f(maxPos));
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
    auto                   count = static_cast<int>(candidates.size());

    for (int index = 0; index < count; index++) {
        auto pose    = candidates[ index ];
        bool matched = true;

        for (int currentLevel = numLevels - 1; currentLevel >= 0; currentLevel--) {
            const auto &currentLayel   = model.layels[ currentLevel ];
            const auto  scoreThreshold = minScore * pow(0.9, currentLevel);
            const auto  angleStep      = currentLayel.angleStep;

            for (int i = -1; i <= 1; i++) {
                auto rotation = pose.angle + i * angleStep;

                auto result = matchTemplate(angle, currentLayel.templates, rotation);
            }
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
    auto angleStep = atan(1 / radius);

    Model model{edgeParam, minMag, metric, reduce, radius, src, {}};
    model.layels.emplace_back(angleStep, baseTemplate);
    for (std::size_t i = 1; i < pyramids.size(); i++) {
        center    /= 2.f;
        angleStep *= 2.f;

        model.layels.emplace_back(angleStep, buildTemplate(pyramids[ i ], edgeParam, center));
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
    if (dst.empty() || model.layels.empty()) {
        return {};
    }

    const auto templateLevel = static_cast<int>(model.layels.size());
    if (numLevels < 0 || numLevels > templateLevel) {
        numLevels = templateLevel;
    }

    auto pyramids = buildPyramid(dst, numLevels);

    // compute top
    const std::vector<Candidate> candidates = matchTopLayel();

    // match candidate each layel
    std::vector<Candidate> matched = matchDownLayel();
}

int main(int argc, const char *argv[]) {
    // scene
    auto dst = cv::imread(argv[ 2 ], cv::IMREAD_GRAYSCALE);

    int alignedWidth2  = cv::alignSize(dst.size().width, step);
    int alignedHeight2 = cv::alignSize(dst.size().height, step);

    int paddWidth2  = alignedWidth2 - dst.size().width;
    int paddHeight2 = alignedHeight2 - dst.size().height;

    // template
    cv::Mat sceneImg;
    cv::copyMakeBorder(dst, sceneImg, 0, paddWidth2, 0, paddHeight2, cv::BORDER_REFLECT);
    // cv::imshow("scene1", sceneImg);
    std::cout << sceneImg.size() << std::endl;

    cv::Mat sceneImg2;
    cv::resize(sceneImg, sceneImg2, sceneImg.size() / 2, 0, 0, cv::INTER_AREA);
    // cv::imshow("scene2", sceneImg2);
    std::cout << sceneImg2.size() << std::endl;

    cv::Mat sceneImg3;
    cv::resize(sceneImg2, sceneImg3, sceneImg2.size() / 2, 0, 0, cv::INTER_AREA);
    // cv::imshow("scene3", sceneImg3);
    std::cout << sceneImg3.size() << std::endl;

    cv::Mat sceneImg4;
    cv::resize(sceneImg3, sceneImg4, sceneImg3.size() / 2, 0, 0, cv::INTER_AREA);
    // cv::imshow("scene4", sceneImg4);
    std::cout << sceneImg4.size() << std::endl;

    // match top level
    const float            minMag     = 5;
    const float            minScore   = 0.8;
    const int              maxCount   = 1;
    const int              CANDIDATA  = 5;
    const float            maxOverlap = 0.5;
    std::vector<Candidate> candidates;
    {
        cv::Mat blured;
        cv::GaussianBlur(sceneImg4, blured, cv::Size(5, 5), 0);

        cv::Mat dx;
        cv::Mat dy;
        cv::spatialGradient(blured, dx, dy);

        cv::Mat angle(dx.size(), CV_32FC1);
        cv::Mat mag(dx.size(), CV_32FC1);
        angle.forEach<float>([ & ](float &pixel, const int *pos) {
            auto x = dx.at<short>(pos[ 0 ], pos[ 1 ]);
            auto y = dy.at<short>(pos[ 0 ], pos[ 1 ]);

            auto angle = atan2f(y, x);
            pixel      = angle;

            mag.at<float>(pos[ 0 ], pos[ 1 ]) = sqrt(x * x + y * y) / 4.f;
        });

        cv::Mat score(dx.size(), CV_32FC1);
        for (float rot = 0; rot < CV_2PI; rot += angleStep) {
            auto alpha = std::cos(rot);
            auto beta  = std::sin(rot);

            auto &tempPoints = points4[ 0 ];
            auto &tempAngles = angles4[ 0 ];

            for (int y = 0; y < angle.rows; y++) {
                for (int x = 0; x < angle.cols; x++) {
                    float tmpScore = 0;
                    for (int i = 0; i < tempPoints.size(); i++) {
                        const auto &point = tempPoints[ i ];
                        auto        rx    = point.x * alpha - point.y * beta + x;
                        auto        ry    = point.x * beta + point.y * alpha + y;

                        cv::Point pos(cvRound(rx), cvRound(ry));
                        if (pos.x < 0 || pos.y < 0 || pos.x >= angle.cols || pos.y >= angle.rows ||
                            mag.at<float>(pos) < minMag) {
                            continue;
                        }

                        auto ra   = tempAngles[ i ] + rot - angle.at<float>(pos);
                        tmpScore += cos(ra);
                    }

                    score.at<float>(y, x) = tmpScore / tempPoints.size();
                }
            }

            double    maxScore;
            cv::Point maxPos;
            cv::minMaxLoc(score, nullptr, &maxScore, nullptr, &maxPos);
            if (maxScore < minScore) {
                continue;
            }

            candidates.emplace_back(Candidate{(float)maxScore, rot, maxPos});

            for (int i = 0; i < maxCount + CANDIDATA; i++) {
                nextMaxLoc(score, maxPos, templateImg4.size(), maxOverlap, maxScore, maxPos);
                if (maxScore < minScore) {
                    break;
                }

                candidates.emplace_back(Candidate{(float)maxScore, rot, maxPos});
            }
        }
    }

    std::vector<Candidate> candidates2;
    for (const auto &candidate : candidates) {}

    cv::waitKey();
}