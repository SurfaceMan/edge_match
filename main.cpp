#include "edgesubpix.h"

constexpr int   MIN_AREA  = 256;
constexpr int   CANDIDATE = 5;
constexpr float INVALID   = -1.;
constexpr float F_2PI     = 6.283185307179586476925286766559f;
constexpr float COS[]     = {
    1.f,        0.994522f,  0.978148f,  0.951057f,  0.913545f,  0.866025f,  0.809017f,  0.743145f,
    0.669131f,  0.587785f,  0.5f,       0.406737f,  0.309017f,  0.207912f,  0.104528f,  0.f,
    -0.104529f, -0.207912f, -0.309017f, -0.406737f, -0.5f,      -0.587785f, -0.669131f, -0.743145f,
    -0.809017f, -0.866025f, -0.913545f, -0.951056f, -0.978148f, -0.994522f, -1.f,       -0.994522f,
    -0.978148f, -0.951056f, -0.913545f, -0.866025f, -0.809017f, -0.743145f, -0.669131f, -0.587785f,
    -0.5f,      -0.406737f, -0.309017f, -0.207912f, -0.104528f, 0.f,        0.104528f,  0.207912f,
    0.309017f,  0.406737f,  0.5f,       0.587785f,  0.669131f,  0.743145f,  0.809017f,  0.866025f,
    0.913545f,  0.951056f,  0.978148f,  0.9999f,    1.f};

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

enum Reduce { NONE = 0, LOW = 10, MEDIUM = 5, HIGH = 2, AUTO };

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
    std::vector<Template> reducedTemplates;
};

// inline double sizeAngleStep(const cv::Size &size) {
//     return atan(2. / std::max(size.width, size.height)) * 180. / CV_PI;
// }

Template downSample(Template &src, int step) {
    auto size         = src.angles.size();
    auto reduceCount  = size / step;
    auto reserveCount = size - reduceCount;

    std::vector<cv::Point2f> edges;
    std::vector<float>       angles;

    edges.reserve(reserveCount);
    angles.reserve(reserveCount);

    int count = 0;
    for (std::size_t i = 0; i < size; i++) {
        count++;
        if (count == step) {
            count = 0;
            continue;
        }

        edges.push_back(src.edges[ i ]);
        angles.push_back(src.angles[ i ]);
    }

    return {src.angleStep, src.radius, edges, angles};
}

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
                auto        rx    = point.x * alpha - point.y * beta;
                auto        ry    = point.x * beta + point.y * alpha;

                cv::Point pos(cvRound(rx), cvRound(ry));
                pos.x += x;
                pos.y += y;
                if (pos.x < 0 || pos.y < 0 || pos.x >= angle.cols || pos.y >= angle.rows) {
                    continue;
                }

                auto ra = temp.angles[ i ] + rotation - angle.at<float>(pos);
                ra      = fabs(ra);
                if (ra > F_2PI) {
                    ra -= F_2PI;
                }
                int index  = cvCeil(ra * 9.54927f);
                tmpScore  += COS[ index ];
                // tmpScore += cos(ra);
            }

            score.at<float>(y, x) = tmpScore / static_cast<float>(size);
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

    std::vector<cv::Point> tmpEdge(size);
    std::transform(temp.edges.begin(),
                   temp.edges.end(),
                   tmpEdge.begin(),
                   [ & ](const cv::Point2f &point) {
                       auto rx = point.x * alpha - point.y * beta;
                       auto ry = point.x * beta + point.y * alpha;

                       return cv::Point(cvRound(rx), cvRound(ry));
                   });

    for (int py = 0; py < rect.height; py++) {
        for (int px = 0; px < rect.width; px++) {
            float tmpScore = 0;
            int   x        = rect.x + px;
            int   y        = rect.y + py;
            for (std::size_t i = 0; i < size; i++) {
                auto pos  = tmpEdge[ i ];
                pos.x    += x;
                pos.y    += y;
                if (pos.x < 0 || pos.y < 0 || pos.x >= angle.cols || pos.y >= angle.rows) {
                    continue;
                }

                auto ra = temp.angles[ i ] + rotation - angle.at<float>(pos);
                ra      = fabs(ra);
                if (ra > F_2PI) {
                    ra -= F_2PI;
                }
                int index  = cvCeil(ra * 9.54927f); // ceil(ra / 0.10472f);
                tmpScore  += COS[ index ];
                // tmpScore += cos(ra);
            }

            score.at<float>(py, px) = tmpScore / static_cast<float>(size);
        }
    }

    return score;
}

std::vector<cv::Mat> buildPyramid(const cv::Mat &src, int numLevels) {
    auto srcWidth      = static_cast<std::size_t>(src.cols);
    auto srcHeight     = static_cast<std::size_t>(src.rows);
    auto step          = 1 << static_cast<std::size_t>(numLevels - 1);
    auto alignedWidth  = cv::alignSize(srcWidth, (int)step);
    auto alignedHeight = cv::alignSize(srcHeight, (int)step);

    std::size_t padWidth  = alignedWidth - srcWidth;
    std::size_t padHeight = alignedHeight - srcHeight;

    // build pyramids
    std::vector<cv::Mat> pyramids;
    cv::Mat              templateImg = src;
    if (0 != padHeight || 0 != padWidth) {
        cv::copyMakeBorder(src,
                           templateImg,
                           0,
                           (int)padWidth,
                           0,
                           (int)padHeight,
                           cv::BORDER_REFLECT);
    }

    pyramids.emplace_back(std::move(templateImg));
    for (std::size_t i = 0; i < static_cast<std::size_t>(numLevels - 1); i++) {
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

        pixel                             = atan2f(y, x);
        auto sqrMag                       = static_cast<float>(x * x + y * y);
        mag.at<float>(pos[ 0 ], pos[ 1 ]) = sqrtf(sqrMag) / 4.f;
    });
}

#pragma omp declare reduction(combine : std::vector<Candidate> : omp_out                           \
                                  .insert(omp_out.end(), omp_in.begin(), omp_in.end()))

std::vector<Candidate> matchTopLayer(const cv::Mat &dstTop,
                                     float          startAngle,
                                     float          spanAngle,
                                     float          maxOverlap,
                                     float          minScore,
                                     int            maxCount,
                                     const Model   &model,
                                     int            numLevels) {
    std::vector<Candidate> candidates;

    const auto &templates         = NONE == model.reduce ? model.templates : model.reducedTemplates;
    const auto &templateTop       = templates[ numLevels - 1 ];
    const auto  topScoreThreshold = minScore * pow(0.9, numLevels - 1);
    const auto  angleStep         = templateTop.angleStep;
    const auto  count             = static_cast<int>(spanAngle / angleStep) + 1;

    cv::Mat angle;
    cv::Mat mag;
    buildEdge(dstTop, angle, mag);

#pragma omp parallel for reduction(combine : candidates)
    for (int i = 0; i < count; i++) {
        const auto rotation = startAngle + angleStep * i;

        auto result =
            matchTemplate(angle, templateTop, rotation, cv::Rect(0, 0, angle.cols, angle.rows));

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

std::vector<Candidate> matchDownLayer(const std::vector<cv::Mat>   &pyramids,
                                      const std::vector<Candidate> &candidates,
                                      double                        minScore,
                                      int                           subpixel,
                                      const Model                  &model,
                                      int                           numLevels) {
    (void)(subpixel);

    std::vector<Candidate> levelMatched;
    std::vector<cv::Mat>   angles(numLevels - 1);
    std::vector<cv::Mat>   mags(numLevels - 1);

    for (std::size_t i = 0; i < static_cast<std::size_t>(numLevels - 1); i++) {
        cv::Mat angle;
        cv::Mat mag;
        buildEdge(pyramids[ i ], angle, mag);

        angles[ i ] = std::move(angle);
        mags[ i ]   = std::move(mag);
    }

    auto        count     = candidates.size();
    const auto &templates = NONE == model.reduce ? model.templates : model.reducedTemplates;

#pragma omp parallel for reduction(combine : levelMatched)
    for (std::size_t index = 0; index < count; index++) {
        auto pose    = candidates[ index ];
        bool matched = true;

        for (int currentLevel = numLevels - 2; currentLevel >= 0; currentLevel--) {
            const auto    &currentTemp    = templates[ currentLevel ];
            const auto     scoreThreshold = minScore * pow(0.9, currentLevel);
            const auto     angleStep      = currentTemp.angleStep;
            const auto     center         = pose.pos * 2.f;
            const cv::Rect rect(cvRound(center.x) - 3, cvRound(center.y) - 3, 7, 7);

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

    Model model{edgeParam, minMag, metric, reduce, radius, src, {}, {}};
    model.templates.emplace_back(std::move(baseTemplate));
    for (std::size_t i = 1; i < pyramids.size(); i++) {
        center /= 2.f;

        auto &temImg = pyramids[ i ];
        model.templates.emplace_back(buildTemplate(temImg, edgeParam, center));
        model.templates.back().radius = radius /= 2.f;
        model.templates.back().angleStep = angleStep *= 2.f;
    }

    if (NONE != reduce) {
        model.reducedTemplates.reserve(model.templates.size());
        for (auto &temp : model.templates) {
            model.reducedTemplates.emplace_back(downSample(temp, reduce));
        }
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
    (void)(angleStep);
    (void)(greediness);

    if (dst.empty() || model.templates.empty()) {
        return {};
    }

    const auto templateLevel = static_cast<int>(model.templates.size());
    if (numLevels < 0 || numLevels > templateLevel) {
        numLevels = templateLevel;
    }

    auto pyramids = buildPyramid(dst, numLevels);

    // compute top
    const std::vector<Candidate> candidates = matchTopLayer(pyramids.back(),
                                                            angleStart,
                                                            angleExtent,
                                                            maxOverlap,
                                                            minScore,
                                                            numMatches,
                                                            model,
                                                            numLevels);

    // match candidate each Layer
    std::vector<Candidate> matched =
        matchDownLayer(pyramids, candidates, minScore, subpixel, model, numLevels);

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

void drawEdge(cv::Mat &img, const Pose &pose, const Template &temp) {
    auto alpha = std::cos(pose.angle);
    auto beta  = std::sin(pose.angle);

    for (const auto &point : temp.edges) {
        auto      rx = point.x * alpha - point.y * beta + pose.x;
        auto      ry = point.x * beta + point.y * alpha + pose.y;
        cv::Point pos(cvRound(rx), cvRound(ry));

        cv::circle(img, pos, 1, cv::Scalar(0, 0, 255));
    }
}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        throw std::runtime_error("too few args");
    }

    auto src = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);
    auto dst = cv::imread(argv[ 2 ], cv::IMREAD_GRAYSCALE);

    auto t1     = cv::getTickCount();
    auto model  = trainModel(src, -1, HIGH, USE_POLARITY, {1, 10, 29, 5}, 10);
    auto t2     = cv::getTickCount();
    auto result = matchModel(dst, model, 0, F_2PI, -1, 0.9f, 2, 0.5f, false, -1, 0.9f);
    auto t3     = cv::getTickCount();

    auto trainCost = double(t2 - t1) / cv::getTickFrequency();
    std::cout << "train(s):" << trainCost << std::endl;

    auto matchCost = double(t3 - t2) / cv::getTickFrequency();
    std::cout << "match(s):" << matchCost << std::endl;

    // cv::Mat color;
    // cv::cvtColor(dst, color, cv::COLOR_GRAY2RGB);
    for (const auto &pose : result) {
        // drawEdge(color, pose, model.templates.front());
        std::cout << pose.x << "," << pose.y << "," << pose.angle << "," << pose.score << std::endl;
    }
    //
    // cv::imshow("img", color);
    // cv::waitKey();
}
