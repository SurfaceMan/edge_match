#include "edgesubpix.h"

constexpr int MIN_AREA = 256;

struct Candidate {
    float       score;
    float       angle;
    cv::Point2f pos;
};

struct Template {
    cv::Size                 size;
    std::vector<cv::Point2f> edges;
    std::vector<float>       angles;
};

struct Layer {
    double   startAngle = 0;
    double   stopAngle  = 0;
    double   angleStep  = 0;
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

    cv::Mat            source;
    std::vector<Layer> layels;
};

struct Pose {
    float x;
    float y;
    float angle;
    float score;
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

Model trainModel(const cv::Mat &src,
                 int            numLevels,
                 float          angleStart,
                 float          angleExtent,
                 float          angleStep,
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
}

std::vector<Pose> matchModel(const cv::Mat &dst,
                             const Model   &model,
                             float          angleStart,
                             float          angleExtent,
                             float          angleStep,
                             int            numMatches,
                             float          maxOverlap,
                             bool           subpixel,
                             int            numLevels,
                             float          greediness) {}

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cout << "Too few arguments" << std::endl;
        return -1;
    }

    auto src = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);

    auto        srcWidth      = static_cast<std::size_t>(src.cols);
    auto        srcHeight     = static_cast<std::size_t>(src.rows);
    std::size_t level         = 4;
    std::size_t step          = 1 << (level - 1);
    auto        alignedWidth  = cv::alignSize(srcWidth, (int)step);
    auto        alignedHeight = cv::alignSize(srcHeight, (int)step);

    std::size_t paddWidth  = alignedWidth - srcWidth;
    std::size_t paddHeight = alignedHeight - srcHeight;

    // template
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
    for (std::size_t i = 0; i < level - 1; i++) {
        const auto &last = pyramids[ i ];
        cv::Mat     tmp;
        cv::resize(last, tmp, last.size() / 2, 0, 0, cv::INTER_AREA);

        pyramids.emplace_back(std::move(tmp));
    }

    std::vector<std::vector<cv::Point2f>> edge4;
    std::vector<std::vector<cv::Vec2f>>   dir4;
    EdgePoint(templateImg4, edge4, dir4, 1., 10, 29);

    cv::Point2f center((templateImg4.cols - 1) / 2.f, (templateImg4.rows - 1) / 2.f);
    auto        angleStep = sizeAngleStep(templateImg4.size());

    std::vector<std::vector<cv::Point2f>> points4;
    std::vector<std::vector<float>>       angles4;

    const int minFeatureLength = 5;
    for (int i = 0; i < dir4.size(); i++) {
        const auto &feature = dir4[ i ];
        if (feature.size() < minFeatureLength) {
            continue;
        }

        const auto              &edge = edge4[ i ];
        std::vector<cv::Point2f> points;
        points.reserve(edge.size());
        for (const auto &point : edge) {
            points.emplace_back(point - center);
        }
        points4.emplace_back(std::move(points));

        std::vector<float> angles;
        for (const auto &dir : feature) {
            auto angle = atan2f(dir[ 1 ], dir[ 0 ]);
            angles.push_back(angle);
        }

        angles4.emplace_back(std::move(angles));
    }

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