#include "edgesubpix.h"

int main(int argc, const char *argv[]) {
    if (argc < 3) {
        std::cout << "Too few arguments" << std::endl;
        return -1;
    }
    // auto img = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);
    //
    // cv::Mat  mask(img.size(), CV_8UC1);
    // cv::Rect roi(0, 0, 1000, 1000);
    // mask(roi) = 1;
    //
    // std::vector<std::vector<cv::Point2f>> edge;
    // std::vector<std::vector<cv::Vec2f>>   dir;
    // EdgePoint(img, edge, dir, 1., 10, 29, mask);

    auto src           = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);
    int  level         = 4;
    int  step          = 1 << (level - 1);
    int  alignedWidth  = cv::alignSize(src.size().width, step);
    int  alignedHeight = cv::alignSize(src.size().height, step);

    int paddWidth  = alignedWidth - src.size().width;
    int paddHeight = alignedHeight - src.size().height;

    // template
    cv::Mat templateImg;
    {
        cv::copyMakeBorder(src, templateImg, 0, paddWidth, 0, paddHeight, cv::BORDER_REFLECT);
        // cv::imshow("img1", templateImg);
        // cv::imwrite("img1.png", templateImg);
        // std::cout << templateImg.size() << std::endl;
        // std::vector<std::vector<cv::Point2f>> edge1;
        // std::vector<std::vector<cv::Vec2f>>   dir1;
        // EdgePoint(templateImg, edge1, dir1, 1., 10, 29);
    }

    cv::Mat templateImg2;
    {
        cv::resize(templateImg, templateImg2, templateImg.size() / 2, 0, 0, cv::INTER_AREA);
        // cv::imshow("img2", templateImg2);
        // cv::imwrite("img2.png", templateImg2);
        // std::cout << templateImg2.size() << std::endl;
        // std::vector<std::vector<cv::Point2f>> edge2;
        // std::vector<std::vector<cv::Vec2f>>   dir2;
        // EdgePoint(templateImg2, edge2, dir2, 1., 10, 29);
    }

    cv::Mat templateImg3;
    {
        cv::resize(templateImg2, templateImg3, templateImg2.size() / 2, 0, 0, cv::INTER_AREA);
        // cv::imshow("img3", templateImg3);
        // cv::imwrite("img3.png", templateImg3);
        // std::cout << templateImg3.size() << std::endl;
        // std::vector<std::vector<cv::Point2f>> edge3;
        // std::vector<std::vector<cv::Vec2f>>   dir3;
        // EdgePoint(templateImg3, edge3, dir3, 1., 10, 29);
    }

    cv::Mat templateImg4;
    cv::resize(templateImg3, templateImg4, templateImg3.size() / 2, 0, 0, cv::INTER_AREA);
    // cv::imshow("img4", templateImg4);
    cv::imwrite("img4.png", templateImg4);
    std::cout << templateImg4.size() << std::endl;
    std::vector<std::vector<cv::Point2f>> edge4;
    std::vector<std::vector<cv::Vec2f>>   dir4;
    EdgePoint(templateImg4, edge4, dir4, 1., 10, 29);

    cv::Point2f center((templateImg4.cols - 1) / 2.f, (templateImg4.rows - 1) / 2.f);

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
            if (angle < 0) {
                angle += CV_PI;
            }
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

    cv::Mat blured;
    cv::GaussianBlur(sceneImg4, blured, cv::Size(5, 5), 0);

    cv::Mat dx;
    cv::Mat dy;
    cv::spatialGradient(blured, dx, dy);

    cv::Mat angle(dx.size(), CV_32FC1);
    angle.forEach<float>([ & ](float &pixel, const int *pos) {
        auto x = dx.at<short>(pos[ 0 ], pos[ 1 ]);
        auto y = dy.at<short>(pos[ 0 ], pos[ 1 ]);

        auto angle = atan2(y, x);
        if (angle < 0) {
            angle += CV_PI;
        }

        pixel = angle;
    });

    // score table
    constexpr int count = 16;
    float         vstep = CV_PI / 2. / (count - 1);
    uint8_t       table[ count ];
    for (int i = 0; i < count; i++) {
        table[ i ] = cosf(i * vstep) * 255;
    }
    table[ count - 1 ] = 0;

    cv::waitKey();
}