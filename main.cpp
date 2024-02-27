#include "edgesubpix3.h"

#include <opencv2/opencv.hpp>

int main() {
    auto src =
        cv::imread("/Users/paopaohi/project/1_match/test_shape_match/2D-Shape-Match/model1.bmp",
                   cv::IMREAD_GRAYSCALE);
    int level         = 4;
    int step          = 1 << (level - 1);
    int alignedWidth  = cv::alignSize(src.size().width, step);
    int alignedHeight = cv::alignSize(src.size().height, step);

    int paddWidth  = alignedWidth - src.size().width;
    int paddHeight = alignedHeight - src.size().height;

    // template
    cv::Mat templateImg;
    cv::copyMakeBorder(src, templateImg, 0, paddWidth, 0, paddHeight, cv::BORDER_REFLECT);
    cv::imshow("img1", templateImg);
    cv::imwrite("img1.png", templateImg);
    std::cout << templateImg.size() << std::endl;

    cv::Mat templateImg2;
    cv::resize(templateImg, templateImg2, templateImg.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("img2", templateImg2);
    cv::imwrite("img2.png", templateImg2);
    std::cout << templateImg2.size() << std::endl;

    cv::Mat templateImg3;
    cv::resize(templateImg2, templateImg3, templateImg2.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("img3", templateImg3);
    cv::imwrite("img3.png", templateImg3);
    std::cout << templateImg3.size() << std::endl;

    cv::Mat templateImg4;
    cv::resize(templateImg3, templateImg4, templateImg3.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("img4", templateImg4);
    cv::imwrite("img4.png", templateImg4);
    std::cout << templateImg4.size() << std::endl;

    // subpixel edge
    std::vector<std::vector<cv::Point2f>> points;
    std::vector<std::vector<cv::Vec2f>>   dirs;
    EdgePoint3(templateImg4, points, dirs, 0.3, 10, 30);

    // scene
    auto dst = cv::imread(
        "/Users/paopaohi/project/1_match/test_shape_match/2D-Shape-Match/model1_src2.bmp",
        cv::IMREAD_GRAYSCALE);

    int alignedWidth2  = cv::alignSize(dst.size().width, step);
    int alignedHeight2 = cv::alignSize(dst.size().height, step);

    int paddWidth2  = alignedWidth2 - dst.size().width;
    int paddHeight2 = alignedHeight2 - dst.size().height;

    // template
    cv::Mat sceneImg;
    cv::copyMakeBorder(dst, sceneImg, 0, paddWidth2, 0, paddHeight2, cv::BORDER_REFLECT);
    cv::imshow("scene1", sceneImg);
    std::cout << sceneImg.size() << std::endl;

    cv::Mat sceneImg2;
    cv::resize(sceneImg, sceneImg2, sceneImg.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("scene2", sceneImg2);
    std::cout << sceneImg2.size() << std::endl;

    cv::Mat sceneImg3;
    cv::resize(sceneImg2, sceneImg3, sceneImg2.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("scene3", sceneImg3);
    std::cout << sceneImg3.size() << std::endl;

    cv::Mat sceneImg4;
    cv::resize(sceneImg3, sceneImg4, sceneImg3.size() / 2, 0, 0, cv::INTER_AREA);
    cv::imshow("scene4", sceneImg4);
    std::cout << sceneImg4.size() << std::endl;

    cv::Mat blured;
    cv::GaussianBlur(sceneImg4, blured, cv::Size(5, 5), 0);

    cv::Mat dx;
    cv::Mat dy;
    cv::spatialGradient(blured, dx, dy);

    cv::waitKey();
}