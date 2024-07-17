#include "edgesubpix.h"

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::cout << "Too few arguments" << std::endl;
        return -1;
    }
    auto img = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);

    cv::Mat  mask(img.size(), CV_8UC1);
    cv::Rect roi(0, 0, 1000, 1000);
    mask(roi) = 1;

    std::vector<std::vector<cv::Point2f>> edge;
    std::vector<std::vector<cv::Vec2f>>   dir;
    EdgePoint(img, edge, dir, 1., 10, 29, mask);
}