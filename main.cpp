#include "edgesubpix.h"

int main(int argc, const char *argv[]) {
    if (argc < 2) {
        std::cout << "Too few arguments" << std::endl;
        return -1;
    }
    auto img = cv::imread(argv[ 1 ], cv::IMREAD_GRAYSCALE);

    std::vector<std::list<cv::Point2f>> edge;
    std::vector<std::list<cv::Vec2f>>   dir;
    EdgePoint(img, edge, dir, 1., 10, 29);
}