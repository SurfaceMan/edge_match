#ifndef EDGESUBPIX2_H
#define EDGESUBPIX2_H

#include <opencv2/opencv.hpp>

void EdgePoint3(const cv::Mat &img, std::vector<std::vector<cv::Point2f>> &points,
                std::vector<std::vector<cv::Vec2f>> &dirs, float sigma, float low, float high);
#endif // EDGESUBPIX2_H
