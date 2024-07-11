#pragma once

#include <opencv2/opencv.hpp>

void EdgePoint(const cv::Mat                       &img,
               std::vector<std::list<cv::Point2f>> &points,
               std::vector<std::list<cv::Vec2f>>   &dirs,
               float                                sigma,
               float                                low,
               float                                high);
