#pragma once

#include <opencv2/opencv.hpp>

void EdgePoint(const cv::Mat                         &img,
               std::vector<std::vector<cv::Point2f>> &points,
               std::vector<std::vector<cv::Vec2f>>   &dirs,
               float                                  sigma,
               float                                  low,
               float                                  high,
               cv::InputArray                         mask = cv::noArray());
