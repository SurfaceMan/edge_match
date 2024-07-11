#include <opencv2/opencv.hpp>

void gradient(cv::InputArray  _src,
                     cv::OutputArray _grad,
                     cv::OutputArray _mag,
                     int             ksize      = 3,
                     int             borderType = cv::BORDER_DEFAULT);