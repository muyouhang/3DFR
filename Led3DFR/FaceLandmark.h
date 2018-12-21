#pragma once
#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/face.hpp>
class FaceLandmark
{
private:
	cv::CascadeClassifier faceDetector;
	cv::Ptr<cv::face::Facemark> facemark;
public:
	FaceLandmark();
	std::pair<cv::Rect, cv::Point2f> detectFaceAndNTP(cv::Mat gray_image);
	~FaceLandmark();
};