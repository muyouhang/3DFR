#pragma once
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include "facedetect-dll.h"

#define DETECT_BUFFER_SIZE 0x20000
using namespace std;
class FaceDetect
{
public:
	FaceDetect();
	~FaceDetect();
	std::pair<cv::Rect, cv::Point> detectFaceAndNTP(cv::Mat image);
private:
	cv::Mat mask;
	vector<int> getPixels(cv::Mat gray_image);
	ofstream result;
	string filename;
	vector<int> last_blocks;
	vector<int> curr_blocks;
};

