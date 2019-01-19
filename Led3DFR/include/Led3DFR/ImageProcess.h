#pragma oncec
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>
using namespace std;
class ImageProcess
{
private:
	std::ifstream file_depth, file_infrared, file_color;
	cv::Mat	depth = cv::Mat::zeros(424, 512, CV_16UC1);
	cv::Mat infrared = cv::Mat::zeros(424, 512, CV_16UC1);
	cv::Mat color = cv::Mat::zeros(1080, 1920, CV_8UC3);

public:
	ImageProcess();
	~ImageProcess();
	bool openDepthVideo(string filename);
	bool openInfraredVideo(string filename);
	bool openColorVideo(string filename);

	bool readDepthImage();
	bool readInfraredImage();
	bool readColorImage();

	cv::Mat getDepthImage();
	cv::Mat getInfraredImage();
	cv::Mat getColorImage();

	int computeNTP(cv::Mat image);
	int computeNTP(cv::Mat image,int min=500,int max = 800, int edge=10);

	cv::Mat crop3DFace(int ntp_value,cv::Mat image);
	cv::Mat normalizeInfrared(cv::Mat image);
	cv::Mat segmentDepthFace(cv::Mat depth_face);
	cv::Mat cropDepthFace(cv::Mat depth_face);
	cv::Mat ImageProcess::resize(cv::Mat inputImage, cv::Size size);
	cv::Mat deNoise(cv::Mat image);

	std::pair<cv::Mat,float> computeAdaptiveThreshold(cv::Mat image);
	std::vector<std::vector<int>> transDepth2Points(cv::Mat image);
};

