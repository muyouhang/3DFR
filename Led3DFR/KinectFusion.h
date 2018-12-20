#pragma once
#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/kinfu.hpp>
using namespace cv::kinfu;
using namespace std;

class KinectFusion
{
private:
	cv::Ptr<Params> params;
	cv::Ptr<KinFu> kf;
public:
	KinectFusion();
	~KinectFusion();
	void Init(cv::Size frame_size);
	void Update(cv::Mat depth_image);
	void Reset();
	std::vector<std::vector<float>> GetPoints();
	cv::Mat GetRender();
};

