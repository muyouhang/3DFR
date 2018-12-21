#pragma once
#define _CRT_SECURE_NO_WARNINGS
#ifndef CALCNORMAL_H
#define CALCNORMAL_H

#include <cstdio>
#include <vector>
#include <algorithm>
#include <pcl/io/io.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/surface/mls.h>
#include <opencv2/opencv.hpp>
#include <pcl/search/kdtree.h>
#include <pcl/features/normal_3d.h>
#include <pcl/visualization/cloud_viewer.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/filters/statistical_outlier_removal.h>

using namespace std;
class CalcNormal {

public:
	CalcNormal();
	void SetPoints(std::vector<std::vector<float>> points);
	cv::Mat GetDepth();
	cv::Mat GetNormal();
	pcl::PointCloud<pcl::PointXYZ>::Ptr GetPoints();
	void ShowPoints();
private:
	cv::Mat depth_image;
	cv::Mat normal_image;
	std::vector<std::vector<float>> points;
	pcl::PointCloud<pcl::PointXYZ>::Ptr points_cloud;

	void convertPoints2PointXYZ();
	void convertPointXYZ2Depth();
	void convertPointXYZ2Normal();

	void deOutlier(int neighbour, double dev);
	void upsample(float search_radius, float upsample_radius, float step);
};
#endif // !CALCNORMAL_H

