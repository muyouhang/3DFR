#include "CalcNormal.h"
void CalcNormal::SetPoints(std::vector<std::vector<float>> points) {
	this->points = points;
	this->convertPoints2PointXYZ();
}
void CalcNormal::SetNormal(std::vector<std::vector<float>> normals) {

}
cv::Mat CalcNormal::GetDepth() {
	convertPointXYZ2Depth();
	return depth_image;
}
cv::Mat CalcNormal::GetNormal() {
	convertPointXYZ2Normal();
	return normal_image;
}
pcl::PointCloud<pcl::PointXYZ> CalcNormal::GetPoints() {
	return this->points_cloud;
}
void CalcNormal::convertPoints2PointXYZ() {
	//pcl::PointCloud<pcl::PointXYZ> point_cloud_ptr;
	this->points_cloud.points.clear();
	for (int p = 0; p < points.size(); p++) {
		pcl::PointXYZ point;
		point.x = points.at(p).at(0) * 500;
		point.y = points.at(p).at(1) * 500;
		point.z = points.at(p).at(2) * 500;
		this->points_cloud.points.push_back(point);
	}
	//
	//this->deOutlier(50, 1);

	this->upsample(5, 3, 1.5);
}
void CalcNormal::deOutlier(int neighbour, double dev)
{
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(this->points_cloud.makeShared());
	sor.setMeanK(neighbour);
	sor.setStddevMulThresh(dev);
	sor.filter(this->points_cloud);
}
void CalcNormal::upsample(float search_radius, float upsample_radius, float step) {
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> filter;
	filter.setInputCloud(points_cloud.makeShared());
	filter.setSearchRadius(search_radius);

	pcl::search::KdTree<pcl::PointXYZ>::Ptr kdtree;
	filter.setSearchMethod(kdtree);

	filter.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
	filter.setUpsamplingRadius(upsample_radius);
	filter.setUpsamplingStepSize(step);

	filter.process(points_cloud);
}

void CalcNormal::convertPointXYZ2Depth() {
	double minx = this->points_cloud.points[0].x, maxx = minx, miny = this->points_cloud.points[0].y, maxy = miny, minz = this->points_cloud.points[0].z, maxz = minz;
	for(int i=0;i<this->points_cloud.points.size();i++)
	{
		minx = min(minx, (double)this->points_cloud.points.at(i).x);
		maxx = max(maxx, (double)this->points_cloud.points.at(i).x);
		miny = min(miny, (double)this->points_cloud.points.at(i).y);
		maxy = max(maxy, (double)this->points_cloud.points.at(i).y);
		minz = min(minz, (double)this->points_cloud.points.at(i).z);
		maxz = max(maxz, (double)this->points_cloud.points.at(i).z);
	}
	cv::Mat M((int)(maxy - miny + 1), (int)(maxx - minx + 1), CV_8UC1);
#pragma omp parallel for
	for (int i = 0; i<M.rows; i++)
	{
		for (int j = 0; j<M.cols; j++)
		{
			M.at<uchar>(i, j) = 0;
		}
	}
#pragma omp parallel for
	for (int i = 0; i < this->points_cloud.points.size(); i++)
	{
		int x = (int)(this->points_cloud.points.at(i).x - minx);
		int y = (int)(this->points_cloud.points.at(i).y - miny);
		double pixel = (this->points_cloud.points.at(i).z - minz) / (maxz - minz) * 255;
		if (x < 0 || x > M.cols || y < 0 || y > M.rows) {
			continue;
		}
		if (pixel > M.at<uchar>(y, x))
		{
			M.at<uchar>(y, x) = max(0.0,(this->points_cloud.points.at(i).z - minz) / (maxz - minz) * 255);
		}

	}
	cv::medianBlur(M, M, 3);

	depth_image = M;
}
void CalcNormal::convertPointXYZ2Normal() {
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::PointCloud<pcl::Normal>::Ptr pcNormal(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);

	ne.setInputCloud(this->points_cloud.makeShared());
	ne.setSearchMethod(tree);
	ne.setKSearch(20);
	ne.compute(*pcNormal);
	
	double minx = 9999, maxx = -9999, miny = 9999, maxy =-9999, minz = 9999, maxz = -9999;
	for(int i=0;i<this->points_cloud.points.size();i++)
	{
		minx = min(minx, (double)this->points_cloud.points.at(i).x);
		maxx = max(maxx, (double)this->points_cloud.points.at(i).x);
		miny = min(miny, (double)this->points_cloud.points.at(i).y);
		maxy = max(maxy, (double)this->points_cloud.points.at(i).y);
		minz = min(minz, (double)this->points_cloud.points.at(i).z);
		maxz = max(maxz, (double)this->points_cloud.points.at(i).z);
	}
	cv::Mat normal_image = cv::Mat::zeros((int)(maxy - miny + 1), (int)(maxx - minx + 1), CV_8UC3);
#pragma omp parallel for
	for (int i = 0; i < pcNormal->size(); ++i)
	{
		//Í³Ò»normal³¯Ïò£¬nz>0
		if (pcNormal->points[i].normal_z < 0)
		{
			pcNormal->points[i].normal_x *= -1;
			pcNormal->points[i].normal_y *= -1;
			pcNormal->points[i].normal_z *= -1;
		}
		int loc_x = this->points_cloud.points[i].x - minx;
		int loc_y = this->points_cloud.points[i].y - miny;
		if (loc_x<0 || loc_x>normal_image.cols || loc_y<0 || loc_y>normal_image.rows) {
			continue;
		}
		normal_image.at<cv::Vec3b>(loc_y, loc_x)[2] = max(0,(int)((pcNormal->points[i].normal_x + 1) * 128 - 1));
		normal_image.at<cv::Vec3b>(loc_y, loc_x)[1] = max(0,(int)((pcNormal->points[i].normal_y + 1) * 128 - 1));
		normal_image.at<cv::Vec3b>(loc_y, loc_x)[0] = max(0,(int)((pcNormal->points[i].normal_z + 1) * 128 - 1));
	}
	cv::medianBlur(normal_image,normal_image,3);

	this->normal_image = normal_image;

}
CalcNormal::CalcNormal() {

}
//void CalcNormal::ShowPoints() {
//	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
//	viewer->setBackgroundColor(0, 0, 0);
//	viewer->addPointCloud<pcl::PointXYZ>(points_cloud, "sample cloud");
//	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
//	while (!viewer->wasStopped())
//	{
//		viewer->spinOnce(100);
//		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
//	}
//}