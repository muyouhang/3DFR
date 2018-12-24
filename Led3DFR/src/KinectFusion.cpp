#include "KinectFusion.h"



KinectFusion::KinectFusion()
{
}
KinectFusion::~KinectFusion()
{
}
void KinectFusion::Init(cv::Size frame_size) {
	params = Params::defaultParams();
	int edge_size = frame_size.width;
	params->frameSize = frame_size;
	params->intr = cv::Matx33f(
		256.f, 0, 128.f,
		0, 256.f, 128.f,
		0, 0, 1
	);
	params->depthFactor = 1000;

	params->bilateral_sigma_depth = 0.04f;
	params->bilateral_sigma_spatial = 4.5; //pixels
	params->bilateral_kernel_size = 7;     //pixels

	params->icpDistThresh = 0.1f;
	params->volumeDims = cv::Vec3i::all(512);
	params->voxelSize = 1.8f / 512;
	params->raycast_step_factor = 0.06f;
	params->tsdf_trunc_dist = 0.01f; //meters;
	params->tsdf_max_weight = 60;   //frames

	cv::setUseOptimized(true);
	kf = KinFu::create(params);

}
bool KinectFusion::Update(cv::Mat depth_image) {

	cv::UMat frame;
	depth_image.copyTo(frame);

	cv::UMat rendered;
	cv::UMat points;
	cv::UMat normals;
	
	cv::UMat cvt8;
	float depthFactor = params->depthFactor;
	convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);

	if (!kf->update(frame))
	{
		kf->reset();
		std::cout << "reset" << std::endl;
		return false;
	}
	return true;
}
void KinectFusion::Reset() {
	kf->reset();
}
std::vector<std::vector<float>> KinectFusion::GetPoints() {
	std::vector<std::vector<float>> point_cloud;
	cv::UMat u_points;
	cv::UMat u_normals;
	kf->getCloud(u_points, u_normals);
	cv::Mat_<cv::Vec4f> points = u_points.getMat(cv::ACCESS_RW);
	cv::Mat_<cv::Vec4f> normals = u_normals.getMat(cv::ACCESS_RW);
	for (int r = 0; r < points.rows; r++) {
		point_cloud.push_back(std::vector<float>{
			points.ptr<cv::Vec4f>(0, 0)[r][0],
			points.ptr<cv::Vec4f>(0, 0)[r][1],
			points.ptr<cv::Vec4f>(0, 0)[r][2]
		});
	}
	return point_cloud;
	//cout << points.ptr<cv::Vec4f>(0, 0)[0][0] << "," << points.ptr<cv::Vec4f>(0, 0)[1] << "," << points.ptr<cv::Vec4f>(0, 0)[2] << "," << points.ptr<cv::Vec4f>(0, 0)[3] << endl;;
}

cv::Mat KinectFusion::GetRender() {
	cv::Mat rendered;
	kf->render(rendered);
	return rendered;
}
