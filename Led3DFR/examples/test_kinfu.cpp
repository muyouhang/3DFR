#include "BasicFuncation.h"
#include "ImageProcess.h"
#include "CalcNormal.h"
#include "KinectFusion.h"
#include "FaceLandmark.h"


void test_kinfu() {
	ImageProcess IP;
	BasicFuncation BF;
	FaceLandmark FL;

	std::vector<string> sp;
	std::vector<cv::Mat> depth_map;

	string root_dir = "data/";
	string depth_raw_name = "001_Kinect_FE_1DEPTH.RAW";
	string infrared_raw_name = "001_Kinect_FE_1INFRARED.RAW";

	IP.openDepthVideo(root_dir + depth_raw_name);
	IP.openInfraredVideo(root_dir + infrared_raw_name);

	sp = BF.split(depth_raw_name, ".");

	bool flag = true;
	cv::Rect roi;
	int count = 0;
	while (IP.readDepthImage() && IP.readInfraredImage()) {

		if (count++ >58) { break; }

		cv::Mat depth = IP.getDepthImage();
		cv::Mat infrared = IP.getInfraredImage();

		if (flag) {
			cv::GaussianBlur(infrared, infrared, cv::Size(3, 3), 0, 0);
			infrared = IP.normalizeInfrared(infrared);
			std::pair<cv::Rect, cv::Point> infrared_face = FL.detectFaceAndNTP(infrared);
			roi = cv::Rect(infrared_face.second.x - 128, infrared_face.second.y - 128, 256, 256);
			if (roi.x <= 0 || roi.y <= 0 || (roi.x + roi.width) >= infrared.cols || (roi.y + roi.height) >= infrared.rows) {
				std::cout << "face detected error or face landmark detected error" << std::endl;
				break;
			}
			flag = false;
		}
		cv::Mat depth_face = depth(roi);
		int nose_tip_value = IP.computeNTP(depth_face.clone(), 400, 1000, 10);
		cv::Mat cropped_face = IP.crop3DFace(nose_tip_value, depth_face.clone());
		if (cropped_face.empty()) continue;
		depth_map.push_back(cropped_face.clone());
	}
	CalcNormal CN;
	KinectFusion KF; KF.Init(cv::Size(256, 256));
	for (int i = 0; i < depth_map.size(); i++) {
		char index[2];		sprintf(index, "%02d", i);

		KinectFusion KF_temp;	KF_temp.Init(cv::Size(256, 256));
		KF_temp.Update(depth_map.at(i));
		std::vector<std::vector<float>> points_temp = KF_temp.GetPoints();
		CalcNormal CN_temp;		CN.SetPoints(points_temp);
		cv::Mat depth_face_temp = CN.GetDepth();		
		cv::imwrite("result/depth/" + sp.at(0) + "_" + index + ".jpg", depth_face_temp);

		cv::Mat normal_face_temp = CN.GetNormal();	
		cv::imwrite("result/normal/" + sp.at(0) + "_" + index + ".jpg", normal_face_temp);

		KF.Update(depth_map.at(i));
		std::vector<std::vector<float>> points = KF.GetPoints();
		CN.SetPoints(points);
		cv::Mat depth_face = CN.GetDepth();
		cv::imwrite("result/kinfu_depth/" + sp.at(0) + "_" + index + ".jpg", depth_face);

		cv::Mat normal_face = CN.GetNormal();
		cv::imwrite("result/kinfu_normal/" + sp.at(0) + "_" + index + ".jpg", normal_face);

		cv::imshow("depth", depth_face);
		cv::imshow("normal", normal_face);
		cv::imshow("render",KF.GetRender());
		cv::waitKey(33);
		//KF.Reset();
	}
}
int main() {
	test_kinfu();
	return 0;
}