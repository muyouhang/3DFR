#include "BasicFuncation.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"
#include "ImageProcess.h"
#include "CalcNormal.h"
#include "KinectFusion.h"
#include "FaceLandmark.h"
#include <fstream>
#include <direct.h>
#include <future>
#include <chrono>
void process_lock3dface() {
	ofstream log_file("log.txt");

	BasicFuncation BF;
	ImageProcess IP;
	KinectFusion KF; KF.Init(cv::Size(256, 256));

	ifstream depth_data("E:/Dataset/lock3dface/DATA/kinect2_all.dat");
	ifstream depth_label("E:/Dataset/lock3dface/DATA/kinect2_all.name");
	ifstream error_data("E:/Dataset/Lock3DFace_face/error_video_depth.txt");

	std::vector<string> error_videos;
	while (!error_data.eof()) {
		string line;
		getline(error_data, line);
		std::vector<string> sp = BF.split(line, " ");
		if (sp.size()>0)	error_videos.push_back(sp.at(0));
	}

	string line;
	string label;
	string save_path = "E:/Dataset/Lock3DFace_crop/";
	cv::Mat mask(cv::Size(256, 256), CV_16UC1);
	int count = 0;
	int unused = 0;
	//cout << "unused=";
	//cin >> unused;
	int control = 0;
	while (getline(depth_label, label)) {

		std::cout << count++ << "  " << label << std::endl;
		std::vector<string> sp = BF.split(label, ",");
		log_file << count - 1 << "  " << sp.at(0) << std::endl;
		if (_access((save_path + "depth/" + sp.at(0)).data(), 0) == -1) {
			_mkdir((save_path + "depth/" + sp.at(0).data()).data());
		}
		if (_access((save_path + "depth_kinfu/" + sp.at(0)).data(), 0) == -1) {
			_mkdir((save_path + "depth_kinfu/" + sp.at(0).data()).data());
		}
		int nose_tip_value = -1;
		for (int i = 0; i < BF.str2int(sp.at(1)); i++) {
			if (i % 10 == 0) KF.Reset();
			char index[2];		sprintf(index, "%02d", i);
			getline(depth_data, line);
			if (unused > 0) continue;
			//if(std::count(error_videos.begin(),error_videos.end(),sp[0])==0) continue;
			//if (i > 0) continue;

			std::vector<string> raw_data = BF.split(line, " ");
			cv::Mat depth_image(cv::Size(180, 180), CV_16UC1);
			unsigned short *p;
			for (int ii = 0; ii < 180; ii++)
			{
				p = depth_image.ptr<uint16_t >(ii);
				for (int j = 0; j < 180; ++j) {
					p[j] = BF.str2int(raw_data.at(ii * 180 + j));
				}
			}
			cv::transpose(depth_image, depth_image);
			nose_tip_value = IP.computeNTP(depth_image.clone(), 400, 1400, 10);
			if (nose_tip_value == -1) {
				nose_tip_value = IP.computeNTP(depth_image.clone(), 400, 1400, 30);
				if (nose_tip_value == -1) {
					nose_tip_value = IP.computeNTP(depth_image.clone(), 400, 1450, 30);
				};
			};
			if (nose_tip_value == -1) {
				nose_tip_value = 700;
			}
			cv::waitKey(5);

			cv::Mat cropped_face = IP.crop3DFace(nose_tip_value, depth_image.clone());
			if (cropped_face.empty()) {
				cropped_face = IP.crop3DFace(700, depth_image.clone());
				if (cropped_face.empty()) {
					std::cout << "error: cropped face empty ! ntp = " << nose_tip_value << std::endl;
					continue;
				}
			}
			cropped_face.copyTo(mask(cv::Rect(38, 38, 180, 180)));

			cv::Mat cropped_depth = mask;
			KinectFusion KF_temp;
			KF_temp.Init(cv::Size(256, 256));

			if (KF_temp.Update(cropped_depth)) {
				std::vector<std::vector<float>> points_temp = KF_temp.GetPoints();
				CalcNormal CN_temp;
				if (points_temp.size() < 100) {
					std::cout << "points_temp number = " << points_temp.size() << std::endl;
					continue;
				}
				else {
					CN_temp.SetPoints(points_temp);
				}
				cv::Mat depth_face_temp = CN_temp.GetDepth();
				cv::imwrite(save_path + "depth/" + sp.at(0) + "/" + index + ".bmp", IP.resize(depth_face_temp, cv::Size(128, 128)));
			}
			else {

			}
			if (KF.Update(cropped_depth)) {
				std::vector<std::vector<float>> points = KF.GetPoints();
				CalcNormal CN;
				if (points.size() < 100) {
					std::cout << "points number = " << points.size() << std::endl;
					continue;
				}
				else {
					CN.SetPoints(points);
				}
				cv::Mat depth_face = CN.GetDepth();
				cv::imwrite(save_path + "depth_kinfu/" + sp.at(0) + "/" + index + ".bmp", IP.resize(depth_face, cv::Size(128, 128)));
			}
			else {

			}
		}
		unused--;
		KF.Reset();
	}
	system("pause");
}
int main() {
	process_lock3dface();
	return 0;
}
