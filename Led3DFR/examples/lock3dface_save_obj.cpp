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
#include <chrono>s
#include <pcl/io/obj_io.h>
void save2OBJ(string OBJFileName, std::vector<std::vector<float>> points) {
	pcl::PointCloud<pcl::PointXYZ> points_cloud;

	for (int p = 0; p < points.size(); p++) {
		pcl::PointXYZ point;
		point.x = points.at(p).at(0);// *500;
		point.y = points.at(p).at(1);// *500;
		point.z = points.at(p).at(2);// *500;
		points_cloud.points.push_back(point);
	}
	pcl::PolygonMesh mesh;
	pcl::toPCLPointCloud2(points_cloud, mesh.cloud);
	pcl::io::saveOBJFile(OBJFileName, mesh);
}
void save2OBJ(string OBJFileName, std::vector<std::vector<int>> points) {
	pcl::PointCloud<pcl::PointXYZ> points_cloud;

	for (int p = 0; p < points.size(); p++) {
		pcl::PointXYZ point;
		point.x = points.at(p).at(0);// *500;
		point.y = points.at(p).at(1);// *500;
		point.z = points.at(p).at(2);// *500;
		points_cloud.points.push_back(point);
	}
	pcl::PolygonMesh mesh;
	pcl::toPCLPointCloud2(points_cloud, mesh.cloud);
	pcl::io::saveOBJFile(OBJFileName, mesh);
}

void lock3dface_save_obj(string data_name) {
	ofstream log_file("log.txt");

	BasicFuncation BF;
	ImageProcess IP;

	ifstream depth_data("E:/Dataset/lock3dface/DATA/"+ data_name+"_all.dat");
	ifstream depth_label("E:/Dataset/lock3dface/DATA/" + data_name + "_all.name");
	ifstream error_data("E:/Dataset/Lock3DFace_face/code/obj_problem_video.txt");
	float z_min = 400;
	float z_max = 1000;
	int base_ntp = 700;
	int search_ratio = 5;

	std::vector<string> error_videos;
	while (!error_data.eof()) {
		string line;
		getline(error_data, line);
		std::vector<string> sp = BF.split(line, " ");
		if (sp.size()>0)	error_videos.push_back(sp.at(0));
	}

	string line;
	string label;
	string save_path = "E:/Dataset/Lock3DFace_obj_v2/";
	int count = 0;
	int unused = 0;
	//cout << "unused=";
	//cin >> unused;
	int control = 0;
	while (getline(depth_label, label)) {

		std::vector<string> sp = BF.split(label, ",");
		std::cout << count++ << "  " << sp.at(0) << std::endl;

		if (_access((save_path  + sp.at(0)).data(), 0) == -1) {
			_mkdir((save_path + sp.at(0).data()).data());
		}
		

		std::vector<cv::Mat> depth_image_list;
		for (int i = 0; i < BF.str2int(sp.at(1)); i++) {
			getline(depth_data, line);
			if (unused > 0) continue;
			if (std::count(error_videos.begin(), error_videos.end(), sp[0]) == 0) continue;
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
			depth_image_list.push_back(depth_image);
		}

		#pragma omp parallel for
		for (int i = 0; i < depth_image_list.size(); i++) {
			cv::Mat depth_image = depth_image_list.at(i);
			std::pair<cv::Mat,float> td= IP.computeAdaptiveThreshold(depth_image);
			depth_image = td.first;
			z_max = td.second;
			int nose_tip_value = IP.computeNTP(depth_image.clone(), z_min, z_max, search_ratio);
			if (nose_tip_value == -1) {
				int search_ratio_temp = search_ratio;
				while (nose_tip_value == -1 && search_ratio_temp<=30) {
					nose_tip_value = IP.computeNTP(depth_image.clone(), z_min, z_max, ++search_ratio_temp);
				}
				if (nose_tip_value == -1) {
					nose_tip_value = base_ntp;
				}
			}

			cv::Mat cropped_face = IP.crop3DFace(nose_tip_value, depth_image.clone());
			if (cropped_face.empty()) {
				cropped_face = IP.crop3DFace(base_ntp, depth_image.clone());
				if (cropped_face.empty()) {
					std::cout << "error: cropped face empty ! ntp = " << nose_tip_value << std::endl;
					continue;
				}
			}
			depth_image_list.at(i) = cropped_face;
		}


		#pragma omp parallel for
		for (int i = 0; i < depth_image_list.size(); i++) {
			char index[2];		sprintf(index, "%02d", i);

			cv::Mat cropped_face = depth_image_list.at(i);
			
			std::vector<std::vector<int>> points_temp = IP.transDepth2Points(cropped_face);
			if (points_temp.size() < 100) {
				std::cout << "points_temp number = " << points_temp.size() << std::endl;
				continue;
			}
			else {
				save2OBJ(save_path + "/" + sp.at(0) + "/" + index + ".obj", points_temp);
			}
			/*
			
			
			cv::Mat mask(cv::Size(256, 256), CV_16UC1);
			cropped_face.copyTo(mask(cv::Rect(38, 38, 180, 180)));

			cv::Mat cropped_depth = mask;
			KinectFusion KF_temp;
			KF_temp.Init(cv::Size(256, 256));

			if (KF_temp.Update(cropped_depth)) {
				std::vector<std::vector<float>> points_temp = KF_temp.GetPoints();
				if (points_temp.size() < 100) {
					std::cout << "points_temp number = " << points_temp.size() << std::endl;
					continue;
				}
				else {
					save2OBJ(save_path + "/" + sp.at(0) + "/" + index + ".obj", points_temp);
				}
			}
			*/
		}
		unused--;
	}
}
int main() {
	lock3dface_save_obj("kinect");
	lock3dface_save_obj("kinect2");
	std::system("pause");

	return 0;
}

