
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
void test_lock3dface();
void test_kinfu();
void process_lock3dface();
int main() {
	//test_lock3dface();
	//test_kinfu();
	process_lock3dface();
	return 0;
}
void process_lock3dface() {
	ofstream log_file("log.txt");

	BasicFuncation BF;
	ImageProcess IP;
	KinectFusion KF; KF.Init(cv::Size(256, 256));

	ifstream depth_data("E:/Dataset/lock3dface/DATA/kinect_all.dat");
	ifstream depth_label("E:/Dataset/lock3dface/DATA/kinect_all.name");

	string line;
	string label;
	string save_path = "E:/Dataset/Lock3DFace_/";
	cv::Mat mask(cv::Size(256, 256), CV_16UC1);
	int count = 0;
	int unused = 0;// 575;
	int control = 0;
	while (getline(depth_label, label)) {
		
		std::cout << count++<<"  "<<label << std::endl;
		std::vector<string> sp = BF.split(label,",");
		log_file << count-1 << "  " << sp.at(0) << std::endl;
			if (_access((save_path + "depth/" + sp.at(0)).data(), 0) == -1) {
				_mkdir((save_path + "depth/" + sp.at(0).data()).data());
			}
			if (_access((save_path + "depth_kinfu/" + sp.at(0)).data(), 0) == -1) {
				_mkdir((save_path + "depth_kinfu/" + sp.at(0).data()).data());
			}
			if (_access((save_path + "normal/" + sp.at(0)).data(), 0) == -1) {
				_mkdir((save_path + "normal/" + sp.at(0).data()).data());
			}
			if (_access((save_path + "normal_kinfu/" + sp.at(0)).data(), 0) == -1) {
				_mkdir((save_path + "normal_kinfu/" + sp.at(0).data()).data());
			}
		for (int i = 0; i < BF.str2int(sp.at(1)); i++) {
			if (i % 10 == 0) KF.Reset();
			char index[2];		sprintf(index, "%02d", i);
			getline(depth_data, line);
			if (unused > 0) continue;
			if (control-- > 0) continue;
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
			cv::Mat segment_face = IP.segmentDepthFace(depth_image.clone());
			segment_face.copyTo(mask(cv::Rect(38, 38, 180, 180)));

			cv::Mat cropped_depth = mask;
			KinectFusion KF_temp;
			KF_temp.Init(cv::Size(256, 256));
			if(!KF_temp.Update(cropped_depth)) continue;
			if(!KF.Update(cropped_depth)) continue;

			std::vector<std::vector<float>> points = KF.GetPoints();
			std::vector<std::vector<float>> points_temp = KF_temp.GetPoints();

			CalcNormal CN_temp;
			CalcNormal CN;

			if (points.size() < 100) {				continue;			}
			else {				CN.SetPoints(points);			}

			if (points_temp.size() < 100) {				continue;			}
			else {				CN_temp.SetPoints(points_temp);			}

			std::future<void> t_depth = async(std::launch::async, [&]() {
				cv::Mat depth_face_temp = CN_temp.GetDepth();
				cv::imwrite(save_path + "depth/" + sp.at(0) + "/" + index + ".jpg", IP.resize(depth_face_temp, cv::Size(128, 128)));
			});
			std::future<void> t_normal = async(std::launch::async, [&]() {
				cv::Mat normal_face_temp = CN_temp.GetNormal();
				cv::imwrite(save_path + "normal/" + sp.at(0) + "/" + index + ".jpg", IP.resize(normal_face_temp, cv::Size(128, 128)));
			});
			std::future<void> t_depth_kinfu = async(std::launch::async, [&]() {
				cv::Mat depth_face = CN.GetDepth();
				cv::imwrite(save_path + "depth_kinfu/" + sp.at(0) + "/" + index + ".jpg", IP.resize(depth_face, cv::Size(128, 128)));
			});
			std::future<void> t_normal_kinfu = async(std::launch::async, [&]() {
				cv::Mat normal_face = CN.GetNormal();
				cv::imwrite(save_path + "normal_kinfu/" + sp.at(0) + "/" + index + ".jpg", IP.resize(normal_face, cv::Size(128, 128)));
			});

			t_depth.wait();
			t_normal.wait();
			t_depth_kinfu.wait();
			t_normal_kinfu.wait();
		}
		unused--;
		KF.Reset();
	}
}
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
		cv::Mat cropped_depth = IP.cropDepthFace(depth(roi));
		depth_map.push_back(cropped_depth.clone());
	}
	CalcNormal CN;
	KinectFusion KF; KF.Init(cv::Size(256, 256));
	for (int i = 0; i < depth_map.size(); i++) {
		char index[2];		sprintf(index, "%02d", i);

		KinectFusion KF_temp;	KF_temp.Init(cv::Size(256, 256));
		KF_temp.Update(depth_map.at(i));
		std::vector<std::vector<float>> points_temp = KF_temp.GetPoints();
		CalcNormal CN_temp;		CN.SetPoints(points_temp);
		cv::Mat depth_face_temp = CN.GetDepth();		cv::transpose(depth_face_temp, depth_face_temp);
		cv::imwrite("result/depth/" + sp.at(0) + "_" + index + ".jpg", depth_face_temp);

		cv::Mat normal_face_temp = CN.GetNormal();	cv::transpose(normal_face_temp, normal_face_temp);
		cv::imwrite("result/normal/" + sp.at(0) + "_" + index + ".jpg", normal_face_temp);

		KF.Update(depth_map.at(i));
		std::vector<std::vector<float>> points = KF.GetPoints();
		CN.SetPoints(points);
		cv::Mat depth_face = CN.GetDepth();
		cv::transpose(depth_face, depth_face);
		cv::imwrite("result/kinfu_depth/" + sp.at(0) + "_" + index + ".jpg", depth_face);

		cv::Mat normal_face = CN.GetNormal();
		cv::transpose(normal_face, normal_face);
		cv::imwrite("result/kinfu_normal/" + sp.at(0) + "_" + index + ".jpg", normal_face);

		//cv::imshow("depth", depth_face);
		//cv::imshow("normal", normal_face);
		//cv::imshow("render",KF.GetRender());
		//cv::waitKey(33);
		//KF.Reset();
	}
	//KinectFusion KF;	KF.Init(cv::Size(256, 256));
	//for (int i = 0; i < depth_map.size(); i++) {
	//	KF.Update(depth_map.at(i));

	//	cv::imshow("render",KF.GetRender());
	//	cv::waitKey(33);
	//	//KF.Reset();
	//}
}
#ifdef _WIN32
void test_lock3dface() {
	BasicFuncation BF;
	FeatureExtractor FE;
	FaceRecognition FR;
	FE.LoadModel("models/normal/msff_net", "0000", "s5_global_conv_output");

	string gallery_path = "data\\test_fold=1\\normal\\NUO";
	string probe_path = "data\\test_fold=1\\normal\\NUT";

	vector<string> gallery_folders = BF.listDir(gallery_path);
	for (int g = 0; g < gallery_folders.size(); g++) {
		vector<string> image_list = BF.listFile(gallery_path + "\\" + gallery_folders.at(g), "jpg");
		for (int f = 0; f < image_list.size(); f++) {
			cout << gallery_path + "\\" + gallery_folders.at(g) + "\\" + image_list.at(f) << endl;
			vector<double> feature = FE.Extract(gallery_path + "\\" + gallery_folders.at(g) + "\\" + image_list.at(f));
			int label = g;
			FR.AddGallery(feature, label);
		}
	}
	int right_num = 0;
	int image_num = 0;
	vector<string> probe_folders = BF.listDir(probe_path);
	for (int p = 0; p < probe_folders.size(); p++) {
		vector<string> image_list = BF.listFile(probe_path + "\\" + probe_folders.at(p), "jpg");
		for (int f = 0; f < image_list.size(); f++) {
			vector<double> feature = FE.Extract(probe_path + "\\" + probe_folders.at(p) + "\\" + image_list.at(f));
			int label = p;
			std::pair<int, double> predict = FR.RecProbe(feature);
			if (label == predict.first) {
				right_num++;
			}
			image_num++;
			cout << "R: " << predict.first << "," << predict.second << "L:"<< label <<"  File:" << probe_path + "\\" + probe_folders.at(p) + "\\" + image_list.at(f) << endl;
		}
	}
	cout << "FR = " << double(right_num) / image_num << endl;
	system("pause");
}
#else

void test_lock3dface() {

	FeatureExtractor FE;
	FaceRecognition FR;
	FE.LoadModel("models/normal/msff_net", "0000", "s5_global_conv_output");

	BasicFuncation BF;
	string data_dir = "data/test_fold=1/";
	std::string gallery_path = "data/test_fold=1/NUO.txt";
	std::string gallery_label = "data/test_fold=1/NUO_label.txt";
	std::string probe_path = "data/test_fold=1/NUT.txt";
	std::string probe_label = "data/test_fold=1/NUT_label.txt";
	ifstream fg(gallery_path);
	ifstream fgl(gallery_label);
	ifstream fp(probe_path);
	ifstream fpl(probe_label);

	std::vector<string> gallery_image_path, probe_image_path;
	std::vector<int> gallery_image_label, probe_image_label;

	string image_path, image_label;
	while (getline(fg, image_path) && getline(fgl, image_label)) {
		if (image_path.size() == 0 || image_label.size() == 0) {
			break;
		}
		vector<double> feature = FE.Extract(data_dir + image_path);
		FR.AddGallery(feature, BF.str2int(image_label));
	}

	int right_num = 0;
	int image_num = 0;
	while (getline(fp, image_path) && getline(fpl, image_label)) {
		if (image_path.size() == 0 || image_label.size() == 0) {
			break;
		}
		vector<double> feature = FE.Extract(data_dir + image_path);
		std::pair<int, double> predict = FR.RecProbe(feature);
		if (BF.str2int(image_label) == predict.first) {
			right_num++;
		}
		image_num++;
		cout << "R: " << predict.first << "," << predict.second << "L:" << image_label << "  File:" << image_path << endl;
	}
	std::cout << "FR = " << double(right_num) / image_num << endl;
	system("pause");
}
#endif
