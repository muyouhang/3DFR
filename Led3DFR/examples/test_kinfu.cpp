
#include "BasicFuncation.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"
#include "ImageProcess.h"
#include "CalcNormal.h"
#include "KinectFusion.h"
#include "FaceLandmark.h"
#include <fstream>
void test_lock3dface();
void test_kinfu();
int main() {
	test_lock3dface();
	test_kinfu();
	return 0;
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

		cv::imshow("depth", depth_face);
		cv::imshow("normal", normal_face);
		cv::imshow("render",KF.GetRender());
		cv::waitKey(33);
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
