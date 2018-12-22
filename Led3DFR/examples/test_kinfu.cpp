#include "BasicFuncation.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"
#include "ImageProcess.h"
//#include "CalcNormal.h"
#include "KinectFusion.h"
#include "FaceLandmark.h"
void test_lock3dface();
int main() {

	ImageProcess IP;
	KinectFusion KF;
	BasicFuncation BF;
	FaceLandmark FL;
	KF.Init(cv::Size(256,256));

	std::vector<string> sp;
	std::vector<cv::Mat> depth_map;

	string root_dir = "data/";
	string depth_raw_name = "001_Kinect_FE_1DEPTH.RAW";
	string infrared_raw_name = "001_Kinect_FE_1INFRARED.RAW";
	
	IP.openDepthVideo(root_dir+ depth_raw_name);
	IP.openInfraredVideo(root_dir + infrared_raw_name);

	sp = BF.split(depth_raw_name, ".");

	bool flag = true;
	cv::Rect roi;
	int count = 0;

	while (IP.readDepthImage() && IP.readInfraredImage()) {

		if (count++ >58) {		break;		}

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
	//CalcNormal CN;
	for (int i = 0; i < depth_map.size(); i++) {	
		char index[2];		sprintf(index, "%02d", i);

		KF.Update(depth_map.at(i));		
		//std::vector<std::vector<float>> points = KF.GetPoints();		
		//CN.SetPoints(points);
		////CN.ShowPoints();
		//cv::Mat depth_face = CN.GetDepth();
		//cv::transpose(depth_face, depth_face);
		//cv::imwrite("result/depth/" + sp.at(0) + "_" + index + ".jpg", depth_face);

		//cv::Mat normal_face = CN.GetNormal();
		//cv::transpose(normal_face, normal_face);
		//cv::imwrite("result/normal/" + sp.at(0) + "_" + index + ".jpg", normal_face);

		//cv::imshow("depth",depth_face);
		//cv::imshow("normal", normal_face);
		//cv::waitKey(33);
		cv::imshow("render",KF.GetRender());
		cv::waitKey(33);
		KF.Reset();
	}
	return 0;
}
#ifdef WINDOWS
void test_lock3dface() {
	BasicFuncation BF;
	FeatureExtractor FE;
	FaceRecognition FR;
	FE.LoadModel("models/normal/msff_net", "0000", "s5_global_conv_output");

	string gallery_path = "E:\\Dataset\\lock3dface\\Kfold\\fold=1\\Normal\\test\\NUO";
	string probe_path = "E:\\Dataset\\lock3dface\\Kfold\\fold=1\\Normal\\test\\NUT";

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
			cout << "R: " << predict.first << "," << predict.second << "  File:" << probe_path + "\\" + probe_folders.at(p) + "\\" + image_list.at(f) << endl;
		}
	}
	cout << "FR = " << double(right_num) / image_num << endl;
	system("pause");
}
#endif