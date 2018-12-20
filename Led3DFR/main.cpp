#include "BasicFuncation.h"
#include "FeatureExtractor.h"
#include "FaceDetect.h"
#include "FaceRecognition.h"
#include "ImageProcess.h"
#include "CalcNormal.h"
#include "KinectFusion.h"

void test_lock3dface();
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud);
int main() {
	FaceDetect FD;
	CalcNormal CN;
	ImageProcess IP;
	KinectFusion KF;
	BasicFuncation BF;
	
	KF.Init(cv::Size(256,256));

	std::vector<string> sp;
	std::vector<cv::Mat> depth_map;

	string root_dir = "D:/Dataset/kinect2_data/";
	string depth_raw_name = "001_Kinect_FE_1DEPTH.RAW";
	string color_raw_name = "001_Kinect_FE_1COLOR.RAW";
	string infrared_raw_name = "001_Kinect_FE_1INFRARED.RAW";
	
	IP.openDepthVideo(root_dir+ depth_raw_name);
	IP.openInfraredVideo(root_dir + infrared_raw_name);
	IP.openColorVideo(root_dir+ color_raw_name);

	sp = BF.split(depth_raw_name, ".");

	bool flag = true;
	cv::Rect roi;
	int count = 0;

	while (IP.readDepthImage() && IP.readInfraredImage() && IP.readColorImage()) {

		if (count++ >58) {		break;		}

		cv::Mat depth = IP.getDepthImage();
		cv::Mat infrared = IP.getInfraredImage();
		cv::Mat color = IP.getColorImage();

		if (flag) {
			cv::GaussianBlur(infrared, infrared, cv::Size(3, 3), 0, 0);
			infrared = IP.normalizeInfrared(infrared);
			std::pair<cv::Rect, cv::Point> infrared_face = FD.detectFaceAndNTP(infrared);
			roi = cv::Rect(infrared_face.second.x - 128, infrared_face.second.y - 128, 256, 256);
			flag = false;
		}

		cv::Mat cropped_depth = IP.cropDepthFace(depth(roi));
		depth_map.push_back(cropped_depth.clone());
	}

	for (int i = 0; i < depth_map.size(); i++) {	
		char index[2];		sprintf(index, "%02d", i);

		KF.Update(depth_map.at(i));		
		std::vector<std::vector<float>> points = KF.GetPoints();		
		CN.SetPoints(points);

		cv::Mat depth_face = CN.GetDepth();
		cv::transpose(depth_face, depth_face);
		cv::imwrite("result/depth/" + sp.at(0) + "_" + index + ".jpg", depth_face);

		cv::Mat normal_face = CN.GetNormal();
		cv::transpose(normal_face, normal_face);
		cv::imwrite("result/normal/" + sp.at(0) + "_" + index + ".jpg", normal_face);

		//cv::imshow("depth",depth_face);
		//cv::imshow("normal", normal_face);
		//cv::waitKey(33);
		KF.Reset();
	}
	//boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer = simpleVis(CN.GetPoints());
	//while (!viewer->wasStopped())
	//{
	//	viewer->spinOnce(100);
	//	boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	//}
	return 0;
}
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
//simpleVis函数实现最基本的点云可视化操作，
boost::shared_ptr<pcl::visualization::PCLVisualizer> simpleVis(pcl::PointCloud<pcl::PointXYZ>::ConstPtr cloud)
{
	// --------------------------------------------
	// -----Open 3D viewer and add point cloud-----
	// --------------------------------------------
	//创建视窗对象并给标题栏设置一个名称“3D Viewer”并将它设置为boost::shared_ptr智能共享指针，这样可以保证指针在程序中全局使用，而不引起内存错误
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	//设置视窗的背景色，可以任意设置RGB的颜色，这里是设置为黑色
	viewer->setBackgroundColor(0, 0, 0);
	/*这是最重要的一行，我们将点云添加到视窗对象中，并定一个唯一的字符串作为ID 号，利用此字符串保证在其他成员中也能
	标志引用该点云，多次调用addPointCloud可以实现多个点云的添加，，每调用一次就会创建一个新的ID号，如果想更新一个
	已经显示的点云，必须先调用removePointCloud（），并提供需要更新的点云ID 号，
	*******************************************************************************************/
	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud");
	//用于改变显示点云的尺寸，可以利用该方法控制点云在视窗中的显示方法，
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
	/*******************************************************************************************************
	查看复杂的点云，经常让人感到没有方向感，为了保持正确的坐标判断，需要显示坐标系统方向，可以通过使用X（红色）
	Y（绿色 ）Z （蓝色）圆柱体代表坐标轴的显示方式来解决，圆柱体的大小可以通过scale参数来控制，本例中scale设置为1.0

	******************************************************************************************************/
	//viewer->addCoordinateSystem(1.0);
	//通过设置照相机参数使得从默认的角度和方向观察点云
	//viewer->initCameraParameters();
	return (viewer);
}