#include "ImageProcess.h"



ImageProcess::ImageProcess()
{
	depth = cv::Mat::zeros(424, 512, CV_16UC1);

}


ImageProcess::~ImageProcess()
{
}
bool ImageProcess::openDepthVideo(string filename) {
	file_depth.open(filename, std::ios::in | std::ios::binary);
	if (!file_depth) {
		std::cerr << "open file error!" << std::endl;
		return false;
	}
	return true;
}
bool ImageProcess::openInfraredVideo(string filename) {
	file_infrared.open(filename, std::ios::in | std::ios::binary);
	if (!file_infrared) {
		std::cout << "open file error!" << std::endl;
		return false;
	}
	return true;
}
bool ImageProcess::openColorVideo(string filename) {
	file_color.open(filename, std::ios::in | std::ios::binary);
	if (!file_color) {
		std::cout << "open file error!" << std::endl;
		return false;
	}
	return true;
}
bool ImageProcess::readDepthImage() {
	if (file_depth.eof()) {
		file_depth.close();
		return false;
	}
	else {
		char *ptr = (char *)depth.data;
		file_depth.read(ptr, 424 * 512 * 2);
		unsigned short *p = (unsigned short *)ptr;
		return true;
	}
}
bool ImageProcess::readInfraredImage() {
	if (file_infrared.eof()) {
		file_infrared.close();
		return false;
	}
	else {
		char *ptr = (char *)infrared.data;
		file_infrared.read(ptr, 424 * 512 * 2);
		unsigned short *p = (unsigned short *)ptr;
		return true;
	}
}
bool ImageProcess::readColorImage() {
	if (file_color.eof()) {
		file_color.close();
		return false;
	}
	else {
		char *pBuffer = new char[1920 * 1080 * 4];
		uchar * ptr = (uchar *)color.data;
		file_color.read(pBuffer, 1920 * 1080 * 4);
		for (int j = 0; j < 1920 * 1080; j++)
		{
			//transfer Kinect BGR-none to RGB 
			ptr[3 * j] = pBuffer[4 * j];
			ptr[3 * j + 1] = pBuffer[4 * j + 1];
			ptr[3 * j + 2] = pBuffer[4 * j + 2];
		}
		delete pBuffer;
		return true;
	}
}

cv::Mat ImageProcess::getDepthImage() {
	return depth;
}
cv::Mat ImageProcess::getInfraredImage() {
	return infrared;
}
cv::Mat ImageProcess::getColorImage() {
	return color;
}

cv::Mat ImageProcess::normalizeInfrared(cv::Mat image) {
	cv::Mat new_infrared = cv::Mat::zeros(cv::Size(image.cols, image.rows), CV_8UC1);
	int nRows = image.rows;
	int nCols = image.cols;
	unsigned short *p;
	unsigned char *q;
	int min_value = 9999;
	int max_value = -9999;
	vector<vector<int>> pixels;
	for (int i = 0; i < nRows; i++)
	{
		vector<int> line;
		p = image.ptr<uint16_t >(i);//获取每行首地址
		for (int j = 0; j < nCols; ++j)
		{
			if (min_value >= p[j]) {
				min_value = p[j];
			}
			if (max_value <= p[j]) {
				max_value = p[j];
			}
			line.push_back(p[j]);
		}
		pixels.push_back(line);
	}
	for (int i = 0; i < pixels.size(); i++)
	{
		q = new_infrared.ptr< uchar>(i);//获取每行首地址	
		for (int j = 0; j < pixels.at(i).size(); j++)
		{
			int pixel = (pixels.at(i).at(j) - min_value) / ((max_value - min_value) / 255);
			q[j] = pixel;
		}
	}
	return new_infrared;
}
cv::Mat ImageProcess::cropDepthFace(cv::Mat depth_face) {
	cv::Point ntp(depth_face.rows/2,depth_face.cols/2);
	std::vector<std::vector<int>> face_points;
	int nRows = depth_face.rows;
	int nCols = depth_face.cols;
	vector<vector<int>> pixels;
	unsigned short *p;
	//首先计算鼻尖点的估计值
	std::vector<int> ntp_area;
	for (int i = ntp.x-10; i < ntp.x+10; i++)
	{
		p = depth_face.ptr<uint16_t >(i);//获取每行首地址
		for (int j = ntp.y-10; j < ntp.y+10; ++j)
		{
			ntp_area.push_back(p[j]);
		}
	}
	sort(ntp_area.begin(), ntp_area.end());
	int ntp_value = 0;
	int __sum = 0;
	for (int i = 300; i < 400; i++) {
		__sum += ntp_area.at(i);
	}
	ntp_value = __sum / 100;

	//阈值分割法



	//然后裁剪人脸
	int __min = 9999;
	int __max = 0;
	int count = 0;//计算有效点数
	for (int i = 0; i < nRows; i++)
	{
		vector<int> line;
		p = depth_face.ptr<uint16_t >(i);//获取每行首地址
		for (int j = 0; j < nCols; ++j)
		{
			if ((ntp.x-i)*(ntp.x - i) + (ntp.y - j)*(ntp.y - j) + (p[j] - ntp_value)*(p[j]-ntp_value) > 60 * 60) {
				p[j] = 0;
			}
			else {
				if (__min >= p[j]) __min = p[j];
				if (__max <= p[j]) __max = p[j];
				p[j] = 2*ntp_value-p[j];
				count++;
			}
		}
	}
	if(count>1000)	return depth_face;
	else return cv::Mat();
}
cv::Mat ImageProcess::resize(cv::Mat inputImage, cv::Size size) {
	int s;
	if (inputImage.rows > inputImage.cols) {
		s = inputImage.rows;
	}
	else {
		s = inputImage.cols;
	}
	cv::Mat image_temp_ep(s, s, CV_8UC3, cv::Scalar(0, 0, 0));
	if (inputImage.channels() == 1) {
		cvtColor(image_temp_ep, image_temp_ep, cv::COLOR_BGR2GRAY);
	}
	cv::Mat image_temp_ep_roi = image_temp_ep(cv::Rect((s - inputImage.cols) / 2, (s - inputImage.rows) / 2, inputImage.cols, inputImage.rows));
	cv::Mat dstNormImg;
	addWeighted(image_temp_ep_roi, 0., inputImage, 1.0, 0., image_temp_ep_roi);
	cv::resize(image_temp_ep, dstNormImg, size, 0, 0, 1);    //大小归一化
	return dstNormImg;
}
cv::Mat ImageProcess::segmentDepthFace(cv::Mat depth_face) {
	cv::Mat cvt;
	cv::Mat dst;
	cv::Mat gray;

	cv::convertScaleAbs(depth_face, cvt, 0.25*256. / 1000);
	cvt.convertTo(gray, CV_8UC1);

	cv::threshold(gray,dst,0,255,cv::THRESH_OTSU);
	cv::bitwise_not(dst,dst);

	uint16_t *f;
	uint8_t *b;
	for (int i = 0; i < depth_face.rows; i++) {
		f = depth_face.ptr<uint16_t >(i);//获取每行首地址
		b = dst.ptr<uint8_t>(i);
		for (int j = 0; j < depth_face.cols; j++) {	if (b[j] == 0) {f[j] = 0; }	}
	}
	return depth_face;
}