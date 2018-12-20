#include "FaceDetect.h"
FaceDetect::FaceDetect()
{
	mask = cv::imread("mask.bmp",0);
	filename = "area.txt";
	result.open(filename);
	for (int i = 1; i <= 25; i++) {
		last_blocks.push_back(i);
	}
}
FaceDetect::~FaceDetect()
{
}
std::pair<cv::Rect,cv::Point> FaceDetect::detectFaceAndNTP(cv::Mat image) {
	cv::Mat gray;
	int * pResults = NULL;
	if (image.channels() == 3) {
		cvtColor(image, gray, cv::COLOR_BGR2GRAY);
	}
	else if (image.channels() == 1) {
		gray = image;
	}
	unsigned char * pBuffer = (unsigned char *)malloc(DETECT_BUFFER_SIZE);
	int doLandmark = 1;
	pResults = facedetect_multiview(pBuffer, (unsigned char*)(gray.ptr(0)), gray.cols, gray.rows, (int)gray.step,
		1.1f, 2, 48, 0, doLandmark);
	cv::Mat result_multiview = image.clone();;
	int face_num = 0;
	cv::Rect face;
	cv::Point ntp;
	for (int i = 0; i < (pResults ? *pResults : 0); i++)
	{
		short * p = ((short*)(pResults + 1)) + 142 * i;
		int x = p[0];
		int y = p[1];
		int w = p[2];
		int h = p[3];
		int neighbors = p[4];
		int angle = p[5];

		x = x - w / 3;
		y = y - h / 3;
		w = w + w*2/3 ;
		h = h + h*2/3 ;

		face = cv::Rect(x, y, w, h);
		ntp = cv::Point((int)p[6 + 2 * 30], (int)p[6 + 2 * 30 + 1]);
		//rectangle(result_multiview, cv::Rect(x, y, w, h), cv::Scalar(0, 255, 0), 2);
		//if (doLandmark)
		//{
		//	for (int j = 0; j < 68; j++)
		//		circle(result_multiview, cv::Point((int)p[6 + 2 * j], (int)p[6 + 2 * j + 1]), 1, cv::Scalar(0, 255, 0));
		//}
	}
	/*imshow("face",result_multiview);
	waitKey(0);*/
	return std::make_pair(face,ntp);
}
vector<int> FaceDetect::getPixels(cv::Mat gray_image) {
	vector<int> pixels;
	pixels.push_back(int(gray_image.at<uchar>(0, 0)));
	for (int i = 0; i < gray_image.rows; i++) {
		for (int j = 0; j < gray_image.cols; j++)
		{
			bool flag = true;
			for (int k = 0; k < pixels.size(); k++) {
				if (pixels.at(k) == int(gray_image.at<uchar>(i, j))) {
					flag = false;
					break;
				}
			}
			if (flag) {
				pixels.push_back(int(gray_image.at<uchar>(i, j)));
			}
		}
	}
	return pixels;
}