#include "FaceLandmark.h"
FaceLandmark::FaceLandmark()
{
	faceDetector.load("lbpcascade_frontalface.xml");
	facemark = cv::face::FacemarkLBF::create();
	facemark->loadModel("lbfmodel.yaml");
}
FaceLandmark::~FaceLandmark()
{
}
std::pair<cv::Rect, cv::Point2f> FaceLandmark::detectFaceAndNTP(cv::Mat gray_image) {
	std::vector<cv::Rect> faces;
	std::vector< std::vector<cv::Point2f> > landmarks;
	faceDetector.detectMultiScale(gray_image, faces);
	bool success = facemark->fit(gray_image, faces, landmarks);
	cv::Rect max_face(0,0,0,0);
	std::vector<cv::Point2f> max_face_landmark;
	for (int i = 0; i < faces.size(); i++) {
		if (faces.at(i).width * faces.at(i).height > max_face.width * max_face.height) {
			max_face = faces.at(i);
			max_face_landmark = landmarks.at(i);
		}
	}
	return std::make_pair(max_face,max_face_landmark.at(30));
}
