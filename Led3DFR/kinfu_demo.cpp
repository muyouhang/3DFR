// This file is part of OpenCV project.
// It is subject to the license terms in the LICENSE file found in the top-level directory
// of this distribution and at http://opencv.org/license.html

// This code is also subject to the license terms in the LICENSE_KinectFusion.md file found in this module's directory

#include <iostream>
#include <fstream>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/rgbd/kinfu.hpp>

using namespace cv;
using namespace cv::kinfu;
using namespace std;

static vector<string> readDepth(std::string fileList)
{
	vector<string> v;

	fstream file(fileList);
	if (!file.is_open())
		throw std::runtime_error("Failed to read depth list");

	std::string dir;
	size_t slashIdx = fileList.rfind('/');
	slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
	dir = fileList.substr(0, slashIdx);

	while (!file.eof())
	{
		std::string s, imgPath;
		std::getline(file, s);
		if (s.empty() || s[0] == '#') continue;
		std::stringstream ss;
		ss << s;
		double thumb;
		ss >> thumb >> imgPath;
		v.push_back(dir + '/' + imgPath);
	}

	return v;
}

struct DepthWriter
{
	DepthWriter(string fileList) :
		file(fileList, ios::out), count(0), dir()
	{
		size_t slashIdx = fileList.rfind('/');
		slashIdx = slashIdx != std::string::npos ? slashIdx : fileList.rfind('\\');
		dir = fileList.substr(0, slashIdx);

		if (!file.is_open())
			throw std::runtime_error("Failed to write depth list");

		file << "# depth maps saved from device" << endl;
		file << "# useless_number filename" << endl;
	}

	void append(InputArray _depth)
	{
		Mat depth = _depth.getMat();
		string depthFname = cv::format("%04d.png", count);
		string fullDepthFname = dir + '/' + depthFname;
		if (!imwrite(fullDepthFname, depth))
			throw std::runtime_error("Failed to write depth to file " + fullDepthFname);
		file << count++ << " " << depthFname << endl;
	}

	fstream file;
	int count;
	string dir;
};

struct DepthSource
{
public:
	DepthSource(int cam) :
		DepthSource("", cam)
	{ }

	DepthSource(String fileListName) :
		DepthSource(fileListName, -1)
	{ }

	DepthSource(String fileListName, int cam) :
		depthFileList(fileListName.empty() ? vector<string>() : readDepth(fileListName)),
		frameIdx(0),
		vc(cam >= 0 ? VideoCapture(VideoCaptureAPIs::CAP_OPENNI2 + cam) : VideoCapture()),
		undistortMap1(),
		undistortMap2(),
		useKinect2Workarounds(true)
	{ }

	UMat getDepth()
	{
		UMat out;

		if (frameIdx < depthFileList.size())
		{
			Mat f = cv::imread(depthFileList[frameIdx++], IMREAD_ANYDEPTH);
			f.copyTo(out);
		}
		else
		{
			return UMat();
		}
		
		if (out.empty())
			throw std::runtime_error("Matrix is empty");
		return out;
	}

	bool empty()
	{
		return depthFileList.empty() && !(vc.isOpened());
	}

	void updateParams(Params& params)
	{
	}

	vector<string> depthFileList;
	size_t frameIdx;
	VideoCapture vc;
	UMat undistortMap1, undistortMap2;
	bool useKinect2Workarounds;
};

int main(int argc, char **argv)
{
	bool coarse = false;
	bool idle = false;
	string recordPath;

	Ptr<DepthSource> ds;
	ds = makePtr<DepthSource>("D:/Led3DFR/Led3DFR/depth.txt");

	if (ds->empty())
	{
		std::cerr << "Failed to open depth source" << std::endl;
		return -1;
	}

	Ptr<DepthWriter> depthWriter;
	if (!recordPath.empty())
		depthWriter = makePtr<DepthWriter>(recordPath);

	Ptr<Params> params;
	Ptr<KinFu> kf;

	params = Params::defaultParams();
	params->frameSize = cv::Size(512, 424);
	params->depthFactor = 1000;
	params->volumeDims = Vec3i::all(512);
	params->voxelSize = 3.f / 512;

	cv::setUseOptimized(true);

	kf = KinFu::create(params);
	UMat rendered;
	UMat points;
	UMat normals;

	int64 prevTime = getTickCount();
	int idx = 0;
	for (UMat frame = ds->getDepth(); !frame.empty(); frame = ds->getDepth())
	{
		if (depthWriter)	depthWriter->append(frame);
		
		UMat cvt8;
		float depthFactor = params->depthFactor;
		convertScaleAbs(frame, cvt8, 0.25*256. / depthFactor);

		imshow("depth", cvt8);
		std::cout << idx ++<< std::endl;
		if (!kf->update(frame))
		{
			kf->reset();
			std::cout << "reset" << std::endl;
		}
		kf->render(rendered);

		int64 newTime = getTickCount();
		putText(rendered, cv::format("FPS: %2d press R to reset, P to pause, Q to quit",
			(int)(getTickFrequency() / (newTime - prevTime))),
			Point(0, rendered.rows - 1), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0, 255, 255));
		prevTime = newTime;

		imshow("render", rendered);

		int c = waitKey(33);
		switch (c)
		{

		case 'r':
			kf->reset();
			break;
		case 'q':
			return 0;
		default:
			break;
		}
	}

	return 0;
}