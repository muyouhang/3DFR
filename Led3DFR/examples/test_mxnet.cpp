#include "BasicFuncation.h"
#include "FeatureExtractor.h"
#include "FaceRecognition.h"

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
int main() {
	test_lock3dface();
	return 0;
}