#include "FeatureExtractor.h"

FeatureExtractor::FeatureExtractor()
{
}


FeatureExtractor::~FeatureExtractor()
{
}
NDArray FeatureExtractor::Mat2NDArray(cv::Mat image) {
	std::vector<float> array;

	vector<cv::Mat> channels;
	cv::split(image,channels);//BGR
	cv::Mat new_channels[3] = {channels.at(2),channels.at(1),channels.at(0)};
	cv::Mat merge_image; cv::merge(new_channels,3,merge_image);
	cv::resize(merge_image, image, cv::Size(128, 128));
	for (int c = 0; c < 3; ++c) {
		for (int i = 0; i < 128; ++i) {
			for (int j = 0; j < 128; ++j) {
				array.push_back(static_cast<float>(image.data[(i * 128 + j) * 3 + c]));
			}
		}
	}
	NDArray ret(Shape(1, 3, 128, 128), ctx, true);
	ret.SyncCopyFromCPU(array.data(), 1 * 3 * 128 * 128);
	NDArray::WaitAll();
	return ret;
}
int FeatureExtractor::LoadModel(string net_name, string epoch,string layer_name) {
	GetFeatureSymbol(net_name+"-symbol.json",layer_name);
	LoadParamtes(net_name+"-"+epoch +".params");
	NDArray data = Mat2NDArray(cv::Mat::zeros(cv::Size(128,128),CV_8UC3));
	args_map["data"] = data;
	executor = net.SimpleBind(ctx, args_map, map<string, NDArray>(),
		map<string, OpReqType>(), aux_map);
	return 0;
}
void FeatureExtractor::GetFeatureSymbol(string symbol_name,string layer_name) {
	net = Symbol::Load(symbol_name).GetInternals()[layer_name];
}
void FeatureExtractor::LoadParamtes(string model_name) {
	map<string, NDArray> paramters;
	NDArray::Load(model_name, 0, &paramters);
	for (const auto &k : paramters) {
		if (k.first.substr(0, 4) == "aux:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			aux_map[name] = k.second.Copy(ctx);
		}
		if (k.first.substr(0, 4) == "arg:") {
			auto name = k.first.substr(4, k.first.size() - 4);
			args_map[name] = k.second.Copy(ctx);
		}
	}
	/*WaitAll is need when we copy data between GPU and the main memory*/
	NDArray::WaitAll();
}
std::vector<double> FeatureExtractor::Extract(string image_name) {
	cv::Mat image = cv::imread(image_name);

	if (image.empty()) {
		cout << "File Error : "<<image_name << endl;
	}
	NDArray data = Mat2NDArray(image);
	data.CopyTo(&args_map["data"]);
	//args_map["data"] = data;
	
	executor->Forward(false);
	/*print out the features*/
	auto array = executor->outputs[0].Copy(ctx_cpu);
	NDArray::WaitAll();

	std::vector<double> feature;
	for (int i = 0; i < array.Size(); i++) {
		feature.push_back(array.At(0,i));
		//std::cout << array.At(0, i) << std::endl;
	}
	return feature;
}