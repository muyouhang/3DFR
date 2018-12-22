#include "FaceRecognition.h"



FaceRecognition::FaceRecognition()
{
}
FaceRecognition::~FaceRecognition()
{
}
double FaceRecognition::dotProduct(const std::vector<double> v1, const std::vector<double> v2){
     assert(v1.size() == v2.size());
     double ret = 0.0;
     for (std::vector<double>::size_type i = 0; i != v1.size(); ++i){
         ret += v1[i] * v2[i];
     }
     return ret;
 }
double FaceRecognition::module(const std::vector<double> v){
     double ret = 0.0;
     for (std::vector<double>::size_type i = 0; i != v.size(); ++i){
         ret += v[i] * v[i];
     }
     return sqrt(ret);
}
// º–Ω«”‡œ“
double FaceRecognition::cosine(const std::vector<double> v1, const std::vector<double> v2){
    assert(v1.size() == v2.size());
    return dotProduct(v1, v2) / (module(v1) * module(v2));
}
int FaceRecognition::AddGallery(std::vector<double> feature, int label) {
	this->gallerys.push_back(std::make_pair(feature,label));
	return this->gallerys.size();
}
std::pair<int, double> FaceRecognition::RecProbe(std::vector<double> feature) {
	std::vector<std::pair<int, double>> prob;
	int rec_label = -1;
	float rec_p = 0;
	for (int i = 0; i < gallerys.size(); i++) {
		float p= this->cosine(gallerys.at(i).first,feature);
		//prob.push_back(std::make_pair(gallerys.at(i).second,p));
		//std::cout << p << std::endl;
		if (p > rec_p) {
			rec_p = p;
			rec_label = gallerys.at(i).second;
		}
	}
	return std::make_pair(rec_label,rec_p);
}