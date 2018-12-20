#include "BasicFuncation.h"



BasicFuncation::BasicFuncation()
{
}


BasicFuncation::~BasicFuncation()
{
}
vector<string> BasicFuncation::listDir(string path) {
	vector<string> dirpath;
	//文件句柄    
	intptr_t hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*").c_str(), &fileinfo)) != -1){
		do{
			if ((fileinfo.attrib &  _A_SUBDIR)){
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0){
					dirpath.push_back(p.assign(fileinfo.name));
				}
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return dirpath;
}
vector<string> BasicFuncation::listFile(string path, string format) {
	vector<string> files;
	//文件句柄    
	intptr_t hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*." + format).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
					GetAllFormatFiles(p.assign(fileinfo.name), files, format);
				}
			}
			else
			{
				files.push_back(p.assign(fileinfo.name));  //将文件路径保存，也可以只保存文件名:    p.assign(path).append("\\").append(fileinfo.name)  
			}
		} while (_findnext(hFile, &fileinfo) == 0);
		_findclose(hFile);
	}
	return files;
}
//获取特定格式的文件名 
void BasicFuncation::GetAllFormatFiles(string path, vector<string>& files, string format){
	files.clear();
	//文件句柄    
	intptr_t hFile = 0;
	//文件信息    
	struct _finddata_t fileinfo;
	string p;
	if ((hFile = _findfirst(p.assign(path).append("/*." + format).c_str(), &fileinfo)) != -1)
	{
		do
		{
			if ((fileinfo.attrib &  _A_SUBDIR))
			{
				if (strcmp(fileinfo.name, ".") != 0 && strcmp(fileinfo.name, "..") != 0)
				{
					//files.push_back(p.assign(path).append("\\").append(fileinfo.name) );  
					GetAllFormatFiles(p.assign(path).append("/").append(fileinfo.name), files, format);
				}
			}
			else
			{
				files.push_back(p.assign(fileinfo.name));  //将文件路径保存，也可以只保存文件名:    p.assign(path).append("\\").append(fileinfo.name)  
			}
		} while (_findnext(hFile, &fileinfo) == 0);

		_findclose(hFile);
	}
}
int BasicFuncation::str2int(string str) {
	stringstream ss;
	int num;
	ss << str;
	ss >> num;
	return num;
}
vector< string> BasicFuncation::split(string str, string pattern)
{
	vector<string> ret;
	if (pattern.empty()) return ret;
	size_t start = 0, index = str.find_first_of(pattern, 0);
	while (index != str.npos)
	{
		if (start != index)
			ret.push_back(str.substr(start, index - start));
		start = index + 1;
		index = str.find_first_of(pattern, start);
	}
	if (!str.substr(start).empty())
		ret.push_back(str.substr(start));
	return ret;
}
