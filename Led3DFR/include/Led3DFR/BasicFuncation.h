#pragma once
#ifdef _WIN32
#include <io.h>
#endif
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
using namespace std;
class BasicFuncation
{
private:
#ifdef _WIN32
	void GetAllFormatFiles(string path, vector<string>& files, string format);
#endif
public:
	BasicFuncation();
	~BasicFuncation();
#ifdef _WIN32
	vector<string> listDir(string path);
	vector<string> listFile(string path, string format);
#endif
	int str2int(string str);
	vector< string> split(string str, string pattern);
};

