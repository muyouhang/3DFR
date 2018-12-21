#pragma once
#include <io.h>  
#include <string>
#include <vector>
#include <sstream>
#include <iostream>
using namespace std;
class BasicFuncation
{
private:
	void GetAllFormatFiles(string path, vector<string>& files, string format);
public:
	BasicFuncation();
	~BasicFuncation();
	vector<string> listDir(string path);
	vector<string> listFile(string path, string format);
	int str2int(string str);
	vector< string> split(string str, string pattern);
};

