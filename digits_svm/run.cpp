#include <stdio.h>  
#include <time.h>  
#include <opencv2/opencv.hpp>   
#include <iostream> 
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/ml/ml.hpp>  
#include <io.h>
#include <vector>  
#include <string> 
#include <filesystem>  

using namespace std;
using namespace cv;
namespace fs = std::filesystem;

void getFiles(std::string& path, vector<string>& files) {
	for (const auto& entry : fs::directory_iterator(path)) {
		// 只检查是否为普通文件（不是目录）  
		if (entry.is_regular_file()) {
			files.push_back(entry.path().string());
		}
	}
}

int main()
{
	int result = 0;
	string filePath = "data\\test_image\\0";
	vector<string> files;
	getFiles(filePath, files);
	int number = files.size();
	cout << number << endl;
	Ptr<ml::SVM> svm;
	
	string modelpath = "digits_svm.yml";
	vector<float> predictions;
	FileStorage svm_fs(modelpath, FileStorage::READ);
	if (svm_fs.isOpened())
	{
		svm->load(modelpath.c_str());
	}
	for (int i = 0; i < number; i++)
	{
		Mat inMat = imread(files[i].c_str());
		Mat p = inMat.reshape(1, 1);
		p.convertTo(p, CV_32FC1);
		int response = (int)svm->predict(p);
		if (response == 0)
		{
			result++;
		}
	}
	cout << result << endl;
	getchar();
	return  0;
}
