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

void getFiles2(std::string& path, vector<string>& files);
void get_1(Mat& trainingImages, vector<int>& trainingLabels);
void get_0(Mat& trainingImages, vector<int>& trainingLabels);

int main3()
{
	//获取训练数据
	Mat classes;
	Mat trainingData;
	Mat trainingImages;
	vector<int> trainingLabels;
	get_1(trainingImages, trainingLabels);
	get_0(trainingImages, trainingLabels);
	Mat(trainingImages).copyTo(trainingData);
	trainingData.convertTo(trainingData, CV_32FC1);
	Mat(trainingLabels).copyTo(classes);
	//配置SVM训练器参数
	Ptr<ml::SVM> svm;
	svm = ml::SVM::create();
	svm->setGamma(1);
	svm->setC(1);
	svm->setDegree(0);
	svm->setCoef0(0);
	svm->setKernel(ml::SVM::LINEAR);
	svm->setNu(0);
	svm->setP(0);
	svm->setTermCriteria(cv::TermCriteria(TermCriteria::Type::COUNT,1000,0.01));
	svm->setType(ml::SVM::C_SVC);
	svm->train(trainingData, ml::ROW_SAMPLE, classes);
	svm->save("digits_svm.yml");
	cout << "训练好了！！！" << endl;
	getchar();
	return 0;
}

void getFiles2(std::string& path, std::vector<std::string>& files) {
	for (const auto& entry : fs::directory_iterator(path)) {
		  
		if (entry.is_regular_file()) {
			files.push_back(entry.path().string());
		}
	}
}
void get_1(Mat& trainingImages, vector<int>& trainingLabels)
{
	string filePath = "D:\\vcworkspace\\digits_svm\\digits_svm\\data\\train_image\\1";
	vector<string> files;
	getFiles2(filePath, files);
	int number = files.size();
	for (int i = 0; i < number; i++)
	{
		Mat  SrcImage = imread(files[i].c_str());
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(1);
	}
}
void get_0(Mat& trainingImages, vector<int>& trainingLabels)
{
	string filePath = "data\\train_image\\0";
	vector<string> files;
	getFiles2(filePath, files);
	int number = files.size();
	for (int i = 0; i < number; i++)
	{
		Mat  SrcImage = imread(files[i].c_str());
		SrcImage = SrcImage.reshape(1, 1);
		trainingImages.push_back(SrcImage);
		trainingLabels.push_back(0);
	}
}