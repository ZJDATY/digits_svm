#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main2()
{
	char ad[128] = { 0 };
	int  filename = 0, filenum = 0;
	Mat img = imread("digits.png");
	Mat gray;
	cvtColor(img, gray, COLOR_BGR2GRAY);
	int b = 20;
	int m = gray.rows / b;   //ԭͼΪ1000*2000
	int n = gray.cols / b;   //�ü�Ϊ5000��20*20��Сͼ��

	for (int i = 0; i < m; i++)
	{
		int offsetRow = i * b;  //���ϵ�ƫ����
		if (i % 5 == 0 && i != 0)
		{
			filename++;
			filenum = 0;
		}
		for (int j = 0; j < n; j++)
		{
			int offsetCol = j * b; //���ϵ�ƫ����
			sprintf_s(ad, "data\\%d\\%d.jpg", filename, filenum++);
			//��ȡ20*20��С��
			Mat tmp;
			gray(Range(offsetRow, offsetRow + b), Range(offsetCol, offsetCol + b)).copyTo(tmp);
			imwrite(ad, tmp);
		}
	}
	return 0;
}