#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//�ο����ģ��������ֹ�դͶӰ�ṹ�����ά�ؽ������о�--����

/*
	ͶӰ��Ͷ�������ͼ������δ���ǵ�����������ӳ�
	������������ڵ���ϵʹ�ò�������δ��ͶӰ��Ͷ�䵽������������ɼ�������Ӱ��ͼ��
	--��Ҫ��ɨ�����Ч�������ɸѡ:
	1 ����Ͷ��N����դ���ƵĹ�դģʽ����λҪ��֤ 0<=theta<=2��N
	2 ������λͼ��ĳһ�㣬���Ӧ��ÿ����λģʽ�ĻҶ�ֵӦ���нϴ�ı仯�������϶�Ϊ����������Ӱ
*/

Mat findShadowMask(const vector<Mat>& phase, int thresh = 10)
{
	assert(phase.size() != 0);

	Mat mask = Mat::zeros(phase[0].size(), CV_8U);
	const int w = phase[0].cols;
	const int h = phase[0].rows;

	for (size_t i = 0; i < h; i++)
	{
		const uchar* p0 = phase[0].ptr<uchar>(i);
		const uchar* p1 = phase[1].ptr<uchar>(i);
		const uchar* p2 = phase[2].ptr<uchar>(i);
		const uchar* p3 = phase[3].ptr<uchar>(i);
		uchar* p = mask.ptr<uchar>(i);
		for (size_t j = 0; j < w; j++)
		{
			if (abs(p0[j] - p2[j]) < thresh && abs(p1[j] - p3[j]) < thresh) 
				p[j] = 255;
			continue;
		}
	}

	return mask;
}

Mat findPhaseMask()
{
	return Mat();
}

void main()
{
	const int imageNum = 4;
	vector<Mat> image(imageNum);
	for (size_t i = 0; i < imageNum; i++)
	{
		string path = "../����������+��Ƶ�������/test/0 (" + to_string(i + 2) + ").bmp";
		image[i] = imread(path, IMREAD_GRAYSCALE);
	}
	Mat mask = findShadowMask(image, 8);
	bitwise_not(mask, mask);
	Mat absphase = imread("./multifrq.jpg", IMREAD_GRAYSCALE);
	Mat dst;
	absphase.copyTo(dst, mask);
	imwrite("./frq.jpg", dst);

	return;
}