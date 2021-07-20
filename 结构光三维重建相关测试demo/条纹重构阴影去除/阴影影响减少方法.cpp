#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//参考论文：基于数字光栅投影结构光的三维重建技术研究--王曦

/*
	投影仪投射的条纹图案可能未覆盖到相机的整个视场
	物体表面的起伏遮挡关系使得部分区域未被投影仪投射到，进而由相机采集包含阴影的图像
	--需要对扫描的有效区域进行筛选:
	1 对于投射N个光栅条纹的光栅模式，相位要保证 0<=theta<=2πN
	2 对于相位图上某一点，其对应在每个相位模式的灰度值应当有较大的变化；否则被认定为背景或者阴影
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
		string path = "../相移轮廓术+多频外差解包裹/test/0 (" + to_string(i + 2) + ").bmp";
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