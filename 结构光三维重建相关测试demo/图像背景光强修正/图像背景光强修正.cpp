#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//参考论文：光栅投影双目视觉形貌测量及影响因素研究_尚明诺（天津大学）
//计算中实际用到的光强部分为调制后的部分
//实际应用中，由于背景光噪声和相机曝光脉冲的影响，无法保证背景光强完全一致
//平均灰度值，保证同频率的四张图背景光强一致

float calMean(const vector<Mat>& srcImage)
{
	const size_t imgNum = srcImage.size();
	float m = .0f;
	for (const Mat& src : srcImage)
	{
		Scalar s = mean(src);
		m += static_cast<float>(s.val[0]);
	}
	return m /= imgNum;
}

void BGLightCorrect1(const string& imgPath, const int imgNum, const Rect& r)
{
	vector<Mat> srcImage(imgNum);
	vector<Mat> rectImage(imgNum);
	vector<Mat> resImage(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		string path = imgPath + "/0 (" + to_string(i + 2) + ").bmp";
		Mat src = imread(path, IMREAD_GRAYSCALE);
		srcImage[i] = src;
		rectImage[i] = src(r);;
	}
	float m_ = calMean(rectImage);
	for (size_t i = 0; i < imgNum; i++)
	{
		Scalar s = mean(rectImage[i]);
		float m = static_cast<float>(s.val[0]);
		float val = m - m_;	
		resImage[i] = srcImage[i] - val;
		Mat t1 = srcImage[i];
		Mat t2 = resImage[i];
		imwrite(imgPath + "/" + to_string(i + 2) + ".bmp", resImage[i]);
	}
	return;
}

/******************************************************************************************/

//自己改进：使用平滑滤波后的结果指代背景光强部分

Scalar calMeanMat(const vector<Mat>& srcImage)
{
	const int imgNum = srcImage.size();
	Mat output = Mat::zeros(srcImage[0].size(), CV_8UC4);
	merge(srcImage, output);
	return mean(output);//求四张图的平均像素值

	//Mat output = Mat::zeros(srcImage[0].size(), CV_32F);//求四张图的平均背景
	//for (size_t i = 0; i < imgNum; i++)
	//{
	//	accumulate(srcImage[i], output);
	//}
	//output /= imgNum;
	//output.convertTo(output, CV_8U);
}

Mat calMeanMat_(const vector<Mat>& srcImage)
{
	const int imgNum = srcImage.size();
	Mat output = Mat::zeros(srcImage[0].size(), CV_32F);//求四张图的平均背景
	for (size_t i = 0; i < imgNum; i++)
	{
		accumulate(srcImage[i], output);
	}
	output /= imgNum;
	output.convertTo(output, CV_8U);
	return output;
}

void BGLightCorrect2(const string& imgPath, const int imgNum, int kSz = 121)
{
	vector<Mat> srcImage(imgNum);
	vector<Mat> bgImage(imgNum);//背景光强
	vector<Mat> fgImage(imgNum);//调制光强
	vector<Mat> resImage(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		string path = imgPath + "/0 (" + to_string(i + 2) + ").bmp";
		Mat src = imread(path, IMREAD_GRAYSCALE);
		srcImage[i] = src;
		Mat bg;
		medianBlur(src, bg, kSz);
		bgImage[i] = bg;//背景
		fgImage[i] = src - bg;
	}

#if 0
	Scalar s = calMeanMat(bgImage);//分别获取四个通道背景像素的平均值
	float m_ = 0.f;
	for (size_t i = 0; i < imgNum; i++)
		m_ += s.val[i];
	m_ /= imgNum;//获取四个通道背景像素整体的平均值
	for (size_t i = 0; i < imgNum; i++)
	{
		float val = s.val[i] - m_;//同频率图像背景光强差值
		bgImage[i] -= val;//背景光强一致
		resImage[i] = bgImage[i] + fgImage[i];
		imwrite(imgPath + "/" + to_string(i + 2) + ".bmp", resImage[i]);
	}
#endif

#if 1
	//同频率图像平均背景
	Mat bgMean = calMeanMat_(bgImage);
	imwrite("./bgMean.jpg", bgMean);
	for (size_t i = 0; i < imgNum; i++)
	{
		resImage[i] = bgMean + fgImage[i];
		imwrite(imgPath + "/" + to_string(i + 2) + "test.bmp", resImage[i]);
	}
#endif

	//究竟是使用平均像素值还是平均背景图像
	//需要具体写论文时详细对比试验一下

}


void main()
{
	//BGLightCorrect1("./fringePattern", 4, Rect(215, 832, 740, 1068));
	BGLightCorrect2("./fringePattern", 4);

	waitKey(0);
	destroyAllWindows();
	return;
}