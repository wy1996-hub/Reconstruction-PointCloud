#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

enum way{AVERAGE, STDDEV, GAUSSIAN, REPROJ};

//相位结果求平均
Mat phaseDataFusion(const vector<Mat>& image, way num = AVERAGE, float* reproj = nullptr)
{
	assert(image.size() != 0);

	Mat res = Mat::zeros(image[0].size(), CV_32F);
	//accumulate函数会自动完成convertto功能

	switch (num)
	{
		case AVERAGE:
		{
			Mat avg = Mat::zeros(image[0].size(), CV_32F);
			for (size_t i = 0; i < image.size(); i++)
			{
				accumulate(image[i], avg);
			}
			avg /= image.size();
			res = avg.clone();
			break;
		}		
		case STDDEV:
		{
			//方差：刻画了我们期望未来出现的背景像素和观察到的均值的相似程度
			Mat avg = Mat::zeros(image[0].size(), CV_32F);
			Mat sum = Mat::zeros(image[0].size(), CV_32F);
			Mat sqsum = Mat::zeros(image[0].size(), CV_32F);
			Mat variance = Mat::zeros(image[0].size(), CV_32F);
			for (size_t i = 0; i < image.size(); i++)
			{
				accumulate(image[i], sum);
				accumulateSquare(image[i], sqsum);
			}
			avg = sum / image.size();
			double oneByN = 1.f / image.size();
			variance = oneByN*sqsum - (oneByN*oneByN)*sum.mul(sum);
			//根据方差找到这四张图中那些位置的点变化较大
			Mat var = variance.clone();
			normalize(var, var, 0, 255, NORM_MINMAX);
			var.convertTo(var, CV_8U);
			threshold(var, var, 0, 255, THRESH_OTSU);
			//去除偏差最大的
			int w = var.cols;
			int h = var.rows;
			for (size_t i = 0; i < h; i++)
			{
				const float* p1 = image[0].ptr<float>(i);
				const float* p2 = image[1].ptr<float>(i);
				const float* p3 = image[2].ptr<float>(i);
				const float* p4 = image[3].ptr<float>(i);
				const float* pavg = avg.ptr<float>(i);
				const uchar* pvar = var.ptr<uchar>(i);
				float* pres = res.ptr<float>(i);
				for (size_t j = 0; j < w; j++)
				{
					if (pvar[j] == 0)
					{
						pres[j] = pavg[j];
						continue;
					}
					vector<float> diff(4);
					diff[0] = abs(p1[j] - pavg[j]);
					diff[1] = abs(p2[j] - pavg[j]);
					diff[2] = abs(p3[j] - pavg[j]);
					diff[3] = abs(p4[j] - pavg[j]);
					sort(diff.begin(), diff.end());
					diff.pop_back();
					pres[j] = (diff[0] + diff[1] + diff[2]) / 3;
					vector<float>().swap(diff);
				}
			}
			break;
		}
		case GAUSSIAN:
		{
			Mat avg = Mat::zeros(image[0].size(), CV_32F);
			Mat sum = Mat::zeros(image[0].size(), CV_32F);
			Mat sqsum = Mat::zeros(image[0].size(), CV_32F);
			Mat variance = Mat::zeros(image[0].size(), CV_32F);
			for (size_t i = 0; i < image.size(); i++)
			{
				accumulate(image[i], sum);
				accumulateSquare(image[i], sqsum);
			}
			avg = sum / image.size();
			double oneByN = 1.f / image.size();
			variance = oneByN*sqsum - (oneByN*oneByN)*sum.mul(sum);
			//根据方差找到这四张图中那些位置的点变化较大
			Mat var = variance.clone();
			normalize(var, var, 0, 255, NORM_MINMAX);
			var.convertTo(var, CV_8U);
			threshold(var, var, 0, 255, THRESH_OTSU);
			//根据与均值的差值大小分配高斯权重
			int w = var.cols;
			int h = var.rows;
			for (size_t i = 0; i < h; i++)
			{
				const float* p1 = image[0].ptr<float>(i);
				const float* p2 = image[1].ptr<float>(i);
				const float* p3 = image[2].ptr<float>(i);
				const float* p4 = image[3].ptr<float>(i);
				const float* pavg = avg.ptr<float>(i);//平均值
				const float* psum = sum.ptr<float>(i);//和
				const float* pv = variance.ptr<float>(i);//pv[j]==sigma*sigma，原始方差
				const uchar* pvar = var.ptr<uchar>(i);//二值化后的方差
				float* pres = res.ptr<float>(i);
				for (size_t j = 0; j < w; j++)
				{
					if (pvar[j] == 0)
					{
						pres[j] = pavg[j];
						continue;
					}
					vector<float> gauss(4);
					gauss[0] = ((1 / sqrt(2 * CV_PI*pv[j]))*exp((-1)*pow(p1[j] - pavg[j], 2) / (2 * pv[j]))) / psum[j];
					gauss[1] = ((1 / sqrt(2 * CV_PI*pv[j]))*exp((-1)*pow(p2[j] - pavg[j], 2) / (2 * pv[j]))) / psum[j];
					gauss[2] = ((1 / sqrt(2 * CV_PI*pv[j]))*exp((-1)*pow(p3[j] - pavg[j], 2) / (2 * pv[j]))) / psum[j];
					gauss[3] = ((1 / sqrt(2 * CV_PI*pv[j]))*exp((-1)*pow(p4[j] - pavg[j], 2) / (2 * pv[j]))) / psum[j];
					pres[j] = gauss[0] * p1[j] + gauss[1] * p2[j] + gauss[2] * p3[j] + gauss[3] * p4[j];
					vector<float>().swap(gauss);
				}
			}
			break;
		}
		case REPROJ:
		{
			Mat avg = Mat::zeros(image[0].size(), CV_32F);
			Mat sum = Mat::zeros(image[0].size(), CV_32F);
			Mat sqsum = Mat::zeros(image[0].size(), CV_32F);
			Mat variance = Mat::zeros(image[0].size(), CV_32F);
			for (size_t i = 0; i < image.size(); i++)
			{
				accumulate(image[i], sum);
				accumulateSquare(image[i], sqsum);
			}
			avg = sum / image.size();
			double oneByN = 1.f / image.size();
			variance = oneByN*sqsum - (oneByN*oneByN)*sum.mul(sum);
			Mat var = variance.clone();
			normalize(var, var, 0, 255, NORM_MINMAX);
			var.convertTo(var, CV_8U);
			threshold(var, var, 0, 255, THRESH_OTSU);
			//根据各个位置重投影误差大小分配权重
			int w = var.cols;
			int h = var.rows;
			for (size_t i = 0; i < h; i++)
			{
				const float* p1 = image[0].ptr<float>(i);
				const float* p2 = image[1].ptr<float>(i);
				const float* p3 = image[2].ptr<float>(i);
				const float* p4 = image[3].ptr<float>(i);
				const float* pavg = avg.ptr<float>(i);//平均值
				const float* psum = sum.ptr<float>(i);//和
				const uchar* pvar = var.ptr<uchar>(i);//二值化后的方差
				float* pres = res.ptr<float>(i);
				for (size_t j = 0; j < w; j++)
				{
					if (pvar[j] == 0)
					{
						pres[j] = pavg[j];
						continue;
					}
					vector<float> gauss(4);
					float sumproj = reproj[0] + reproj[1] + reproj[2] + reproj[3];
					gauss[0] = sumproj / reproj[0];//取倒数
					gauss[1] = sumproj / reproj[0];
					gauss[2] = sumproj / reproj[0];
					gauss[3] = sumproj / reproj[0];
					float ss = gauss[0] + gauss[1] + gauss[2] + gauss[3];
					for (size_t n = 0; n < 4; n++)
					{
						gauss[n] /= ss;
					}
					pres[j] = gauss[0] * p1[j] + gauss[1] * p2[j] + gauss[2] * p3[j] + gauss[3] * p4[j];
					vector<float>().swap(gauss);
				}
			}
			break;
		}
		default:
			break;
	}	
	return res;
}

void main()
{
	const size_t imgNum = 4;
	vector<Mat> image(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		image[i] = imread("./" + to_string(i + 1) + ".jpg", IMREAD_GRAYSCALE);
		image[i].convertTo(image[i], CV_32F, 1.0 / 255.0);
	}
	Mat t1 = image[0], t2 = image[1], t3 = image[2], t4 = image[3];

	float* p = new float[4];
	for (size_t i = 0; i < 4; i++)
	{
		p[i] = i + 1;
	}
	Mat dst = phaseDataFusion(image, REPROJ, p);
	delete[] p;
	p = nullptr;

	normalize(dst, dst, 0, 255, NORM_MINMAX);//计算中出现负数，要先归一化到0-255
	dst.convertTo(dst, CV_8U);
	imwrite("./res.jpg", dst);

	return;
}