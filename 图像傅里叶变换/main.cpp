#include <iostream>
#include "FTP.h"

void FTPprocess()
{
	Mat img = imread("test.jpg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "load image failed!" << endl;
		return ;
	}
	imshow("img", img);
	int height = getOptimalDFTSize(img.rows);
	int width = getOptimalDFTSize(img.cols);//获得适合DFT的图像尺寸 ,选取最适合做fft的宽和高
	Mat padded;
	copyMakeBorder(img, padded, 0, height - img.rows, 0, width - img.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(),CV_32F) };
	Mat complexImg;
	merge(planes, 2, complexImg);//planes[0], planes[1]是实部和虚部
	cv::dft(complexImg, complexImg, cv::DFT_SCALE | cv::DFT_COMPLEX_OUTPUT);//离散傅里叶变换
	split(complexImg, planes);//获得含有实部、虚部通道的vector<Mat>	

	cv::Mat ph, mag;//定义幅度谱和相位谱
	cv::phase(planes[0], planes[1], ph);
	cv::magnitude(planes[0], planes[1], mag);
	//由实部planes[0]和虚部planes[1]得到幅度谱mag和相位谱ph
	//如果需要对实部planes[0]和虚部planes[1]，或者幅度谱mag和相位谱ph进行操作，在这里进行更改
	mag += Scalar::all(1);
	cv::log(mag, mag);
	Mat org_mag = mag.clone();//此时两个mat均为float
	normalize(org_mag, org_mag, 0, 255, NORM_MINMAX);
	imshow("org_mag", org_mag);


	mag = mag(Rect(0, 0, mag.cols&-2, mag.rows&-2));//调整图像大小为偶数值
	int cx = mag.cols / 2;
	int cy = mag.rows / 2;
	Mat tmp;
	Mat q1(mag, Rect(0, 0, cx, cy));
	Mat q4(mag, Rect(cx, 0, cx, cy));
	Mat q2(mag, Rect(0, cy, cx, cy));
	Mat q3(mag, Rect(cx, cy, cx, cy));
	q1.copyTo(tmp);
	q3.copyTo(q1);
	tmp.copyTo(q3);
	q4.copyTo(tmp);
	q2.copyTo(q4);
	tmp.copyTo(q2);
	normalize(mag, mag, 0, 255, NORM_MINMAX);//频谱图可以normalize
	imshow("center_mag", mag);

	Mat idft_mag1, idft, idft_mag2;
	cv::idft(complexImg, idft_mag1, DFT_REAL_OUTPUT /*| DFT_SCALE*/);//傅里叶反变换
	 //由中心化之后的幅度谱mag和相位谱ph恢复实部planes[0]和虚部planes[1]
	cv::polarToCart(mag, ph, planes[0], planes[1]);
	cv::merge(planes, 2, idft);
	cv::dft(idft, idft_mag2, DFT_REAL_OUTPUT |/* DFT_SCALE|*/ DFT_INVERSE);

	idft_mag1.convertTo(idft_mag1, CV_8UC1);
	idft_mag2.convertTo(idft_mag2, CV_8UC1);//单通道的空域图片需要convertTo而不是normalize
	imshow("idft_mag1", idft_mag1);
	imshow("idft_mag2", idft_mag2);
}

int main(int argc, char* argv[])
{	
	Mat img = imread("test.jpg", IMREAD_GRAYSCALE);
	if (img.empty())
	{
		cout << "load image failed!" << endl;
		return -1;
	}

	FTP t(img);
	//t.FTPprocess();

	Mat p = Mat::ones(9, 9, CV_8UC1);
	Mat p2, img2, result;
	p.convertTo(p2, CV_32FC1, 1.0 / 81);
	img.convertTo(img2, CV_32FC1, 1.0 / 255);
	result = t.DFTconvolve(img2, p2);
	imshow("result", result);

	cv::waitKey();
	system("pause");
	return 0;
}
