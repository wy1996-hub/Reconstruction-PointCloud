#pragma once

#include <iostream>
#include <opencv2/opencv.hpp>
#include <cmath>

using namespace std;
using namespace cv;

class wavelet
{
public:
	wavelet();
	~wavelet();

/*********************************************000000000*****************************************************/
	int thundershock();


/************************************************1111111*****************************************************/
	void laplace_decompose(Mat& src, int s, Mat &wave);//小波分解
	void wave_recover(Mat full_scale, Mat &original, int level);//小波复原
	void ware_operate(Mat &full_scale, int level);//小波操作


/***********************************************2222222222******************************************************/
	Mat DWT(const Mat &_src, const string _wname, const int _level);// 小波变换	
	Mat IDWT(const Mat &_src, const string _wname, const int _level);// 小波逆变换
	void wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter);// 分解包
	void wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter);// 重构包
	Mat waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);// 小波分解
	Mat waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter);// 小波重建
};

