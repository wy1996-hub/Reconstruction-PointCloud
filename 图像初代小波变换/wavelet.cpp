#include "wavelet.h"

wavelet::wavelet()
{
}

wavelet::~wavelet()
{
}


int wavelet::thundershock() {
	Mat src = imread("2.jpg", 0);
	Mat dst;
	int Width = src.cols;
	int Height = src.rows;
	//小波分解次数
	int depth = 1;//
	int depthcount = 1;
	//改变数据格式防止溢出
	Mat tmp = Mat::zeros(src.size(), CV_32FC1);
	Mat  wavelet = Mat::zeros(src.size(), CV_32FC1);
	Mat  imgtmp = src.clone();
	imgtmp.convertTo(imgtmp, CV_32FC1);
	//执行小波变换
	while (depthcount <= depth) {
		Width = src.cols / pow(2, depthcount - 1);
		Height = src.rows / pow(2, depthcount - 1);
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width / 2; j++) {
				tmp.at<float>(i, j) = (imgtmp.at<float>(i, 2 * j) + imgtmp.at<float>(i, 2 * j + 1)) / 2;
				tmp.at<float>(i, j + Width / 2) = (imgtmp.at<float>(i, 2 * j) - imgtmp.at<float>(i, 2 * j + 1)) / 2;
			}
		}
		for (int i = 0; i < Height / 2; i++) {
			for (int j = 0; j < Width; j++) {
				wavelet.at<float>(i, j) = (tmp.at<float>(2 * i, j) + tmp.at<float>(2 * i + 1, j)) / 2;
				wavelet.at<float>(i + Height / 2, j) = (tmp.at<float>(2 * i, j) - tmp.at<float>(2 * i + 1, j)) / 2;
			}
		}
		imgtmp = wavelet;
		depthcount++;
	}
	namedWindow("DST", WINDOW_AUTOSIZE);
	convertScaleAbs(wavelet, dst);
	imshow("DST", dst);
	//反变换
	while (depthcount > 1) {

		for (int i = 0; i < Height / 2; i++) {
			for (int j = 0; j < Width; j++) {
				tmp.at<float>(2 * i, j) = wavelet.at<float>(i, j) + wavelet.at<float>(i + Height / 2, j);
				tmp.at<float>(2 * i + 1, j) = wavelet.at<float>(i, j) - wavelet.at<float>(i + Height / 2, j);
			}
		}
		for (int i = 0; i < Height; i++) {
			for (int j = 0; j < Width / 2; j++) {
				imgtmp.at<float>(i, 2 * j) = tmp.at<float>(i, j) + tmp.at<float>(i, j + Width / 2);
				imgtmp.at<float>(i, 2 * j + 1) = tmp.at<float>(i, j) - tmp.at<float>(i, j + Width / 2);
			}
		}
		depthcount--;
		wavelet = imgtmp;
		Height *= 2;
		Width *= 2;
	}
	namedWindow("restruct", WINDOW_AUTOSIZE);
	convertScaleAbs(imgtmp, imgtmp);
	imshow("restruct", imgtmp);
	waitKey();
	//imwrite("haar.jpg", dst);
	return 0;
}

/************************************************************************/

//小波分解
void wavelet::laplace_decompose(Mat& src, int s, Mat &wave)
{
	Mat full_src(src.size(), CV_32FC1);
	Mat dst = src.clone();
	for (int m = 0; m < s; ++m)
	{
		dst.convertTo(dst, CV_32FC1);
		Mat wave_src(dst.size(), CV_32FC1);
		//列变换，detail=mean-original
		for (int i = 0; i < wave_src.rows; i++)
		{
			float* pwave = wave_src.ptr<float>(i);
			float* pdst = dst.ptr<float>(i);
			for (int j = 0; j < wave_src.cols / 2; j++)
			{
				//wave_src.at<float>(i, j) = (dst.at<float>(i, 2 * j) + dst.at<float>(i, 2 * j + 1)) / 2;
				//wave_src.at<float>(i, j + wave_src.cols / 2) = wave_src.at<float>(i, j) - dst.at<float>(i, 2 * j);
				pwave[j] = (pdst[2 * j] + pdst[2 * j + 1]) / 2;
				pwave[j + wave_src.cols / 2] = pwave[j] - pdst[2 * j];
			}
		}
		Mat temp = wave_src.clone();
		for (int i = 0; i < wave_src.rows / 2; i++)
		{
			float* pwave = wave_src.ptr<float>(i);
			float* pwave2 = wave_src.ptr<float>(i + wave_src.rows / 2);
			float* ptmp= temp.ptr<float>(2 * i);
			float* ptmp2 = temp.ptr<float>(2 * i + 1);
			for (int j = 0; j < wave_src.cols / 2; j++)
			{
				//wave_src.at<float>(i, j) = (temp.at<float>(2 * i, j) + temp.at<float>(2 * i + 1, j)) / 2;
				//wave_src.at<float>(i + wave_src.rows / 2, j) = wave_src.at<float>(i, j) - temp.at<float>(2 * i, j);
				pwave[j] = (ptmp[j] + ptmp2[j]) / 2;
				pwave2[j] = pwave[j] - ptmp[j];
			}
		}
		dst.release();
		dst = wave_src(Rect(0, 0, wave_src.cols / 2, wave_src.rows / 2));
		wave_src.copyTo(full_src(Rect(0, 0, wave_src.cols, wave_src.rows)));
	}
	wave = full_src.clone();
	imshow("laplace_decompose", wave);
}

//小波操作
void wavelet::ware_operate(Mat &full_scale, int level)
{
	//取出最低尺度的那一层，对其进行操作，仅最低尺度那层可以对时域进行操作，其他层只能对频域进行操作
	Mat temp = full_scale(Rect(0, 0, full_scale.cols / 4, full_scale.rows / 4));
	temp = temp(Rect(0, 0, temp.cols / 2, temp.rows / 2));
	Mat temp2 = temp.clone();
	//这里对时域操作，降低灰度
	for (int i = 0; i < temp2.rows; i++)
		for (int j = 0; j < temp2.cols; j++)
			temp2.at<float>(i, j) -= 20;
	temp2.copyTo(temp);
	//这里对频域操作，拉伸细节
	//先处理左下角
	for (int i = temp.rows / 2; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols / 2; j++)
		{
			if (temp.at<float>(i, j) > 0)
				temp.at<float>(i, j) += 5;
			if (temp.at<float>(i, j) < 0)
				temp.at<float>(i, j) -= 5;
		}
	}
	//再处理右半边
	for (int i = 0; i < temp.rows; i++)
	{
		for (int j = 0; j < temp.cols; j++)
		{
			if (temp.at<float>(i, j) > 0)
				temp.at<float>(i, j) += 5;
			if (temp.at<float>(i, j) < 0)
				temp.at<float>(i, j) -= 5;
		}
	}
	imshow("ware_operate", temp);
}

//小波复原
void wavelet::wave_recover(Mat full_scale, Mat &original, int level)
{
	//每一个循环中把一个级数的小波还原
	for (int m = 0; m < level; m++)
	{
		Mat temp = full_scale(Rect(0, 0, full_scale.cols / pow(2, level - m - 1), full_scale.rows / pow(2, level - m - 1)));

		//先恢复左边
		Mat recover_src(temp.rows, temp.cols, CV_32FC1);
		for (int i = 0; i < recover_src.rows; i++)
		{
			for (int j = 0; j < recover_src.cols / 2; j++)
			{
				if (i % 2 == 0)
					recover_src.at<float>(i, j) = temp.at <float>(i / 2, j) - temp.at<float>(i / 2 + recover_src.rows / 2, j);
				else
					recover_src.at<float>(i, j) = temp.at <float>(i / 2, j) + temp.at<float>(i / 2 + recover_src.rows / 2, j);
			}
		}
		Mat temp2 = recover_src.clone();
		//再恢复整个
		for (int i = 0; i < recover_src.rows; i++)
		{
			for (int j = 0; j < recover_src.cols; j++)
			{
				if (j % 2 == 0)
					recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) - temp.at<float>(i, j / 2 + temp.cols / 2);
				else
					recover_src.at<float>(i, j) = temp2.at<float>(i, j / 2) + temp.at<float>(i, j / 2 + temp.cols / 2);
			}
		}
		recover_src.copyTo(temp);
	}
	original = full_scale.clone();
	original.convertTo(original, CV_8UC1);
}


/************************************************************************/

// 离散小波变换
Mat wavelet::DWT(const Mat &_src, const string _wname, const int _level)
{
	Mat src = Mat_<float>(_src);
	Mat dst = Mat::zeros(src.rows, src.cols, src.type());
	int N = src.rows;
	int D = src.cols;
	//高通低通滤波器 
	Mat lowFilter;
	Mat highFilter;
	wavelet_D(_wname, lowFilter, highFilter);
	//小波变换
	int t = 1;
	int row = N;
	int col = D;
	while (t <= _level)
	{
		//先进行 行小波变换
		for (int i = 0; i < row; i++)
		{
			//取出src中要处理的数据的一行
			Mat oneRow = Mat::zeros(1, col, src.type());
			for (int j = 0; j < col; j++)
			{
				oneRow.at<float>(0, j) = src.at<float>(i, j);
			}
			oneRow = waveletDecompose(oneRow, lowFilter, highFilter);
			for (int j = 0; j < col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}
		
		//小波列变换
		for (int j = 0; j < col; j++)
		{
			Mat oneCol = Mat::zeros(row, 1, src.type());
			for (int i = 0; i < row; i++)
			{
				oneCol.at<float>(i, 0) = dst.at<float>(i, j);//dst,not src
			}
			oneCol = (waveletDecompose(oneCol.t(), lowFilter, highFilter)).t();
			for (int i = 0; i < row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}
		}

		char s[10];
		itoa(t, s, 10);
		//namedWindow(s, WINDOW_NORMAL);
		//imshow(s, Mat_<uchar>(dst));
		///*Mat dst1 = dst.clone();
		//normalize(dst1, dst1, 0, 255, CV_MINMAX);
		//dst1.convertTo(dst1, CV_8UC1);
		//imwrite("kkk.bmp", dst1);*/
		//waitKey(0);

		//更新
		row /= 2;
		col /= 2;
		t++;
		src = dst;
	}
	return dst;
}

// 小波逆变换
Mat wavelet::IDWT(const Mat &_src, const string _wname, const int _level)
{
	Mat src = Mat_<float>(_src);
	Mat dst;//=Mat::zeros(src.rows,src.cols,src.type());
	src.copyTo(dst);
	int N = src.rows;
	int D = src.cols;

	//高低通滤波器
	Mat lowFilter;
	Mat highFilter;
	wavelet_R(_wname, lowFilter, highFilter);

	//小波变换
	int t = 1;
	int row = N / std::pow(2., _level - 1);
	int col = D / std::pow(2., _level - 1);

	while (row <= N && col <= D)
		//while(t<=_level)
	{
		//列逆变换
		for (int j = 0; j<col; j++)
		{
			Mat oneCol = Mat::zeros(row, 1, src.type());

			for (int i = 0; i<row; i++)
			{
				oneCol.at<float>(i, 0) = src.at<float>(i, j);
			}
			oneCol = (waveletReconstruct(oneCol.t(), lowFilter, highFilter)).t();

			for (int i = 0; i<row; i++)
			{
				dst.at<float>(i, j) = oneCol.at<float>(i, 0);
			}

		}

		//行逆变换
		for (int i = 0; i<row; i++)
		{
			Mat oneRow = Mat::zeros(1, col, src.type());
			for (int j = 0; j<col; j++)
			{
				oneRow.at<float>(0, j) = dst.at<float>(i, j);
			}
			oneRow = waveletReconstruct(oneRow, lowFilter, highFilter);
			for (int j = 0; j<col; j++)
			{
				dst.at<float>(i, j) = oneRow.at<float>(0, j);
			}
		}

		char s[10];
		itoa(t, s, 10);
		//Rect rrr=Rect(Point(col-1,row-1),Point(src.cols-1,src.rows-1));
		//Rect rrr=Rect(Point(0,0),Point(col-1,row-1));
		/*
		Mat showImg;//=//dst;//(rrr);
		dst.copyTo(showImg);
		Mat showImg1;
		showImg.copyTo(showImg1);
		normalize(showImg1,showImg1,0,255,CV_MINMAX);
		imshow(s,Mat_<uchar>(showImg1));
		waitKey(1);
		*/

		row *= 2;
		col *= 2;
		t++;
		src = dst;
	}
	return dst;
}

// 分解包
void wavelet::wavelet_D(const string _wname, Mat &_lowFilter, Mat &_highFilter)
{
	if (_wname == "haar" || _wname == "db1")
	{
		int N = 2;
		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		_lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		_highFilter.at<float>(0, 0) = -1 / sqrtf(N);
		_highFilter.at<float>(0, 1) = 1 / sqrtf(N);
	}
	else if (_wname == "sym2")
	{
		int N = 4;
		float h[] = { -0.4830, 0.8365, -0.2241, -0.1294 };
		float l[] = { -0.1294, 0.2241,  0.8365, 0.4830 };

		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			_lowFilter.at<float>(0, i) = l[i];
			_highFilter.at<float>(0, i) = h[i];
		}
	}
}

// 重构包
void wavelet::wavelet_R(const string _wname, Mat &_lowFilter, Mat &_highFilter)
{
	if (_wname == "haar" || _wname == "db1")
	{
		int N = 2;
		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		_lowFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_lowFilter.at<float>(0, 1) = 1 / sqrtf(N);

		_highFilter.at<float>(0, 0) = 1 / sqrtf(N);
		_highFilter.at<float>(0, 1) = -1 / sqrtf(N);
	}
	else if (_wname == "sym2")
	{
		int N = 4;
		float h[] = { -0.1294,-0.2241,0.8365,-0.4830 };
		float l[] = { 0.4830, 0.8365, 0.2241, -0.1294 };

		_lowFilter = Mat::zeros(1, N, CV_32F);
		_highFilter = Mat::zeros(1, N, CV_32F);

		for (int i = 0; i < N; i++)
		{
			_lowFilter.at<float>(0, i) = l[i];
			_highFilter.at<float>(0, i) = h[i];
		}
	}
}

// 小波分解
Mat wavelet::waveletDecompose(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	Mat &src = Mat_<float>(_src);

	int D = src.cols;

	Mat &lowFilter = Mat_<float>(_lowFilter);
	Mat &highFilter = Mat_<float>(_highFilter);

	//频域滤波或时域卷积；ifft( fft(x) * fft(filter)) = cov(x,filter) 
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());

	filter2D(src, dst1, -1, lowFilter);
	filter2D(src, dst2, -1, highFilter);

	//下采样
	Mat downDst1 = Mat::zeros(1, D / 2, src.type());
	Mat downDst2 = Mat::zeros(1, D / 2, src.type());

	pyrDown(dst1, downDst1, downDst1.size());
	pyrDown(dst2, downDst2, downDst2.size());
	//resize(dst1, downDst1, downDst1.size());
	//resize(dst2, downDst2, downDst2.size());

	//数据拼接
	for (int i = 0; i < D / 2; i++)
	{
		src.at<float>(0, i) = downDst1.at<float>(0, i);
		src.at<float>(0, i + D / 2) = downDst2.at<float>(0, i);

	}
	return src;
}

// 小波重建
Mat wavelet::waveletReconstruct(const Mat &_src, const Mat &_lowFilter, const Mat &_highFilter)
{
	assert(_src.rows == 1 && _lowFilter.rows == 1 && _highFilter.rows == 1);
	assert(_src.cols >= _lowFilter.cols && _src.cols >= _highFilter.cols);
	Mat &src = Mat_<float>(_src);

	int D = src.cols;

	Mat &lowFilter = Mat_<float>(_lowFilter);
	Mat &highFilter = Mat_<float>(_highFilter);

	//上采样
	Mat roi1(src, Rect(0, 0, D / 2, 1));
	Mat roi2(src, Rect(D / 2, 0, D / 2, 1));

	Mat Up1 = Mat::zeros(1, D, src.type());
	Mat Up2 = Mat::zeros(1, D, src.type());

	//线性插值,插值为0
	//for (int i = 0, cnt = 0; i < D / 2; i++, cnt += 2)
	//{
	//	Up1.at<float>(0, cnt) = src.at<float>(0, i);     // 前一半
	//	Up2.at<float>(0, cnt) = src.at<float>(0, i + D / 2); // 后一半
	//}

	pyrUp(roi1, Up1, Up1.size());
	pyrUp(roi2, Up2, Up2.size());
	//resize(roi1, Up1, Up1.size(), 0, 0, INTER_CUBIC);
	//resize(roi2, Up2, Up2.size(), 0, 0, INTER_CUBIC);

	// 前一半低通，后一半高通
	Mat dst1 = Mat::zeros(1, D, src.type());
	Mat dst2 = Mat::zeros(1, D, src.type());
	filter2D(Up1, dst1, -1, lowFilter);
	filter2D(Up2, dst2, -1, highFilter);

	// 结果相加
	dst1 = dst1 + dst2;
	return dst1;
}