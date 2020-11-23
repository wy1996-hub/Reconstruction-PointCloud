#include "FTP.h"

FTP::FTP(Mat src):img(src)
{
	if (img.empty())
	{
		cout << "load image failed!" << endl;
		return;
	}
}

void FTP::FTPprocess()
{
	int height = getOptimalDFTSize(img.rows);
	int width = getOptimalDFTSize(img.cols);//获得适合DFT的图像尺寸
	Mat padded;
	copyMakeBorder(img, padded, 0, height - img.rows, 0, width - img.cols, BORDER_CONSTANT, Scalar::all(0));
	vector<Mat> planes = { Mat_<float>(padded),Mat::zeros(padded.size(),CV_32F) };
	Mat complexImg;
	merge(planes, complexImg);//合并成一个对象，作为dft的输入

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
	normalize(org_mag, org_mag, 0, 255, NORM_MINMAX);//频谱图可以normalize
	imshow("org_mag", org_mag);

	Mat center_mag = FTPCentralize(mag);//中心化
	cv::normalize(center_mag, center_mag, 0, 255, NORM_MINMAX);//float的mat需归一化到【0,1】或【0,255】
	imshow("center_mag", center_mag);

	Mat idft_mag;
	cv::idft(complexImg, idft_mag, DFT_REAL_OUTPUT /*| DFT_SCALE*/);//傅里叶反变换
	//此时不能加DFT_SCALE，否则无法显示，全是黑图
	idft_mag = idft_mag(Rect(0, 0, img.cols, img.rows));
	/*normalize(idft_mag, idft_mag, 0, 255, NORM_MINMAX);*/
	idft_mag.convertTo(idft_mag, CV_8UC1);//单通道的空域图片需要convertTo而不是normalize
	imshow("idft_mag", idft_mag);
	imwrite("idft_mag.jpg", idft_mag);
}

Mat& FTP::FTPCentralize(Mat& src)//频谱中心化
{
	src = src(Rect(0, 0, src.cols&-2, src.rows&-2));//调整图像大小位偶数值
	int cx = src.cols / 2;
	int cy = src.rows / 2;
	Mat tmp;
	Mat q1(src, Rect(0, 0, cx, cy));
	Mat q4(src, Rect(cx, 0, cx, cy));
	Mat q2(src, Rect(0, cy, cx, cy));
	Mat q3(src, Rect(cx, cy, cx, cy));

	q1.copyTo(tmp);
	q3.copyTo(q1);
	tmp.copyTo(q3);

	q4.copyTo(tmp);
	q2.copyTo(q4);
	tmp.copyTo(q2);	
	return src;
}

/*
	功能：通过傅里叶逆变换，进行频域乘积，实现空域卷积
	输入A，浮点型，【0,1】
	输入B，浮点型，模板，【0,1】（不然结果为全白图）
	输出C，空域卷积结果图
*/
Mat FTP::DFTconvolve(Mat A, Mat B)
{
	Mat C;
	C.create(abs(A.rows-B.rows)+1, abs(A.cols - B.cols) + 1, A.type());

	Size dftSize;
	dftSize.width = getOptimalDFTSize(A.cols + B.cols - 1);
	dftSize.height = getOptimalDFTSize(A.rows + B.rows - 1);

	Mat tmpA(dftSize, A.type(), Scalar::all(0)); 
	Mat tmpB(dftSize, B.type(), Scalar::all(0));
	Mat roiA(tmpA, Rect(0, 0, A.cols, A.rows));
	Mat roiB(tmpB, Rect(0, 0, B.cols, B.rows));
	A.copyTo(roiA);
	B.copyTo(roiB);

	dft(tmpA, tmpA, 0, A.rows);
	dft(tmpB, tmpB, 0, B.rows);
	mulSpectrums(tmpA, tmpB, tmpA, 0, false);//频域相乘

	idft(tmpA, tmpA, DFT_SCALE, C.rows);

	tmpA(Rect(0, 0, C.cols, C.rows)).copyTo(C);
	return C;
}

FTP::~FTP()
{
	cv::destroyAllWindows();
}
