#include "wavelet.h"

int thundershock();

void main() {

	wavelet w;
	Mat src = imread("plat_cut3.bmp");




	//Mat gray, thresh;
	//cvtColor(src, gray, CV_RGB2GRAY);
	//threshold(gray, thresh, 248, 255, CV_THRESH_BINARY);
	//Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
	//dilate(thresh, thresh, kernel);
	//namedWindow("thresh", WINDOW_NORMAL);
	//imshow("thresh", thresh);
	//vector<vector<Point>> contours;
	//vector<Vec4i> hierarchy;
	//findContours(thresh, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);
	//Mat bin = Mat::zeros(thresh.size(), thresh.type());
	//vector<Rect> rect(contours.size());
	//for (int i = 0; i < contours.size(); i++)
	//{
	//	rect[i] = boundingRect(Mat(contours[i])); 
	//	double rate = (double)rect[i].height / rect[i].width;
	//	if (contourArea(contours[i]) > 60 
	//		&& rect[i].width < 120 
	//		&& rect[i].width > 12
	//		&& rate > 1.3)
	//	{
	//		drawContours(bin, contours, i, Scalar(255), CV_FILLED);
	//		//drawContours(bin, contours, i, Scalar(0), 1);
	//		rectangle(bin, rect[i].tl(), rect[i].br(), Scalar(255), 1, 8, 0);
	//	}
	//}
	//namedWindow("dst", WINDOW_NORMAL);
	//imshow("dst", bin);




	//Ð§¹û½Ï²î
	/*Mat full_src;
	w.laplace_decompose(src, 3, full_src);
	w.ware_operate(full_src, 3);
	Mat src_recover;
	w.wave_recover(full_src, src_recover, 3);
	imshow("recover", src_recover);*/



	const int wavecount = 2;
	int recovercount = std::pow(2, wavecount);
	Mat img, imgGray;
	cvtColor(src, imgGray, CV_RGB2GRAY);
	normalize(imgGray, img, 0, 255, CV_MINMAX);
	namedWindow("img", CV_WINDOW_NORMAL);
	imshow("img", img);
	Mat float_src;

	img.convertTo(float_src, CV_32F);
	Mat imgWave = w.DWT(float_src, "haar", wavecount);
	imgWave.convertTo(float_src, CV_32F);

	//Mat yu = Mat::zeros(float_src.size(), float_src.type());
	for (int i = 0; i < float_src.rows / recovercount; i++)
	{
		for (int j = 0; j < float_src.cols / recovercount; j++)
		{
			float* p = float_src.ptr<float>(i, j);
			*p = 0.;
		/*	float* p = yu.ptr<float>(i, j);
			*p = *float_src.ptr<float>(i, j);*/
		}
	}

	Mat imgIWave = w.IDWT(float_src, "haar", wavecount);
	normalize(imgWave, imgWave, 0, 255, CV_MINMAX);
	namedWindow("imgWave", CV_WINDOW_NORMAL);
	imshow("imgWave", Mat_<uchar>(imgWave));
	normalize(imgIWave, imgIWave, 0, 255, CV_MINMAX);
	namedWindow("IWDT", CV_WINDOW_NORMAL);
	imshow("IWDT", Mat_<uchar>(imgIWave));
	Mat im = Mat_<uchar>(imgIWave);
	Mat dst = imgGray + im;
	imwrite("IWDT_char.jpg", dst);

	waitKey(0);
	destroyAllWindows();
	return;
}


