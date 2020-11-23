#ifndef __FTP_H__
#define __FTP_H__

#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

class FTP
{
public:
	FTP(Mat src);
	virtual ~FTP();
	void FTPprocess();
	Mat DFTconvolve(Mat, Mat);

private:
	Mat img;
	Mat& FTPCentralize(Mat& src);
};

#endif






