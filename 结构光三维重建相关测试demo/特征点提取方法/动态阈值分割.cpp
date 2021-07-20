#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

// ��Ե��ⷽ��
int findFP1(const char* filename)
{
	Mat img = imread(filename);
	if (img.empty())
	{
		cout << "can not open " << filename << endl;
		return -1;
	}

	Mat img3, img2, img4;

	cvtColor(img, img2, COLOR_BGR2GRAY);   //�Ѳ�ɫͼת��Ϊ�ڰ�ͼ��
	GaussianBlur(img2, img2, Size(9, 9), 2, 2);
	threshold(img2, img3, 70, 255, THRESH_BINARY);  //ͼ���ֵ������ע����ֵ�仯
	namedWindow("detecte circles", WINDOW_NORMAL);
	imshow("detecte circles", img3);
	Canny(img3, img3, 60, 100);//��Ե���
	namedWindow("detect circles", WINDOW_NORMAL);
	imshow("detect circles", img3);
	vector<vector<Point>>contours;
	vector<Vec4i>hierarchy;
	findContours(img3, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE);//���ҳ����е�Բ�߽�
	int index = 0;
	for (; index >= 0; index = hierarchy[index][0])
	{
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(img, contours, index, color, FILLED, 8, hierarchy);
	}

	namedWindow("detected circles", WINDOW_NORMAL);
	imshow("detected circles", img);
	//��׼Բ��ͼƬ��һ������Բ�����Բ���OpenCV�������Բ�ķ����������
	Mat pointsf;
	Mat(contours[0]).convertTo(pointsf, CV_32F);
	RotatedRect box = fitEllipse(pointsf);
	cout << box.center;
	waitKey();

	return 0;
}

// ��̬��ֵ�ָ��
bool findFP2(Mat srcImage, Rect rect, int kSize, int filterMode = 0, float minArea = 100.f, float maxArea = 22500.f)
{
	Mat grayImage;
	cvtColor(srcImage, grayImage, COLOR_BGR2GRAY);
	Mat viewGray = grayImage(rect);
	Mat view = srcImage(rect);

	Mat blurImg;
	switch (filterMode)
	{
	case 0:
	{
		blur(viewGray, blurImg, Size(kSize, kSize));//�Լ۱ȸߣ��������ȳ���
		break;
	}
	case 1:
	{
		GaussianBlur(viewGray, blurImg, Size(kSize, kSize), 0);//����ϰ�������ֻ��ȡ�˱�Ե
		break;
	}
	case 2:
	{
		medianBlur(viewGray, blurImg, kSize);//��������Ч������ã���ʱ��������Գ��Ծ�ֵ�˲�
		break;
	}
	default:
		break;
	}
		
	Mat dst;
	subtract(viewGray, blurImg, dst);
	Mat fore = dst.clone();
	threshold(fore, fore, 0, 255, THRESH_OTSU);
	bitwise_not(fore, fore);

	Size board_size = Size(9, 11);
	vector<Point2f> image_points;
	SimpleBlobDetector::Params parameters = SimpleBlobDetector::Params();
	parameters.filterByArea = true;  parameters.minArea = minArea; parameters.maxArea = maxArea;

	bool ok = findCirclesGrid(fore, board_size, image_points, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, SimpleBlobDetector::create(parameters));
	if (0 == ok)
	{
		cout << "��Ƭ��ȡԲ��ʧ��" << endl;
		return false;
	}
	else
	{
		cornerSubPix(fore, image_points, Size(1, 1), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.001));
		drawChessboardCorners(view, board_size, image_points, true);
	}

	return true;
}

void main()
{
	Rect rect(0, 0, 3840, 2748);
	string path = "./camcalib/650/0 (1).bmp";
	Mat grayImage = imread(path, IMREAD_GRAYSCALE);
	Mat viewGray = grayImage(rect);
	Mat srcImage = imread(path);
	Mat view = srcImage(rect);

	Mat blurImg;
	int kSz = 121;
	blur(viewGray, blurImg, Size(kSz, kSz));//�Լ۱ȸߣ��������ȳ���
	//GaussianBlur(viewGray, blurImg, Size(kSz, kSz), 0);//����ϰ�������ֻ��ȡ�˱�Ե
	//medianBlur(viewGray, blurImg, kSz);//��������Ч������ã���ʱ��������Գ��Ծ�ֵ�˲�

	Mat dst, gdst, mdst;
	subtract(viewGray, blurImg, dst);
	Mat fore = dst;
	threshold(fore, fore, 0, 255, THRESH_OTSU);
	//threshold(mdst, mdst, 30, 255, THRESH_BINARY);
	bitwise_not(fore, fore);

	Size board_size = Size(9, 11);
	vector<Point2f> image_points;
	SimpleBlobDetector::Params parameters = SimpleBlobDetector::Params();
	parameters.filterByArea = true;  parameters.minArea = 100; parameters.maxArea = 22500;

	bool ok = findCirclesGrid(fore, board_size, image_points, CALIB_CB_SYMMETRIC_GRID | CALIB_CB_CLUSTERING, SimpleBlobDetector::create(parameters));
	if (0 == ok)
	{
		cout << "��Ƭ��ȡԲ��ʧ��" << endl;
		namedWindow("ʧ����Ƭ", 0);
		imshow("ʧ����Ƭ", viewGray);
		waitKey(0);
	}
	else
	{
		cornerSubPix(fore, image_points, Size(1, 1), Size(-1, -1), TermCriteria(TermCriteria::COUNT | TermCriteria::EPS, 30, 0.001));
		drawChessboardCorners(view, board_size, image_points, true);
	}
	imwrite("./result/result.jpg", view);

	return;
}