#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//�ο����ģ���դͶӰ˫Ŀ�Ӿ���ò������Ӱ�������о�_����ŵ������ѧ��
//������ʵ���õ��Ĺ�ǿ����Ϊ���ƺ�Ĳ���
//ʵ��Ӧ���У����ڱ���������������ع������Ӱ�죬�޷���֤������ǿ��ȫһ��
//ƽ���Ҷ�ֵ����֤ͬƵ�ʵ�����ͼ������ǿһ��

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

//�Լ��Ľ���ʹ��ƽ���˲���Ľ��ָ��������ǿ����

Scalar calMeanMat(const vector<Mat>& srcImage)
{
	const int imgNum = srcImage.size();
	Mat output = Mat::zeros(srcImage[0].size(), CV_8UC4);
	merge(srcImage, output);
	return mean(output);//������ͼ��ƽ������ֵ

	//Mat output = Mat::zeros(srcImage[0].size(), CV_32F);//������ͼ��ƽ������
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
	Mat output = Mat::zeros(srcImage[0].size(), CV_32F);//������ͼ��ƽ������
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
	vector<Mat> bgImage(imgNum);//������ǿ
	vector<Mat> fgImage(imgNum);//���ƹ�ǿ
	vector<Mat> resImage(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		string path = imgPath + "/0 (" + to_string(i + 2) + ").bmp";
		Mat src = imread(path, IMREAD_GRAYSCALE);
		srcImage[i] = src;
		Mat bg;
		medianBlur(src, bg, kSz);
		bgImage[i] = bg;//����
		fgImage[i] = src - bg;
	}

#if 0
	Scalar s = calMeanMat(bgImage);//�ֱ��ȡ�ĸ�ͨ���������ص�ƽ��ֵ
	float m_ = 0.f;
	for (size_t i = 0; i < imgNum; i++)
		m_ += s.val[i];
	m_ /= imgNum;//��ȡ�ĸ�ͨ���������������ƽ��ֵ
	for (size_t i = 0; i < imgNum; i++)
	{
		float val = s.val[i] - m_;//ͬƵ��ͼ�񱳾���ǿ��ֵ
		bgImage[i] -= val;//������ǿһ��
		resImage[i] = bgImage[i] + fgImage[i];
		imwrite(imgPath + "/" + to_string(i + 2) + ".bmp", resImage[i]);
	}
#endif

#if 1
	//ͬƵ��ͼ��ƽ������
	Mat bgMean = calMeanMat_(bgImage);
	imwrite("./bgMean.jpg", bgMean);
	for (size_t i = 0; i < imgNum; i++)
	{
		resImage[i] = bgMean + fgImage[i];
		imwrite(imgPath + "/" + to_string(i + 2) + "test.bmp", resImage[i]);
	}
#endif

	//������ʹ��ƽ������ֵ����ƽ������ͼ��
	//��Ҫ����д����ʱ��ϸ�Ա�����һ��

}


void main()
{
	//BGLightCorrect1("./fringePattern", 4, Rect(215, 832, 740, 1068));
	BGLightCorrect2("./fringePattern", 4);

	waitKey(0);
	destroyAllWindows();
	return;
}