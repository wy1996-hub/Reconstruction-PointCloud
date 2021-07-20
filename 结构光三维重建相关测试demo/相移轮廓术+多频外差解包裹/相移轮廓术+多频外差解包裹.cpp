#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

//�ο����ģ�
//���ڶ�Ƶ����˫Ŀ��ά�ع��붨λ���о�_���Ľ�
//�������ֹ�դͶӰ�Ľṹ����ά����������ϵͳ�о�_����ΰ

Mat subtractPhase(const Mat& src1, const Mat& src2)
{
	assert(!src1.empty() && !src2.empty());

	Mat diffPhase = Mat::zeros(src1.size(), CV_32F);
	for (size_t i = 0; i < src1.rows; i++)
	{
		const float* ptr1 = src1.ptr<float>(i);
		const float* ptr2 = src2.ptr<float>(i);
		float* ptrPhase = diffPhase.ptr<float>(i);

		for (size_t j = 0; j < src1.cols; j++)
		{
			if (ptr1[j] > ptr2[j])
				ptrPhase[j] = ptr1[j] - ptr2[j];
			else
				ptrPhase[j] = static_cast<float>(ptr1[j] - ptr2[j] + CV_2PI);
				//ptrPhase[j] = ptr1[j] - ptr2[j] + CV_2PI * (int)((ptr2[j] - ptr1[j]) / CV_2PI);//ֱ�Ӽ�2PI���������Щȡ�������������������
		}
	}
	return diffPhase;
}

Mat unwrapByTime(const Mat& keyPhase, const Mat& wrappedPhase, float multiple)
{
	//keyPhase--���չ����ľ�����λͼ����һ��Կ��
	//wrappedPhase--������λͼ

	assert(!wrappedPhase.empty() && !keyPhase.empty());

	const int width = wrappedPhase.cols;
	const int height = wrappedPhase.rows;
	Mat unwrappedPhase = Mat::zeros(wrappedPhase.size(), CV_32F);

	//tan(alpha)/tan(belta)�ı�ֵ������ͶӰͼ�����������ֵ��Ϊ����
	for (size_t i = 0; i < height; i++)
	{
		const float* ptrKeyPhase = keyPhase.ptr<float>(i);
		const float* ptrWrappedPhase = wrappedPhase.ptr<float>(i);
		float* ptrUnwrappedPhase = unwrappedPhase.ptr<float>(i);

		for (size_t j = 0; j < width; j++)
		{
			float period = static_cast<float>(round((multiple * ptrKeyPhase[j] - ptrWrappedPhase[j]) / CV_2PI));
			ptrUnwrappedPhase[j] = static_cast<float>(ptrWrappedPhase[j] + period * CV_2PI);
		}
	}

	return unwrappedPhase;
}

vector<Mat> calWrappedPhase(const vector<vector<Mat>>& image)
{
	assert(image.size() != 0);

	const int width = image[0][0].cols;
	const int height = image[0][0].rows; 
	const int frqSz = 3;
	vector<Mat> wrappedPhaseVec(frqSz);

	for (size_t i = 0; i < frqSz; i++)
	{
		Mat img1 = image[i][0];
		Mat img2 = image[i][1];
		Mat img3 = image[i][2];
		Mat img4 = image[i][3];
		Mat wrappedPhase = Mat::zeros(height, width, CV_32F);
		for (size_t m = 0; m < height; m++)
		{
			float* ptr1 = img1.ptr<float>(m);
			float* ptr2 = img2.ptr<float>(m);
			float* ptr3 = img3.ptr<float>(m);
			float* ptr4 = img4.ptr<float>(m);
			float* ptrPhase = wrappedPhase.ptr<float>(m);
			for (size_t n = 0; n < width; n++)
			{
				// atan2ֵ��-pi~pi(2pi�ĸ�����)
				if (ptr3[n] != ptr1[n])
				{
					ptrPhase[n] = atan2((ptr4[n] - ptr2[n]), (ptr1[n] - ptr3[n]));
					if (ptrPhase[n] < 0) 
						ptrPhase[n] += CV_2PI;
				}
				else if (ptr2[n] == ptr4[n]) 
					ptrPhase[n] = 0;
				else if (ptr2[n] > ptr4[n]) 
					ptrPhase[n] = 3.0 * CV_PI / 2.0;
				else 
					ptrPhase[n] = CV_PI / 2.0;
			}
		}
		wrappedPhaseVec[i] = wrappedPhase;
	}
	return wrappedPhaseVec;
}

Mat fringe2AbsPhase(const vector<Mat>& image)
{
	assert(image.size() == 12);

	//����ͼ��Ԥ����
	const int frqSz = 3;
	const int phaseSz = 4;
	vector<vector<Mat>> multiFrq(frqSz, vector<Mat>(phaseSz));
	for (size_t i = 0; i < frqSz; i++)
	{
		for (size_t j = 0; j < phaseSz; j++)
		{
			Mat tmp = image[i * 4 + j].clone();
			tmp.convertTo(tmp, CV_32F, 1.0f / 255.0f);
			GaussianBlur(tmp, tmp, Size(5, 5), 3); //��˹��ͨ�˲�
			normalize(tmp, tmp, 1.0, 0.0, NORM_MINMAX);//��һ��
			multiFrq[i][j] = tmp;
		}
	}

	vector<Mat> wrappedPhase;//�洢���Ų�ͬƵ�ʵİ�����λͼ
	vector<Mat> diffPhase(3);//�洢���ϳ��е�����ͼ
	vector<Mat> unwrappedPhase(2);//�洢���չ�������е�����ͼ

	// 1 ������λ����
	wrappedPhase = calWrappedPhase(multiFrq);

	// 2 ���ϳ�
	diffPhase[0] = subtractPhase(wrappedPhase[0], wrappedPhase[1]);//phase_12
	diffPhase[1] = subtractPhase(wrappedPhase[1], wrappedPhase[2]);//phase_23
	diffPhase[2] = subtractPhase(diffPhase[0], diffPhase[1]);//phase_123

	// 3 ���չ�� : phase_123->phase_12->phase_1  &&  1->1/6->1/70 && phase_123����ֻ��һ�����ڣ���˾���ȫ���ľ�����λ
	unwrappedPhase[0] = unwrapByTime(diffPhase[2], diffPhase[0], 6.0f);//123����+12����->12����
	unwrappedPhase[1] = unwrapByTime(unwrappedPhase[0], wrappedPhase[0], 35.0f / 3.0f);//12����+1����->1����

	// 4 ȥ����λ2PIͻ��
	const int height = unwrappedPhase[1].rows;
	const int width = unwrappedPhase[1].cols;
	for (size_t i = 0; i < height; i++)
	{
		float* pPhase = unwrappedPhase[1].ptr<float>(i);
		for (size_t j = 0; j < width - 1; j++)
		{
			if (pPhase[j + 1] - pPhase[j] > CV_2PI)
				pPhase[j + 1] -= CV_2PI;
		}
	}

	return unwrappedPhase[1];
}

Mat PSP4(const vector<Mat>& image)
{
	assert(image.size() == 4);

	const int width = image[0].cols;
	const int height = image[0].rows;
	Mat Phase = Mat::zeros(height, width, CV_32F);

	for (size_t i = 0; i < height; i++)
	{
		const float* ptr1 = image[0].ptr<float>(i);
		const float* ptr2 = image[1].ptr<float>(i);
		const float* ptr3 = image[2].ptr<float>(i);
		const float* ptr4 = image[3].ptr<float>(i);
		float* ptrPhase = Phase.ptr<float>(i);
		for (size_t j = 0; j < width; j++)
		{
			if (ptr3[j] != ptr1[j])
			{
				ptrPhase[j] = atan2((ptr4[j] - ptr2[j]), (ptr1[j] - ptr3[j]));
				if (ptrPhase[j] < 0)
					ptrPhase[j] += CV_2PI;
			}
			else if (ptr2[j] == ptr4[j])
				ptrPhase[j] = 0;
			else if (ptr2[j] > ptr4[j])
				ptrPhase[j] = 3.0 * CV_PI / 2.0;
			else
				ptrPhase[j] = CV_PI / 2.0;
		}
	}
	return Phase;
}

void main()
{
#if 1 
	const int imgNum = 12;
	vector<Mat> image(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		string pathName = "./test/0 (" + to_string(i + 2) + ").bmp";
		image[i] = imread(pathName, IMREAD_GRAYSCALE);
	}
	Mat res = fringe2AbsPhase(image);

	//normalize��convertTo���������ž���Ĺ��ܣ������в���
	normalize(res, res, 0.0, 255.0, NORM_MINMAX);//��Ҫ�������������������Ҫ��һ����Χ��--�ı�����--�����ݱ���ķ�Χ
	res.convertTo(res, CV_8U);//�ı�ͨ�����--�����ݿ��Գ��ֵķ�Χ��һ������ڵ������ݱ���ķ�Χ.
	//convertTo--Ī���Ų��������scale=1��shift=0���򲻽��б�������
	imwrite("./testmultifrq.jpg", res);
#endif

#if 0
	const int imgNum = 4;
	vector<Mat> image(imgNum);
	for (size_t i = 0; i < imgNum; i++)
	{
		string pathName = "I:/project/��ά�ؽ�/����ϵͳ�궨/Image/0524/650/1/0 (" + to_string(i + 2) + ").bmp";
		image[i] = imread(pathName, IMREAD_GRAYSCALE);
		image[i].convertTo(image[i], CV_32F, 1.0f / 255.0f);
		GaussianBlur(image[i], image[i], Size(5, 5), 3); //��˹��ͨ�˲�
		normalize(image[i], image[i], 1.0, 0.0, NORM_MINMAX);//��һ��
	}
	Mat res = PSP4(image);
	normalize(res, res, 0.0, 255.0, NORM_MINMAX);
	res.convertTo(res, CV_8U);
	imwrite("./testpsp.jpg", res);
#endif
	
	waitKey(0);
	return;
}