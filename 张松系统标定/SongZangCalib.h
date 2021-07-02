#pragma once

#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

#define CHESSBOARD 0
#define CIRCLEGRID 1

// 提取标定板特征点像素坐标，生成特征点世界坐标
bool findFeaturePoints( const vector<Mat>& calibImages,
					    vector<vector<Point3f>>& objPointsSeq,
					    vector<vector<Point2f>>& camPointsSeq,
						Size& imageSize,
						const Size& boardSize,
						const Size& squareSize,
						int blobFilter = 0,
					    int calibPlate = CIRCLEGRID);

// 相机标定
bool camCalibration( const vector<vector<Point3f>>& objPointsSeq,
					 const vector<vector<Point2f>>& camPointsSeq,
					 const Size& imageSize,
					 Mat& camMatrix,
					 Mat& distCoeffs,
					 vector<Mat>& vecsR,
					 vector<Mat>& vecsT,
					 int calibType = 0);

// 特征点坐标映射：相机cam-->投影仪proj
bool featCoordinateMap( const vector<Mat>& horizontalPhases,
					    const vector<Mat>& verticalPhases,
						const Size& projResolution,
						const vector<vector<Point2f>>& camPointsSeq,
						vector<vector<Point2f>>& projPointsSeq,
						int frq = 70);

// 投影仪逆相机标定
bool projCalibration( const vector<vector<Point3f>>& objPointsSeq,
					  const vector<vector<Point2f>>& projPointsSeq,
					  const Size& imageSize,
					  Mat& projMatrix,
					  Mat& distCoeffs,
					  vector<Mat>& vecsR,
					  vector<Mat>& vecsT);

// 相机、投影仪立体双目标定
bool steroCalibration(  const vector<vector<Point3f>>& objPointsSeq,
						const vector<vector<Point2f>>& camPointsSeq,
						const vector<vector<Point2f>>& projPointsSeq,
						Mat& camMatrix, Mat& camCoeffs, 
						Mat& projMatrix, Mat& projCoeffs, 
						const Size& imageSize,
						Mat& vecR,
						Mat& vecT,
						Mat& vecE,
						Mat& vecF);

// 双目立体畸变校正，包含极线矫正，并非二维图像校正
bool undistortImg(const Mat &srcImage,
	Mat &dstImage,
	const Mat &camMatrix,
	const Mat &camCoeffs,
	const Mat &projMatrix,
	const Mat &projCoeffs,
	const Mat &vecR,
	const Mat &vecT);

// 三维重建：像素绝对相位-->物点世界坐标系
bool reconstruction( const Mat& absPhase,
					 const Mat& camMatrix,
					 const Mat& projMatrix, 
					 const Mat& vecR,
					 const Mat& vecT, 
					 const Size& projResolution,
					 Mat& dstImage,
				     int frq = 70,
					 int projMode = 0);

// 保存张氏标定参数
bool writeZhangCalibFiles(	const string& fileName, 
							const double& reprojErr,
							const Mat& intrinsicMatrix,
							const Mat& distCoeffs,
							const vector<Mat>& vecsR,
							const vector<Mat>& vecsT);

// 保存双目立体标定参数
bool writeStereoCalibFiles( const string& fileName, 
							const double& reprojErr, 
							const Mat& camMatrix, 
							const Mat& camCoeffs, 
							const Mat& projMatrix,  
							const Mat& projCoeffs, 
							const Mat& vecR, 
							const Mat& vecT);

//双线性插值
float biInterpolate(const Mat& srcImg, float i, float j);

// 张松系统标定配置参数类型转换
bool vecsMat2vecsFloat(const vector<Mat>& sysParameters,
					   vector<float>& sysFloatParams);

//系统标定总函数
bool SongZhangCalibration(  const vector<Mat>& camCalibImg, 
							const vector<Rect>& imgRectVec,
							const vector<vector<Mat>>& projCalibImg, 
							vector<float>& sysParams);