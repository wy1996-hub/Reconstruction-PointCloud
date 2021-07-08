#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;
//------------------------------------------------------------------------------------------------//

#include <pcl\filters\crop_hull.h>
#include <pcl\surface\concave_hull.h>
//CropHull任意二维多边形内部点云提取
void cutCropHull(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//一种滤波器，定义一个二维图形，然后裁剪位于图形之内的点云

	// 1 为了构造2D封闭多边形，首先输入2D平面点云，这些平面点是2D封闭多边形的顶点
	pcl::PointCloud<pcl::PointXYZ>::Ptr boundingbox(new pcl::PointCloud<pcl::PointXYZ>);
	boundingbox->push_back(pcl::PointXYZ(0.1, -0.1, 0));
	boundingbox->push_back(pcl::PointXYZ(-0.1, 0.1, 0));
	boundingbox->push_back(pcl::PointXYZ(-0.1, -0.1, 0));
	boundingbox->push_back(pcl::PointXYZ(0.1, 0.1, 0));

	// 2 对上述2D平面点构造凸包--保存凸包(封闭多边形)顶点与凸包形状
	pcl::ConvexHull<pcl::PointXYZ> hull;//创建凸包对象
	hull.setInputCloud(boundingbox);//设置输入点云
	hull.setDimension(2);//设置凸包维度
	vector<pcl::Vertices> polygons;//设置pcl:Vertices类型的向量，用于保存凸包顶点
	pcl::PointCloud<pcl::PointXYZ>::Ptr surfaceHull(new pcl::PointCloud<pcl::PointXYZ>);//该点云用于描述凸包形状
	hull.reconstruct(*surfaceHull, polygons);//计算2D凸包结果

	// 3 创建CropHull对象，滤波得到2D封闭凸包范围内的点云，此处的维度需要与输入凸包维度一致
	pcl::CropHull<pcl::PointXYZ> bb_filter;//创建CropHull对象
	bb_filter.setDim(2);                 
	bb_filter.setInputCloud(cloud);      
	bb_filter.setHullIndices(polygons);//输入封闭多边形的顶点
	bb_filter.setHullCloud(surfaceHull);//输入封闭多边形的形状
	bb_filter.filter(*cloudCut);
}

#include <pcl\filters\crop_box.h>
//CropBox立体框内部点云数据提取
void cutCropBox(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//一种滤波器，定义一个三维立体图形，允许用户过滤给定框内所有数据

	pcl::CropBox<pcl::PointXYZ> cb;
	cb.setInputCloud(cloud);	
	cb.setMin(Eigen::Vector4f(-0.1f, -0.1, -0.1, -0.1));//设定立体空间数据
	cb.setMax(Eigen::Vector4f(0.1f, 0.1, 0.1, 0.1));
	cb.setKeepOrganized(false);//如果希望能够提取被删除点的索引，则设置为true
	cb.setUserFilterValue(0.1f);//提供一个被过滤的点应该设置为的值，而不是删除它们,与setKeepOrganized联用
	cb.filter(*cloudCut);
}

#include <pcl\filters\local_maximum.h>
//LocalMaximum消除局部最大的点
void cutLocalMax(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//分析每一个点，并在z方向删除那些相对于他们的邻居（通过半径搜索找到）局部最大的点

	//一种构造点云数据的方式
	//pcl::PointCloud<pcl::PointXYZ> cloud_in, cloud_out;
	//cloud_in.height = 1;
	//cloud_in.width = 3;
	//cloud_in.is_dense = true;
	//cloud_in.resize(4);
	//cloud_in[0].x = 0;    cloud_in[0].y = 0;    cloud_in[0].z = 0.25;
	//cloud_in[1].x = 0.25; cloud_in[1].y = 0.25; cloud_in[1].z = 0.5;
	//cloud_in[2].x = 0.5;  cloud_in[2].y = 0.5;  cloud_in[2].z = 1;
	//cloud_in[3].x = 5;    cloud_in[3].y = 5;    cloud_in[3].z = 2;
	//pcl::LocalMaximum<pcl::PointXYZ> lm;
	//lm.setInputCloud(cloud_in.makeShared());

	pcl::LocalMaximum<pcl::PointXYZ> lm;
	lm.setInputCloud(cloud);
	lm.setRadius(1.0f);//设置用于确定一个点是否为局部最大值的半径
	lm.filter(*cloudCut);//调用滤波方法并返回滤波后的点云
}

#include <pcl\filters\grid_minimum.h>
//GridMinimum获取栅格最低点
void cutGridMin(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//点云体素化是一种比较普遍的点云处理方式
	//voxelgrid划分体素栅格，求取栅格内点云重心
	//approximatevoxelgrid划分体素栅格，求取栅格内点云中心
	//GridMinimum将三维点云在xy平面划分网格，寻找每个网格中最小的z点云代表该网格
	//可以有效过滤高程信息间断的情况，例如车停在树下

	float resolution = .001f;//分辨率越小网格越密集，分辨率越大网格越稀疏
	pcl::GridMinimum<pcl::PointXYZ> gm(resolution);//GridMinimum类不存在默认构造函数
	gm.setInputCloud(cloud);
	gm.filter(*cloudCut);
}

//------------------------------------------------------------------------------------------------//

//显示点云--双窗口
void visualizeCutCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//PCL可视化工具
	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer("ShowClouds"));
	int v1 = 0, v2 = 0;
	view->createViewPort(.0, .0, .5, 1., v1);
	view->setBackgroundColor(0., 0., 0., v1);
	view->addText("Raw point clouds", 10, 10, "text1", v1);
	view->createViewPort(.5, .0, 1., 1., v2);
	view->setBackgroundColor(0., 0., 0., v2);
	view->addText("sampled point clouds", 10, 10, "text2", v2);

	//按照z字段进行渲染
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud, "z");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildCutColor(cloudCut, "z");
	view->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "Raw point clouds", v1);
	view->addPointCloud<pcl::PointXYZ>(cloudCut, fildCutColor, "sampled point clouds", v2);
	//设置点云的渲染属性,string &id = "cloud"相当于窗口ID
	//按照z字段进行渲染，则不需要单独渲染每个点云颜色
	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Raw point clouds", v1);//设置点云颜色
	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "sampled point clouds", v2);
	view->addCoordinateSystem(.1);
	while (!view->wasStopped())
	{
		view->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//------------------------------------------------------------------------------------------------//

int main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudCut(new pcl::PointCloud<pcl::PointXYZ>);

	string cloudPath = "../bunny.pcd";
	if (pcl::io::loadPCDFile(cloudPath, *cloud) != 0)
		return -1;
	cout << "加载点云数量：" << cloud->points.size() << "个" << endl;

	//------------------------------函数调用---------------------------------

	//CropHull任意多边形内部点云提取
	//cutCropHull(cloud, cloudCut);
	//CropBox立体框内部点云数据提取
	//cutCropBox(cloud, cloudCut);
	//LocalMaximum消除局部最大的点
	//cutLocalMax(cloud, cloudCut);
	//GridMinimum获取栅格最低点
	cutGridMin(cloud, cloudCut);

	//----------------------------------------------------------------------

	cout << "滤波点云数量：" << cloudCut->points.size() << "个" << endl;
	visualizeCutCloud(cloud, cloudCut);

	return 0;
}