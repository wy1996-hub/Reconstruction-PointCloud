#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;
//------------------------------------------------------------------------------------------------//

#include <pcl\keypoints\uniform_sampling.h>
//均匀采样
void sampleUniform(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudSampled)
{
	//构建指定半径的球体，将每一个求内距离球体中心最近的点作为下采样后的输出点
	//提速滤波（下采样）建立一个立方体；均匀采样建立一个球
	//下采样的幅度很大

	pcl::UniformSampling<pcl::PointXYZ> us;
	us.setInputCloud(cloud);
	us.setRadiusSearch(0.005f);
	us.filter(*cloudSampled);
}

#include <pcl\filters\random_sample.h>
//随机采样
void sampleRandom(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudSampled)
{
	//最简单，直接设置固定采样点个数

	pcl::RandomSample<pcl::PointXYZ> rs;
	rs.setInputCloud(cloud);
	rs.setSample(3000);//设置素随机采样点的个数
	rs.filter(*cloudSampled);
}

#include <pcl\filters\normal_space.h>//法线空间采样
#include <pcl\features\normal_3d_omp.h>//多线程加速
//计算法向量
void estimateNormals(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::Normal>::Ptr& cloudNormals)
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setSearchMethod(tree);
	n.setNumberOfThreads(6);
	n.setKSearch(30);
	n.compute(*cloudNormals);
}
//法线空间采样
void sampleNormalSpace(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudSampled)
{
	//在法向量空间内均匀随机采样，使所选点之间的法线分布尽可能大
	//结果表现为：地物特征变化大的地方剩余点较多，变化小的地方剩余点稀少；可有效保证地物特征
	//某些场景（切割平面数据集），模型的小特征对于确定正确的对齐至关重要
	//例如随机抽样，其策略通常只会在特征点集中选择几个样本，会导致某些部分无法确定正确刚体变换
	//根据角度空间中的法线位置来存储点，然后尽可能均匀的在这些存储点上采样
	//是使用地表特征进行对齐的；与传统基于特征的方法相比，具有较低的计算成本，但鲁棒性较差

	//计算法向量(封装成了一个函数，但是也可以不封装)
	pcl::PointCloud<pcl::Normal>::Ptr cloudNormals(new pcl::PointCloud<pcl::Normal>);
	estimateNormals(cloud, cloudNormals);
	//采样
	pcl::NormalSpaceSampling<pcl::PointXYZ, pcl::Normal> nss;
	nss.setInputCloud(cloud);// 设置输入点云
	nss.setNormals(cloudNormals);// 设置在输入点云上计算的法线
	nss.setBins(2, 2, 2);// 设置x,y,z方向bins的个数
	nss.setSeed(0); // 设置种子点
	nss.setSample(2000); // 设置要采样的索引数。
	nss.filter(*cloudSampled);
}

#include <pcl\filters\sampling_surface_normal.h>
//索引空间采样
void sampleSurfaceNormal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointNormal>::Ptr& cloudSampled)
{
	//将输入空间划分为网格，直至每个网格中包含最多N个点，并在每个网格中随机采样
	//使用每个网格中的N个点来计算Normal,网格内的所有采样点都被赋予了相同的法线
	//pcl::pointnormal这种数据类型，在normalestimation中无法做到直接得到包含xyz和normal的有效输出
	//因此需要先计算pcl::normal，然后把xyz与normal合并（存在另一种构建pcl::pointnormal的方法）

	// 1 法线估计
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(20);
	n.compute(*normals);

	// 2 连接XYZ和法向量字段--即合并xyz与normal
	pcl::PointCloud<pcl::PointNormal>::Ptr cloudNormals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*cloud, *normals, *cloudNormals);

	// 3 采样
	pcl::SamplingSurfaceNormal <pcl::PointNormal> ssn;
	ssn.setInputCloud(cloudNormals);
	ssn.setSample(10);     // 设置每个网格的最大样本数 n
	ssn.setRatio(0.1f);    // 设置采样率&
	ssn.filter(*cloudSampled);// 输出的点云是用每个网格里的最大样本数n x &
}

#include <pcl\surface\mls.h>
#include <pcl\search\kdtree.h>//单独使用kdtree的头文件（方法之一）
//MLS点云上采样
void sampleMLS(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudSampled)
{
	//上采样是一种表面重建方法：当点云数据少于预期，通过内插拥有的点云数据，恢复原来的表面
	//复杂的猜想假设，结果并不准确；因此在点云下采样时，一定要保存一份原始数据

	// 1 创建上采样对象
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ> up;//该类实现了一个基于移动最小二乘的点云上采样
	up.setInputCloud(cloud);

	// 2 建立搜索对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;

	// 3 设置采样
	up.setSearchMethod(tree);	
	up.setSearchRadius(0.1);//设置搜索邻域的半径
	up.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointXYZ>::SAMPLE_LOCAL_PLANE);
	up.setUpsamplingRadius(0.04);//设置采样的半径	
	up.setUpsamplingStepSize(0.02);//采样步长的大小
	up.process(*cloudSampled);
}
//同时进行上采样和计算法向量
void sampleMLSEx(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointNormal>::Ptr& cloudSampled)
{
	// 1 创建上采样对象
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> up;//该类实现了一个基于移动最小二乘的点云上采样
	up.setInputCloud(cloud);

	// 2 建立搜索对象
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree;

	// 3 设置采样
	up.setSearchMethod(tree);
	up.setSearchRadius(0.1);//设置搜索邻域的半径
	up.setComputeNormals(true);//计算法向量
	up.setPolynomialFit(true);
	up.setUpsamplingMethod(pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal>::SAMPLE_LOCAL_PLANE);
	up.setUpsamplingRadius(0.04);
	up.setUpsamplingStepSize(0.02);
	up.setPolynomialFit(2);  //MLS拟合的阶数，默认是2
	//up.setProjectionMethod(pcl::MLSResult::ProjectionMethod::SIMPLE);
	//up.setNumberOfThreads(6);
	up.setPointDensity(100);
	up.process(*cloudSampled);
}


//------------------------------------------------------------------------------------------------//

//显示点云--双窗口
void visualizeSampledCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudSampled)
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
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildSampledColor(cloudSampled, "z");
	view->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "Raw point clouds", v1);
	view->addPointCloud<pcl::PointXYZ>(cloudSampled, fildSampledColor, "sampled point clouds", v2);
	//设置点云的渲染属性,string &id = "cloud"相当于窗口ID
	//按照z字段进行渲染，则不需要单独渲染每个点云颜色
	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Raw point clouds", v1);//设置点云颜色
	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "sampled point clouds", v2);
	while (!view->wasStopped())
	{
		view->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//------------------------------------------------------------------------------------------------//

int maininstance2()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudSampled(new pcl::PointCloud<pcl::PointXYZ>);

	string cloudPath = "../bunny.pcd";
	if (pcl::io::loadPCDFile(cloudPath, *cloud) != 0)
		return -1;
	cout << "加载点云数量：" << cloud->points.size() << "个" << endl;

	//------------------------------函数调用---------------------------------
	//均匀采样
	//sampleUniform(cloud, cloudSampled);
	//随机采样
	//sampleRandom(cloud, cloudSampled);
	//法线空间采样
	//sampleNormalSpace(cloud, cloudSampled);
	//索引空间采样
	//sampleSurfaceNormal(cloud, cloudSampled);
	//MLS点云上采样
	//sampleMLS(cloud, cloudSampled);
	//sampleMLSEx(cloud, cloudSampled);
	//----------------------------------------------------------------------

	cout << "滤波点云数量：" << cloudSampled->points.size() << "个" << endl;
	visualizeSampledCloud(cloud, cloudSampled);

	return 0;
}