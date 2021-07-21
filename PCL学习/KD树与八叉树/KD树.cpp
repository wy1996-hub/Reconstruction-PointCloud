#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>//kdtree近邻搜索
#include <pcl/io/pcd_io.h>  //文件输入输出
#include <pcl/point_types.h>  //点类型相关定义
#include <pcl/visualization/pcl_visualizer.h>//可视化相关定义
#include <boost/thread/thread.hpp>

using namespace std;

/*
	KD-tree：一种数据结构，用来组织具有K维的空间中的若干个点，是一个具有其他约束的二进位搜索树
	对于1维数据的查询，使用平衡二叉树建立索引即可；KD-tree则是一种高纬数据的快速查询结构,一种针对多维数据的类似一维的索引方法
	构建KD树是针对高维数据，需要针对每一维都进行二分
	depth表示当前是KD树的第几层，如果depth是偶数，通过纵向线对集合进行划分；如果depth是奇数，通过横向线进行划分
	kd树对于范围（区间）搜索好而最近邻搜索非常有用
	一般只处理三维点云，所以kd树都是三维的
	kd树的查找时间复杂度（nlogn）：通过垂直于点云的一维超平面，将空间递归分割为多个子空间实现点云数据快速检索

	三维点云中kd-tree的详细计算过程：（找中位数构建分割平面）
	1、依据点云的全局坐标系建立包含所有点云的立方体包围盒
	2、对每个包含超过1个点的立方体，构建分割平面
	3、两个分割字空间和分割平面上的点构成分支与连接点
	4、分割字空间，若果内部的数量超过1，则执行步骤2继续分割
*/

//PCL KD树的使用
void kdtreeUSE()
{
	// 1、首先用系统时间为rand()种子，然后用随机数据创建并填充PointCloud
	srand(time(nullptr));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->height = 1;//创建无序点云
	cloud->width = 1000;
	cloud->points.resize(cloud->height * cloud->width);
	for (size_t i = 0; i < cloud->size(); i++)
	{
		cloud->points[i].x = 1024.f*rand() / (RAND_MAX + 1.f);
		(*cloud)[i].y = 1024.f*rand() / (RAND_MAX + 1.f);
		(*cloud)[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}
	// 2、创建kdtree对象，并将随机创建的云设置为输入点云。
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	// 3、然后指定一个随机坐标作为“搜索点”
	pcl::PointXYZ searchPoint;  
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	// 4、创建一个整数(并将其设置为10)和两个向量，用于从搜索中存储搜索点的K近邻
	int K = 10;                                   // 需要查找的近邻点个数
	std::vector<int> pointIdxKNNSearch(K);        // 保存每个近邻点的索引
	std::vector<float> pointKNNSquaredDistance(K);// 保存每个近邻点与查找点之间的欧式距离平方
	std::cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;
	// 5、打印出随机“搜索点”的所有10个最近邻居的位置，这些位置已经存储在先前创建的向量中
	if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)// > 0表示能够找到近邻点， = 0表示找不到近邻点
	{
		for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxKNNSearch[i]].x
			<< " " << (*cloud)[pointIdxKNNSearch[i]].y
			<< " " << (*cloud)[pointIdxKNNSearch[i]].z
			<< " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
	}
	// 6、创建一个随机半径和两个向量，用于从搜索中存储搜索点的K近邻
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f); // 查找半径范围
	std::vector<int> pointIdxRadiusSearch;              // 保存每个近邻点的索引
	std::vector<float> pointRadiusSquaredDistance;      // 保存每个近邻点与查找点之间的欧式距离平方
	std::cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;
	// 7、打印出随机“搜索半径”找到的点的位置
	if(kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxRadiusSearch[i]].x
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].y
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].z
			<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
}

//PCL addLine可视化K近邻
void kdtreeVisualNearestK()
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PCDReader reader;
	reader.read("../bunny-color.pcd", *cloud);//读取PCD文件
	cout << "PointCloud  has: " << cloud->points.size() << " data points." << endl;
	//创建kdtree对象，并将读取到的点云设置为输入。
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	//初始化搜索点
	pcl::PointXYZRGBA searchPoint;
	searchPoint.x = 0;
	searchPoint.y = 0;
	searchPoint.z = 0;
	//可视化 注：这里的可视化是创建可视化变量，并开辟一个窗口来展示图像。在后面的代码才会显示点云文件。
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer plane"));
	viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "sample cloud");
	//为了更直观的看到最近邻点，将所有的点与查询点连线，每条连线都需要有自己独立的ID，不能重复，所以定义字符流来创建不同的ID
	string lineId = "line";
	stringstream ss;//通过流来实现字符串和数字的转换
	//K nearest neighbor search
	int K = 10;
	vector<int> pointIdxNKNSearch(K);
	vector<float> pointNKNSquaredDistance(K);
	cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y << " " << searchPoint.z << ") with K=" << K << endl;
	if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
		{
			ss << i;// 将int类型的值放入输入流中
			lineId += ss.str();//可以使用 str() 方法，将 stringstream 类型转换为 string 类型
			//添加点与查询点之间的连线，数字表示添加线的着色
			viewer->addLine<pcl::PointXYZRGBA>(cloud->points[pointIdxNKNSearch[i]], searchPoint, 1, 58, 82, lineId);
			cout << "    " << cloud->points[pointIdxNKNSearch[i]]
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << endl;
			ss.str("");//直接用clear没有效果，clear() 方法适用于进行多次数据类型转换的场景
			lineId = "line";
		}
	}
	// Neighbors within radius search
	vector<int> pointIdxRadiusSearch;
	vector<float> pointRadiusSquaredDistance;
	float radius = 0.05;
	cout << "Neighbors within radius search at (" << searchPoint.x << " " << searchPoint.y
		<< " " << searchPoint.z << ") with radius=" << radius << endl;
	if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (size_t i = 0; i < pointIdxRadiusSearch.size(); ++i) {
			ss << i;
			lineId += ss.str();
			viewer->addLine<pcl::PointXYZRGBA>(cloud->points[pointIdxRadiusSearch[i]], searchPoint, 1, 58, 82, lineId);
			cout << "    " << cloud->points[pointIdxRadiusSearch[i]]
				<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << endl;
			ss.str("");
			lineId = "line";
		}
	}
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

#include <pcl/search/kdtree.h>
#include <pcl/registration/sample_consensus_prerejective.h>
//计算点云的平均密度
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	/*
		采样设备不同，设备距离场景远近不同，会使得点云密度产生差异
		现有点云密度估计方法：基于距离+基于分块
		基于距离：计算点云个点的距离平均值来估计点云分布疏密程度，某一点到据此点距离最近的点的距离
	*/

	double resolution = 0.0;
	int numberOfPoints = 0;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;//并非flann里面的kdtree
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{    
		//检查是否存在无效点
		//pcl数据处理时，很多算法会考虑无效点，主要通过判断pointcloud类中的数据成员是否包含nan
		//isFinite函数返回一个bool，检查某个值是不是正常数值。解决函数pcl::removeNaNFromPointCloud
		if (!pcl::isFinite(cloud->points[i]))
			continue;//skip nans

		//在同一个点云内进行k近邻搜索时，k=1的点为查询点本身。
		int nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;
	cout << "点云密度为：" << resolution << endl;

	return resolution;
}

#include <pcl/common/utils.h>
//计算点云的平均密度优化版
double computeCloudResolutionEx(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, float max_dist, int nr_threads)
{
	/*
		避免空洞区域对整体密度的影响
		在K近邻搜索时，加入距离约束条件，鲁棒性更强
		max_dist:被视为邻域的点的最大距离;
		nr_threads:要使用的线程数(默认值=1，仅在设置OpenMP标志时使用)
	*/

	const float max_dist_sqr = max_dist * max_dist;
	const std::size_t s = cloud->points.size();

	pcl::search::KdTree <pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	float mean_dist = 0.f;
	int num = 0;
	std::vector <int> ids(2);
	std::vector <float> dists_sqr(2);
	//--------------多线程加速开始-------------
	/*pcl::utils::ignore(nr_threads);
	#pragma omp parallel for \
	default(none) \
	shared(tree, cloud) \
	firstprivate(ids, dists_sqr) \
	reduction(+:mean_dist, num) \
	firstprivate(s, max_dist_sqr) \
	num_threads(nr_threads)*/
	//--------------多线程加速结束--------------
	for (int i = 0; i < 1000; i++)//随机采样1000个点
	{
		tree.nearestKSearch((*cloud)[rand() % s], 2, ids, dists_sqr);
		if (dists_sqr[1] < max_dist_sqr)//距离约束条件
		{
			mean_dist += std::sqrt(dists_sqr[1]);
			num++;
		}
	}
	cout << "点云密度为：" << (mean_dist / num) << endl;

	return (mean_dist / num);
}

#include <pcl/filters/extract_indices.h>
//删除点云中重叠的点
void deleteOverlappedPoints(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered)
{
	/*
		若某一点在某一距离阈值(0.000001)邻域内不止其本身一个点，则认为其有重复点。
		将重复点的索引记录下来，由于后续以此重复点为查询点搜索时，此时这一点也会被定义为重复点，
		但pointIdxRadiusSearch中都是升序排列的，故从pointIdxRadiusSearch中的第二个点的索引开始记录，
		这样可以保证仅仅删除重复的点，保留第一个点
	*/

	// 1 KD树半径搜索
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	vector<int> pointIdxRadiusSearch;//保存每个近邻点的索引
	vector<float> pointRadiusSquaredDistance;//保存每个近邻点与查找点之间的欧式距离平方
	vector<int> total_index;
	//若两点之间的距离为0.000001则认为是重合点
	float radius = 0.000001;

	// 2 对cloud中的每个点与邻域内的点进行比较
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		pcl::PointXYZ searchPoint = cloud->points[i];
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			if (pointIdxRadiusSearch.size() != 1)
			{
				for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)//从pointIdxRadiusSearch中的第二个点的索引开始记录
				{
					total_index.push_back(pointIdxRadiusSearch[j]);
				}
			}
		}
	}

	// 3 删除重复索引
	sort(total_index.begin(), total_index.end());//将索引进行排序
	total_index.erase(unique(total_index.begin(), total_index.end()), total_index.end());//将索引中的重复索引去除
	//根据索引删除重复的点
	pcl::PointIndices::Ptr outliners(new pcl::PointIndices());
	outliners->indices.resize(total_index.size());
	for (size_t i = 0; i < total_index.size(); i++)
	{
		outliners->indices[i] = total_index[i];
	}
	cout << "重复点云删除完毕！！！" << endl;

	// 4 提取删除重复点之后的点云
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(outliners);
	extract.setNegative(true);//设置为true则表示保存索引之外的点
	extract.filter(*cloud_filtered);
	cout << "原始点云中点的个数为：" << cloud->points.size() << endl;
	cout << "删除的重复点的个数为:" << total_index.size() << endl;
	cout << "去重之后点的个数为:" << cloud_filtered->points.size() << endl;
}

//------------------------------------------------------------------------------------------------//

//显示点云--双窗口
void visualizeKDCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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

void main_()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloud) == -1)
	{
		PCL_ERROR("Cloudn't read file!");
		return;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	//kdtree--nearestKSearch--radiusSearch
	//kdtreeUSE();
	//addLine可视化K近邻
	//kdtreeVisualNearestK();
	//计算点云的平均密度
	//double density = computeCloudResolution(cloud);
	//double density = computeCloudResolutionEx(cloud, .2f, 1);
	//删除点云中重叠的点
	deleteOverlappedPoints(cloud, cloud_filtered);

	visualizeKDCloud(cloud, cloud_filtered);
	return;
}