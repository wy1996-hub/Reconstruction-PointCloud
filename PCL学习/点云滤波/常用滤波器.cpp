#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;

// 需要进行滤波的4种情况
// 1 点云数据密度不规则
// 2 离群点(比如由于遮挡等原因噪声的)
// 3 下采样
// 4 噪声
//------------------------------------------------------------------------------------------------//

#include <pcl/filters/passthrough.h>
//直通滤波器
void filterPassThrough(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//过滤掉在指定维度方向上取值不在给定值域内的点
	//主要用来在初始步骤做去除偏差较大的离群点操作
	pcl::PassThrough<pcl::PointXYZ> pass;
	pass.setInputCloud(cloud);
	pass.setFilterFieldName("z"); //滤波字段名被设置为Z轴方向
	pass.setFilterLimits(0, 1); //设置在过滤方向上的过滤范围
	pass.setKeepOrganized(true); //保持有序点云结构
	pass.setNegative(true); //设置保留范围内的还是过滤掉范围内的，标志为false时保留范围内的点
	pass.filter(*cloudFiltered);
}

#include <pcl/filters/voxel_grid.h>
//体素滤波器
void filterVoxelGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//点云下采样，同时不破坏点云几何结构，保存形状特征
	//随机下采样效率高，但会破坏点云微观结构
	//用体素重心逼近，表示更为准确
	//以小珊格的重心点云坐标代替小栅格内其他所有点云的坐标
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(.01f, .01f, .01f);
	vg.filter(*cloudFiltered);
}
#include <pcl\filters\approximate_voxel_grid.h>
void filterApproximateVoxelGrid(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//用体素中心逼近，基于哈希表，针对数量巨大的场景速度会快的多
	//以小珊格的中心点云坐标代替小栅格内其他所有点云的坐标
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> avg;
	avg.setInputCloud(cloud);
	avg.setLeafSize(.005f, .005f, .005f);//设置体素栅格叶大小,数值越小点云越密
	avg.filter(*cloudFiltered);
}

#include <pcl\kdtree\kdtree_flann.h>
/*error C2079: “pcl::KdTreeFLANN::param_radius_”使用未定义的 struct“flann::SearchParams”*/
/*opencv与pcl同时加载flann会出现冲突问题，PCLflann有关配置放在与opencv有关内容的前面*/
//改进的体素滤波器
void filterVoxelGridEx(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//使用体素中所有点的重心位置代表整个体素，该重心点不一定是原始点云中的点
	//可能会损失原始点云的细小特征
	//用原始点云中距离体素重心最近的点代替体素重心点，降维的同时，提高点云数据的表达准确性
	pcl::VoxelGrid<pcl::PointXYZ> vg;
	vg.setInputCloud(cloud);
	vg.setLeafSize(.003f, .003f, .003f);
	pcl::PointCloud<pcl::PointXYZ>::Ptr voxelCloud(new pcl::PointCloud<pcl::PointXYZ>);
	vg.filter(*voxelCloud);
	//k近邻搜索
	//根据下采样的结果，选择采样点最近的点作为最终的下采样点
	pcl::KdTreeFLANN<pcl::PointXYZ> kdTree;
	kdTree.setInputCloud(cloud);
	pcl::PointIndicesPtr inds = boost::shared_ptr<pcl::PointIndices>(new pcl::PointIndices());//采样后根据最邻近点提取的样本点下标索引
	for (size_t i = 0; i < voxelCloud->points.size(); i++)
	{
		pcl::PointXYZ pt = voxelCloud->points[i];
		int k = 1;//最近邻搜索
		vector<int> pointIdxKNNSearch(k);
		vector<float> pointKNNSquaredDistance(k);
		//从kdTree（setInputCloud(cloud)）中找最靠近体素滤波后的点云
		if (kdTree.nearestKSearch(pt, k, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)
			inds->indices.emplace_back(pointIdxKNNSearch[0]);
	}
	pcl::copyPointCloud(*cloud, inds->indices, *cloudFiltered);
}

#include <pcl\filters\statistical_outlier_removal.h>
//统计滤波器
void filterStddevMulThresh(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//激光扫描会产生密度不均匀的点云数据；测量中的误差也会产生稀疏的离群点
	//密度不均、稀疏离群点会导致估计局部点云特征时运算复杂，出现错误，影响配准
	//统计滤波器：去除明显离群点（离群点一般由测量噪声引入，在空间中分布稀疏）
	//每个点都表达一定的信息量，某个区域点越密集则可能信息量越大；噪声属于无用信息，信息量越小，所以离群点表达的信息可以忽略不计
	//定义点云密度：计每个点到其最近的K个点的平均距离；点云中所有点的距离应该构成高斯分布，通过给定的均值与方差剔除离群点
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setMeanK(50);//检测每个点周围的50个邻近点
	sor.setStddevMulThresh(2);//2倍的标准差
	sor.filter(*cloudFiltered);
}

#include <pcl\filters\radius_outlier_removal.h>
//半径滤波器
void filterRadius(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//设定每个点云数据一定半径范围d内至少有足够多的近邻n，不满足就删除该点
	//需要首先对所有点云数据构造KD树
	//半径滤波器比统计滤波器更简单粗暴，但速度快，依序迭代留下的点一定是最密集的
	pcl::RadiusOutlierRemoval<pcl::PointXYZ> ror;
	ror.setInputCloud(cloud);
	ror.setRadiusSearch(0.003);//搜索半径0.003m内的邻近点
	ror.setMinNeighborsInRadius(20);//小于20个就删除
	ror.filter(*cloudFiltered);
}

#include <pcl\filters\conditional_removal.h>
//条件滤波器
void filterCondition(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//过滤用户自己设定的满足特定条件的数据

	// 1 定义条件滤波器所要使用的条件
	// 1.1 字段条件
	pcl::ConditionAnd<pcl::PointXYZ>::Ptr rangeCond(new pcl::ConditionAnd<pcl::PointXYZ>);//实例化条件指针
	rangeCond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
		pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::GT, 0)));//添加在z字段上大于(pcl::ComparisonOps::GT greater than)阈值的算子
	rangeCond->addComparison(pcl::FieldComparison<pcl::PointXYZ>::ConstPtr(new
		pcl::FieldComparison<pcl::PointXYZ>("z", pcl::ComparisonOps::LT, 15.0)));//小于阈值
	// 1.2 曲率条件
	pcl::ConditionOr<pcl::PointNormal>::Ptr curvatureCond(new pcl::ConditionOr<pcl::PointNormal>);
	curvatureCond->addComparison(pcl::FieldComparison<pcl::PointNormal>::ConstPtr(new
		pcl::FieldComparison<pcl::PointNormal>("curvature", pcl::ComparisonOps::GT, 1.)));

	// 2 使用滤波器
	pcl::ConditionalRemoval<pcl::PointXYZ> cr;
	cr.setInputCloud(cloud);
	cr.setCondition(rangeCond);
	cr.setKeepOrganized(true);//保持点云结构，滤波后点云数目没有减少，使用nan代替
	cr.filter(*cloudFiltered);

	// 3 去除nan点
	vector<int> mapping;
	pcl::removeNaNFromPointCloud(*cloudFiltered, *cloudFiltered, mapping);
}

#include <pcl\filters\model_outlier_removal.h>
#include <pcl/ModelCoefficients.h>
//模型滤波器
void filterModel(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//基于模型和点之间的距离过滤点云中的噪声
	//对整个输入迭代一次，自动--过滤非有限点--API指定模型之外的点--指定的阈值

	//生成模型
	pcl::ModelCoefficients coeff;
	coeff.values.resize(4);
	coeff.values[0] = 0;
	coeff.values[1] = 0;
	coeff.values[2] = 0;
	coeff.values[3] = 1;

	pcl::ModelOutlierRemoval<pcl::PointXYZ> mor;
	mor.setInputCloud(cloud);
	mor.setModelType(pcl::SACMODEL_SPHERE);
	mor.setModelCoefficients(coeff);
	mor.setThreshold(0.5);
	mor.filter(*cloudFiltered);
}

#include <pcl\filters\project_inliers.h>
//投影滤波器
void filterProject(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//当前三维卷积计算量太大，发展不是很成熟
	//点云投影到二维平面能够借助图像算法进行处理
	//单个维度进行投影会损失一些点云信息
	//多个维度进行投影，能降低点云信息的损失
	//点云投影应用最多的方式：生成俯视图

	//生成ax+by+cz+d=0的平面模型：创建系数为a=b=d=0,c=1的平面，将z轴相关点全部投影到xy平面
	pcl::ModelCoefficients::Ptr coeff(new pcl::ModelCoefficients);
	coeff->values.resize(4);
	coeff->values[0] = coeff->values[1] = coeff->values[3] = 0;
	coeff->values[2] = 1;

	//Assertion `px != 0' failed. 必定是智能指针没有初始化！！！！
	pcl::ProjectInliers<pcl::PointXYZ>::Ptr proj(new pcl::ProjectInliers<pcl::PointXYZ>);
	proj->setInputCloud(cloud);
	proj->setModelType(pcl::SACMODEL_PLANE);
	proj->setModelCoefficients(coeff);
	proj->filter(*cloudFiltered);
}
//将点云投影到球面，而非仅仅是一个平面
struct sphere
{
	float centerX;
	float centerY;
	float centerZ;
	float radius;
};
typedef pcl::PointXYZ PointT;
void filterProjectEx(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	PointT pt;
	sphere sp;
	sp.centerX = sp.centerY = sp.centerZ = 0;
	sp.radius = 1;
	for (size_t i = 0; i < cloud->points.size(); i++)
	{
		float d = sqrt(pow(cloud->points[i].x - sp.centerX, 2) + pow(cloud->points[i].y - sp.centerY, 2) + pow(cloud->points[i].z - sp.centerZ, 2));
		pt.x = cloud->points[i].x * sp.radius / d;
		pt.y = cloud->points[i].y * sp.radius / d;
		pt.z = cloud->points[i].z * sp.radius / d;
		cloudFiltered->points.emplace_back(pt);
	}
}

#include <pcl\filters\extract_indices.h>
//索引提取器
void extractPointIdxs(vector<int> indexes, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//指导一个点的索引，要把这个点从原始点云数据中提取出来，需要使用索引提取器
	//indices的结构就是简单的--vector<int>--pcl::IndicesPtr(new vector<int>())

	// 1 取得需要的索引
	pcl::PointIndices indices;
	for (size_t i = 0; i < indexes.size(); i++)
	{
		indices.indices.emplace_back(indexes[i]);
	}

	// 2 索引提取器
	pcl::ExtractIndices<pcl::PointXYZ> extr;
	extr.setInputCloud(cloud);
	extr.setIndices(boost::make_shared<pcl::PointIndices>(indices));//设置索引

																	// 3 提取索引点以及剩下的点
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudExtract(new pcl::PointCloud<pcl::PointXYZ>);
	extr.filter(*cloudExtract);//提取对应索引的点云
	extr.setNegative(true);//提取索引之外剩下的点
	extr.filter(*cloudFiltered);
}
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
//从一个点集中提取子集
void extractFromPointCloud()
{
	pcl::PCLPointCloud2::Ptr cloud_blob(new pcl::PCLPointCloud2), cloud_filtered_blob(new pcl::PCLPointCloud2);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>), cloud_p(new pcl::PointCloud<pcl::PointXYZ>), cloud_f(new pcl::PointCloud<pcl::PointXYZ>);
	// 填入点云数据
	pcl::PCDReader reader;
	reader.read("table_scene_lms400.pcd", *cloud_blob);
	std::cerr << "PointCloud before filtering: " << cloud_blob->width * cloud_blob->height << " data points." << std::endl;
	// 创建滤波器对象:使用叶大小为1cm的下采样
	pcl::VoxelGrid<pcl::PCLPointCloud2> sor;
	sor.setInputCloud(cloud_blob);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*cloud_filtered_blob);//体素滤波(下采样)后的点云放置到cloud_filtered_blob

	cout << cloud_filtered_blob->width * cloud_filtered_blob->height << endl;

	// 转化为模板点云
	pcl::fromPCLPointCloud2(*cloud_filtered_blob, *cloud_filtered);//将下采样后的点云PCLPointCloud2类型转换为PoinCloud类型
	cout << "PointCloud after filtering: " << cloud_filtered->points.size() << " data points." << endl;
	// 将下采样后的数据存入磁盘
	pcl::PCDWriter writer;
	writer.write<pcl::PointXYZ>("table_scene_lms400_downsampled.pcd", *cloud_filtered, false);
	pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients());
	pcl::PointIndices::Ptr inliers(new pcl::PointIndices());  //创建一个PointIndices结构体指针
	// 创建分割对象
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	// 可选
	seg.setOptimizeCoefficients(true); //设置对估计的模型做优化处理
	// 必选
	seg.setModelType(pcl::SACMODEL_PLANE);//设置分割模型类别
	seg.setMethodType(pcl::SAC_RANSAC);//设置使用哪个随机参数估计方法
	seg.setMaxIterations(1000);//迭代次数
	seg.setDistanceThreshold(0.01);//设置是否为模型内点的距离阈值
	// 创建滤波器对象
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	int i = 0, nr_points = (int)cloud_filtered->points.size();
	// 当还多于30%原始点云数据时
	while (cloud_filtered->points.size() > 0.3 * nr_points)
	{
		// 从余下的点云中分割最大平面组成部分
		seg.setInputCloud(cloud_filtered);
		seg.segment(*inliers, *coefficients);
		if (inliers->indices.size() == 0)
		{
			cout << "Could not estimate a planar model for the given dataset." << endl;
			break;
		}
		// 分离内层
		extract.setInputCloud(cloud_filtered);
		extract.setIndices(inliers);
		extract.setNegative(false);
		extract.filter(*cloud_p);
		cout << "cloud_filtered: " << cloud_filtered->size() << endl;//输出提取之后剩余的
		cout << "----------------------------------" << endl;
		//保存
		cout << "PointCloud representing the planar component: " << cloud_p->points.size() << " data points." << endl;
		std::stringstream ss;
		ss << "table_scene_lms400_plane_" << i << ".pcd"; //对每一次的提取都进行了文件保存
		writer.write<pcl::PointXYZ>(ss.str(), *cloud_p, false);
		// 创建滤波器对象
		extract.setNegative(true);//提取外层
		extract.filter(*cloud_f);//将外层的提取结果保存到cloud_f
		cloud_filtered.swap(cloud_f);//将cloud_filtered与cloud_f交换

		i++;
	}

	cout << "cloud_filtered: " << cloud_filtered->size() << endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_seg1(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_seg2(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_voxel(new pcl::PointCloud<pcl::PointXYZ>);

	pcl::io::loadPCDFile("table_scene_lms400_plane_0.pcd", *cloud_seg1);
	pcl::io::loadPCDFile("table_scene_lms400_plane_1.pcd", *cloud_seg2);
	pcl::io::loadPCDFile("table_scene_lms400_downsampled.pcd", *cloud_voxel);
	/*//将提取结果进行统计学滤波
	pcl::StatisticalOutlierRemoval<pcl::PointXYZ> sor1;
	sor1.setInputCloud(cloud_seg2);
	sor1.setMeanK(50);
	sor1.setStddevMulThresh(1);
	sor1.filter(*cloud_f);
	cout<<cloud_f->size()<<endl;*/

	pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer);
	viewer->initCameraParameters();

	int v1(0);
	viewer->createViewPort(0, 0, 0.25, 1, v1);
	viewer->setBackgroundColor(0, 0, 255, v1);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color1(cloud_voxel, 244, 89, 233);
	viewer->addPointCloud(cloud_voxel, color1, "cloud_voxel", v1);

	int v2(0);
	viewer->createViewPort(0.25, 0, 0.5, 1, v2);
	viewer->setBackgroundColor(0, 255, 255, v2);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color2(cloud_seg1, 244, 89, 233);
	viewer->addPointCloud(cloud_seg1, color2, "cloud_seg1", v2);

	int v3(0);
	viewer->createViewPort(0.5, 0, 0.75, 1, v3);
	viewer->setBackgroundColor(34, 128, 0, v3);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color3(cloud_seg2, 244, 89, 233);
	viewer->addPointCloud(cloud_seg2, color3, "cloud_seg2", v3);

	int v4(0);
	viewer->createViewPort(0.75, 0, 1, 1, v4);
	viewer->setBackgroundColor(0, 0, 255, v4);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color4(cloud_filtered, 244, 89, 233);
	viewer->addPointCloud(cloud_filtered, color4, "cloud_statical", v4);

	viewer->spin();
}

#include <pcl\filters\convolution_3d.h>
//高斯滤波
void filterConvolution3D(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//高斯滤波：利用高斯函数经过傅里叶变换后仍具有高斯函数特性
	//高斯滤波平均效果较小，滤波的同时能较好地保持数据原貌，常被使用
	//将某一点与其前后n个数据加权平均，远大于操作距离的点被处理为固定的端点，有助于识别间隙和端点

	//基于高斯核函数的卷积滤波实现
	pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ> kernel;
	kernel.setSigma(4);//标准方差，决定函数宽度
	kernel.setThresholdRelativeToSigma(4);
	kernel.setThreshold(0.05);//距离阈值

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	tree->setInputCloud(cloud);

	//设置convolution相关参数
	pcl::filters::Convolution3D<pcl::PointXYZ, pcl::PointXYZ, pcl::filters::GaussianKernel<pcl::PointXYZ, pcl::PointXYZ>> convolution;
	convolution.setInputCloud(cloud);
	convolution.setKernel(kernel);
	convolution.setNumberOfThreads(6);
	convolution.setSearchMethod(tree);
	convolution.setRadiusSearch(0.01);
	convolution.convolve(*cloudFiltered);
}
#include <boost/random.hpp> //随机数
//点云添加高斯噪声
void GenerateGaussNoise(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& noiseCloud, double miu = 0, double sigma = 0.002)
{
	noiseCloud->points.resize(cloud->points.size());//将点云cloud的size赋值给噪声 
	noiseCloud->header = cloud->header;
	noiseCloud->width = cloud->width;
	noiseCloud->height = cloud->height;
	//模拟噪声生成，此处并非生成了新点，只是将原点偏移了一定距离
	boost::mt19937 seed;   // 随机数生成
	seed.seed(static_cast<unsigned int>(time(0)));
	boost::normal_distribution<> nd(miu, sigma);  // 创建一个有特定期望值和标准差的正态分布：
	boost::variate_generator<boost::mt19937&, boost::normal_distribution<>> ok(seed, nd);
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		noiseCloud->points[i].x = cloud->points[i].x + static_cast<float>(ok());
		noiseCloud->points[i].y = cloud->points[i].y + static_cast<float>(ok());
		noiseCloud->points[i].z = cloud->points[i].z + static_cast<float>(ok());
	}
}

#include <pcl\filters\bilateral.h>
//读取txt并转换为点云---可当做TXT读取点云函数的模板
bool readTXT2PointCloud(const string& filePath, const char tag,
	pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud)
{
	cout << "read in txt file start ..." << endl;
	ifstream fp(filePath);
	string lineStr;
	while (getline(fp, lineStr))
	{
		vector<string> strVec;
		string s;
		stringstream ss(lineStr);
		//tag == '';一般使用空格符分隔
		while (getline(ss, s, tag))
		{
			strVec.emplace_back(s);
		}
		pcl::PointXYZI pt;
		pt.x = stod(strVec[0]);
		pt.y = stod(strVec[1]);
		pt.z = stod(strVec[2]);
		pt.intensity = stoi(strVec[3]);
		cloud->points.emplace_back(pt);
	}
	return true;
}
//双边滤波
/*
	双边滤波对输入的点云格式有要求
	--pcl::PointXYZI
*/
void filterBilateral(pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZI>::Ptr& cloudFiltered)
{
	//一种非线性滤波，可以达到保持边缘的同时对图像进行平滑的效果；
	//经过实验，时间复杂度较高，大的点云不适合；
	//去噪效果效果需要根据实际点云情况，不是所有需要平滑和保留边缘的情况使用
	//修改点的位置，而非删除点
	//pcl::BilateralFilter<PointT>实现点云数据的双边滤波，需要使用点云的强度信息！！！pcd格式点云没有强度信息，故此直接读取了包含intensity的TXT文件
	//以上原因解释了为何本函数代码会出现以下报错：
	//error LNK2001: 无法解析的外部符号 "public: virtual void __cdecl pcl::BilateralFilter<struct pcl::PointXYZ>::applyFilter(class pcl::PointCloud<struct pcl::PointXYZ> &)"

	//建立kdtree
	pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
	//进行滤波
	pcl::BilateralFilter<pcl::PointXYZI>::Ptr bf(new pcl::BilateralFilter<pcl::PointXYZI>);
	bf->setInputCloud(cloud);
	bf->setSearchMethod(tree);//以树的方式进行查找
	bf->setHalfSize(0.1);//设置高斯双边滤波窗口一般的大小
	bf->setStdDev(0.03);//设置标准差参数
	bf->filter(*cloudFiltered);
}
#include <pcl\filters\fast_bilateral.h>
//快速双边滤波
/*	
	需要注意的是，该滤波的输入点云必须是有组织的，也就是类似图片那样按照宽高来存放的。
	因此该滤波一般是对从rgbd生成的点云进行处理的
	输入的点云格式--pcl::PointXYZRGB--pcl::PointXYZRGBA
*/
void filterBilateralEx(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudFiltered)
{
	//对空间关系和强度进行下采样从而加速
	//与经典的双边滤波器相比，运行相同时间该方法更准确
	//高维空间下，双边滤波器可以表示为简单的非线性卷积
	//高维空间下。下采样对卷积结果影响不大；这种近似技术可以在控制误差的同时，使速度提升几个量级

	pcl::FastBilateralFilter<pcl::PointXYZRGB> fbf;
	fbf.setInputCloud(cloud);
	fbf.setSigmaS(5);
	fbf.setSigmaR(.03f);// 设置双侧滤波器用于空间邻域/窗口的高斯的标准差
	fbf.filter(*cloudFiltered);// 设置高斯的标准差，以控制由于强度差(在我们的情况下是深度)，相邻像素被降权的程度
}
#include <pcl\filters\fast_bilateral_omp.h>
//多线程快速双边滤波
void filterBilateralExOMP(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudFiltered)
{
	//实现在有组织的点云中平滑深度信息多线程快速双边滤波
	//--
	pcl::FastBilateralFilterOMP<pcl::PointXYZRGB> fbf;
	fbf.setInputCloud(cloud);
	fbf.setSigmaS(5);
	fbf.setSigmaR(.03f);
	fbf.setNumberOfThreads(6);//设置线程数量
	fbf.filter(*cloudFiltered);
}

#include <pcl\filters\median_filter.h>
//中值滤波
void filterMedian(pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr& cloudFiltered)
{
	//最简单和广泛应用的图像处理滤波器
	//使用整个窗口点云数据的统计中值，对于消除数据毛刺，效果较好
	//对彼此靠近的混杂点噪声滤除效果不好
	//注意：该算法只过滤--有组织的PointXYZRGB--未转换(自己设定宽度高度及是否dense)的（摄像机坐标）的深度（z分量）
	//提供未组织的点云给类实例会出错

	pcl::MedianFilter<pcl::PointXYZRGB> m;
	m.setInputCloud(cloud);
	m.setWindowSize(10);
	m.setMaxAllowedMovement(.1f);//一个点允许沿着z轴移动的最大距离
	m.filter(*cloudFiltered);
}

#include <pcl\features\normal_3d_omp.h>//使用OMP进行3D加速
#include <pcl\filters\sampling_surface_normal.h>//求点包含的法向量信息
#include <pcl\filters\shadowpoints.h>
//移除边缘不连续点
void filterShadowPoints(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//shadowpoints移除出现在边缘的不连续点上的幽灵点
	//需要输入的点包含法线信息

	//OMP加速
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::PointNormal> omp;
	omp.setInputCloud(cloud);
	omp.setNumberOfThreads(6);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);//建立kd树来进行近邻点搜索
	omp.setSearchMethod(tree);
	omp.setKSearch(10);//点云法向量计算时，需要搜索的近邻数量
	pcl::PointCloud<pcl::PointNormal>::Ptr normals(new pcl::PointCloud<pcl::PointNormal>);//承接法向量计算的输出结果
	omp.compute(*normals);
	//开始滤波
	pcl::ShadowPoints<pcl::PointXYZ, pcl::PointNormal> sp(true);//分离索引extract removed indices
	sp.setInputCloud(cloud);
	sp.setThreshold(.01f);
	sp.setNormals(normals);
	sp.filter(*cloudFiltered);
}


//显示点云--单窗口
void visualizeFilteredCloudSingle(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//点云显示
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D Viewer"));
	viewer->setBackgroundColor(0, 0, 0);
	pcl::visualization::PointCloudColorHandlerCustom<PointT> view(cloud, 0, 0, 255);
	viewer->addPointCloud<PointT>(cloud, view, "Raw point clouds");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "Raw point clouds");
	pcl::visualization::PointCloudColorHandlerCustom<PointT> viewFiltered(cloudFiltered, 255, 0, 0);
	viewer->addPointCloud<PointT>(cloudFiltered, viewFiltered, "filtered point clouds");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "filtered point clouds");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
//显示点云--双窗口
void visualizeFilteredCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	//PCL可视化工具
	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer("ShowClouds"));
	int v1 = 0, v2 = 0;
	view->createViewPort(.0, .0, .5, 1., v1);
	view->setBackgroundColor(0., 0., 0., v1);
	view->addText("Raw point clouds", 10, 10, "text1", v1);
	view->createViewPort(.5, .0, 1., 1., v2);
	view->setBackgroundColor(0., 0., 0., v2);
	view->addText("filtered point clouds", 10, 10, "text2", v2);

	view->addPointCloud<pcl::PointXYZ>(cloud, "Raw point clouds", v1);
	view->addPointCloud<pcl::PointXYZ>(cloudFiltered, "filtered point clouds", v2);
	//设置点云的渲染属性,string &id = "cloud"相当于窗口ID
	view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Raw point clouds", v1);//设置点云颜色
	view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "filtered point clouds", v2);
	//view->addCoordinateSystem(1.0);
	//view->initCameraParameters();
	while (!view->wasStopped())
	{
		view->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
//显示点云--双窗口
void visualizeFilteredCloud2(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudFiltered)
{
	pcl::visualization::PCLVisualizer viewer("Cloud Viewer");
	//--------创建两个显示窗口并设置背景颜色------------
	viewer.setBackgroundColor(0, 0, 0);
	//-----------给点云添加颜色-------------------------
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> in_h(cloud, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> no_h(cloudFiltered, 255, 0, 0);
	//----------添加点云到显示窗口----------------------
	viewer.addPointCloud(cloud, in_h, "cloud_in");
	viewer.addPointCloud(cloudFiltered, no_h, "cloud_out");

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}


//------------------------------------------------------------------------------------------------//

int maininstance()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudFiltered(new pcl::PointCloud<pcl::PointXYZ>);

	string cloudPath = "../bunny.pcd";
	if (pcl::io::loadPCDFile(cloudPath, *cloud) != 0)
		return -1;
	//pcl::PCDReader reader;
	//reader.read(cloudPath, *cloud);	
	cout << "加载点云数量：" << cloud->points.size() << "个" << endl;

	//------------------------------函数调用---------------------------------------------------
	//索引提取
	//vector<int> extr{ 0, 1, 2, 3, 4, 5, 6, 7 };
	//extractPointIdxs(extr, cloud, cloudFiltered);
	//直通滤波器
	//filterPassThrough(cloud, cloudFiltered);
	//体素滤波器
	//filterVoxelGrid(cloud, cloudFiltered);
	//filterApproximateVoxelGrid(cloud, cloudFiltered);
	//改进K近邻体素滤波器
	//filterVoxelGridEx(cloud, cloudFiltered);
	//统计滤波器
	//filterStddevMulThresh(cloud, cloudFiltered);
	//半径滤波器
	//filterRadius(cloud, cloudFiltered);
	//条件滤波器
	//filterCondition(cloud, cloudFiltered);
	//模型滤波
	//filterModel(cloud, cloudFiltered);
	//投影滤波器
	//filterProject(cloud, cloudFiltered);
	//filterProjectEx(cloud, cloudFiltered);
	//高斯滤波
	//filterConvolution3D(cloud, cloudFiltered);
	//GenerateGaussNoise(cloud, cloudFiltered);
	//双边滤波
	//filterBilateral(cloud, cloudFiltered);
	//快速双边滤波
	//filterBilateralEx(cloud, cloudFiltered);
	//多线程快速双边滤波
	//filterBilateralExOMP(cloud, cloudFiltered);
	//中值滤波
	//filterMedian(cloud, cloudFiltered);
	//移除边缘不连续点
	filterShadowPoints(cloud, cloudFiltered);
	//-------------------------------------------------------------------------------------

	cout << "滤波点云数量：" << cloudFiltered->points.size() << "个" << endl;
	//pcl::io::savePCDFile("output.pcd", *cloudFiltered);
	//pcl::PCDWriter writer;
	//writer.write<pcl::PointXYZ>("output.pcd", *cloudFiltered, false);

	visualizeFilteredCloud(cloud, cloudFiltered);
	//visualizeFilteredCloud2(cloud, cloudFiltered);
	//visualizeFilteredCloudSingle(cloud, cloudFiltered);

	return 0;
}