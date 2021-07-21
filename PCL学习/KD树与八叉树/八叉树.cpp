#include <iostream>
#include <pcl/io/pcd_io.h>  
#include <pcl/point_cloud.h>  
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;

/*
	建立空间索引在点云数据处理中已被广泛的应用，常见的空间索引一般是自顶向下逐级划分空间的各种空间索引结构.
	八叉树结构通过对三维空间的几何实体进行体元剖分，每个体元具有相同的时间和空间复杂度;
	通过循环递归的划分方法对大小为( 2 nx 2 n x 2 n ) 的三维空间的几何对象进行剖分，从而构成一个具有根节点的方向图.
	八叉树是一种用于管理稀疏3D点云的树状数据结构，可实现“体素内近邻搜索、K近邻搜索、半径内近邻搜索”

	实现Octree的步骤:
	(1). 设定最大递归深度
	(2). 找出场景的最大尺寸，并以此尺寸建立第一个立方体
	(3). 依序将单位元元素丢入能被包含且没有子节点的立方体
	(4). 若没有达到最大递归深度，就进行细分八等份，再将该立方体所装的单位元元素全部分担给八个子立方体
	(5). 若发现子立方体所分配到的单位元元素数量不为零且跟父立方体是一样的，则该子立方体停止细分;
	因为跟据空间分割理论，细分的空间所得到的分配必定较少，若是一样数目，则再怎么切数目还是一样，会造成无穷切割的情形。
	(6). 重复3，直到达到最大递归深度。

	PCL中octree 在压缩点云数据方面应用:
	点云由海量的数据集组成，这些数据通过距离、颜色、法线等附加信息来描述空间三维点。
	此外，点云能以非常高的速率被创建出来，因此需要占用相当大的存储资源，
	一旦点云需要存储或者通过速率受限制的通信信道进行传输，提供针对这种数据的压缩方法就变得十分有用。
	PCL库提供了点云压缩功能，它允许编码压缩所有类型的点云，包括无序点云，
	它具有无参考点和变化的点的尺寸、分辨率、分布密度和点顺序等结构特征。
	而且，底层的octree数据结构允许从几个输入源高效地合并点云数据。

	八叉树和k-d树比较:
	(1).八叉树算法的算法实现简单，但大数据量点云数据下，比较困难的是最小粒度（叶节点）的确定，
	粒度较大时，有的节点数据量可能仍比较大，后续查询效率仍比较低，反之，粒度较小，八叉树的深度增加，
	需要的内存空间也比较大（每个非叶子节点需要八个指针），效率也降低。
	而等分的划分依据，使得在数据重心有偏斜的情况下，受划分深度限制，其效率不是太高。
	(2).k-d在邻域查找上比较有优势，但在大数据量的情况下，若划分粒度较小时，建树的开销也较大，但比八叉树灵活些。
	在小数据量的情况下，其搜索效率比较高，但在数据量增大的情况下，其效率会有一定的下降，一般是线性上升的规律。
	(3).也有将八叉树和k-d树结合起来的应用，应用八叉树进行大粒度的划分和查找，而后使用k-d树进行细分，
	效率会有一定的提升，但其搜索效率变化也与数据量的变化有一个线性关系。

	OCtoMap，这是一种高效的可以很好的压缩点云节省存储空间，可实时更新地图，可设置分辨率的八叉树地图。
*/

//------------------------------------------------------------------------------------------------//

//八叉树的使用
#include <pcl/octree/octree_search.h>
void octreeUSE(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& mcloud)
{
	srand((unsigned int)time(NULL));
	//定义并实例化一个共享的PointCloud结构，并使用随机点填充它。
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->width = 1000;
	cloud->height = 1;
	cloud->points.resize(cloud->width * cloud->height);
	for (std::size_t i = 0; i < cloud->size(); ++i)
	{
		(*cloud)[i].x = 1024.0f * rand() / (RAND_MAX + 1.0f);
		(*cloud)[i].y = 1024.0f * rand() / (RAND_MAX + 1.0f);
		(*cloud)[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}

	/*-----------------------------------------------------------------------------
	* 1、创建一个八叉树实例，使用八叉树分辨率进行初始化。这个八叉树在它的叶节点中*
	*保留了一个点索引向量。分辨率参数描述最低八叉树级别上最小体素的长度。因此，  *
	*八叉树的深度是分辨率的函数，也是点云的空间维数的函数。如果知道点云的边界框，*
	*则应该使用finebeliingBox方法将其分配给八叉树。                              *
	*然后，为PointCloud分配一个指针，并将所有的点添加到八叉树中。                *
	-----------------------------------------------------------------------------*/
	float resolution = 128.0f;                                            // 八叉树分辨率
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);// 使用分辨率初始化八叉树
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	pcl::PointXYZ searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

	/*------------------------------------------------------------------------------
	* 2、一旦PointCloud与八叉树相关联，就可以执行搜索操作。这里使用的第一个搜索法 *
	*是“Voxel搜索中的邻居”。它将搜索点分配给相应的叶节点体素，并返回点索引的向量*
	*这些指数与属于同一体素范围内的点有关。                                       *
	*因此，搜索点与搜索结果之间的距离取决于八叉树的分辨率参数。                   *
	* ----------------------------------------------------------------------------*/
	std::vector<int> pointIdxVec;
	if (octree.voxelSearch(searchPoint, pointIdxVec))
	{
		std::cout << "Neighbors within voxel search at (" << searchPoint.x
			<< " " << searchPoint.y
			<< " " << searchPoint.z << ")"
			<< std::endl;
		for (std::size_t i = 0; i < pointIdxVec.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxVec[i]].x
			<< " " << (*cloud)[pointIdxVec[i]].y
			<< " " << (*cloud)[pointIdxVec[i]].z << std::endl;
	}

	/*-------------------------------------------------------------------------------
	* 3、其次，证明了K最近邻搜索。在这个例子中，K被设置为10。                           *
	*“K最近邻搜索”方法将搜索结果写入两个独立的向量中。                            *
	*第一个是pointIdxNKNSearch，它将包含搜索结果(引用相关PointCloud数据集的索引)。  *
	*第二个向量在搜索点和最近邻之间保持相应的平方距离。                             *
	*-------------------------------------------------------------------------------*/
	int K = 10;
	std::vector<int> pointIdxNKNSearch;
	std::vector<float> pointNKNSquaredDistance;
	std::cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;
	if (octree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxNKNSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxNKNSearch[i]].x
			<< " " << (*cloud)[pointIdxNKNSearch[i]].y
			<< " " << (*cloud)[pointIdxNKNSearch[i]].z
			<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << std::endl;
	}

	/*-----------------------------------------------------------------------
	* 4、“半径搜索中的邻居”的工作原理非常类似于“K最近邻搜索”。              *
	*它的搜索结果被写入两个分别描述点索引和平方搜索点距离的向量中。         *
	*-----------------------------------------------------------------------*/
	std::vector<int> pointIdxRadiusSearch;
	std::vector<float> pointRadiusSquaredDistance;
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f);
	std::cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;
	if (octree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxRadiusSearch[i]].x
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].y
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].z
			<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
}

#include <pcl/compression/octree_pointcloud_compression.h> //点云压缩
#include <pcl/io/ply_io.h>
//PCL 八叉树的应用——点云压缩
void pointCloudCompression(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& sourceCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudOut)
{
	/*
		配置文件预先为点云压缩定义了参数，解压器则没必要输入这些参数：
		compressionProfile = MED_RES_ONLINE_COMPRESSION_WITH_COLOR, 配置文件
		showStatistics = false, 是否将压缩相关的统计信息打印到标准输出上
		pointResolution = 0.001, 定义点坐标的编码精度，应设置为小于传感器精度的值：点的分辨率决定坐标在编码时可以精确的程度，仅在细节编码时生效
		octreeResolution = 0.01, 八叉树分辨率：划分八叉树时最小块，即voxel的边长
		doVoxelGridDownDownSampling = true, 进行体素下采样，每个体素内只留下体素中心一个点；false则进行细节编码
		iFrameRate = 100, 如果数据流中的帧速率低于这个速率则不进行编码压缩：每隔一定帧数进行I编码，中间帧进行P编码
		doColorEncoding_arg = true, 是否对彩色编码压缩
		colorBitResolution_arg = 6 定义每一个彩色成分编码后所占的位数

		编码流程图：
		八叉树编码结构->是否编码细节->细节编码->是否编码颜色->颜色编码->熵编码
		1、在该流程前，需判定当前帧是I帧还是P帧。I帧，正常编码；P帧，在编码八叉树占位码时，编码的是和之前帧的异或之后的差值
		2、八叉树的结构编码就是在八叉树划分之后，编码体素的占位码，如果doVoxelGridDownDownSampling体素下采样，则跳过细节编码部分。
		解码端在解码时只要根据八叉树的结构信息恢复出八叉树结构，以体素中心点为代表即可
		3、细节编码:编码体素内的细节。若八叉树划分时，体素的大小较大（点的单位是毫米级的，即每个点对应一个边长1mm的正方体块），
		而我们在设置八叉树分辨率时，设置为5mm，则八叉树划分到边长5mm的立方体就停止。此时一个体素内会有多个点。
		若不进行细节编码，doVoxelGridDownDownSampling为true，在解码时一个体素内只会在其中心位置恢复一个点（其余多个点的均值），会导致点的损失
		此时进行细节编码，会编码一个体素内的点的细节信息（具体位置与属性）
	*/

	// 1、设置参数
	pcl::io::compression_Profiles_e compressionProfile = pcl::io::MANUAL_CONFIGURATION; // 设置压缩选项,启用高级参数化设置
	bool showStatistics = true;              // 设置是否输出打印压缩信息
	const float pointResolution = 0.001;     // 定义点坐标的编码精度，该参数应设为小于传感器精度的一个值
	const float octreeResolution = 0.01;     // 八叉树分辨率
	bool doVoxelGridDownDownSampling = true; // 是否进行体素下采样（每个体素内只留下体素中心一个点）
	const unsigned int iFrameRate = 100;     // 差分编码压缩速率
	bool doColorEncoding = false;             // 是否对彩色编码压缩
	const unsigned char colorBitResolution = 8;// 定义每一个彩色成分编码后所占的位数

	// 2、初始化压缩对象,输入参数
	pcl::io::OctreePointCloudCompression<pcl::PointXYZ>* PointCloudEncoder;
	PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compressionProfile,
		showStatistics,
		pointResolution,
		octreeResolution,
		doVoxelGridDownDownSampling,
		iFrameRate,
		doColorEncoding,
		colorBitResolution);

	// 存储压缩点云的字节流对象
	std::stringstream compressedData; 
	// 3、压缩点云
	PointCloudEncoder->encodePointCloud(sourceCloud->makeShared(), compressedData); // 压缩点云
	compressedData.write("compressed.bin", sizeof(compressedData));

	// 4、解压缩点云
	PointCloudEncoder->decodePointCloud(compressedData, cloudOut); // 解压缩点云
}

#include <pcl/octree/octree.h>
//PCL 八叉树的应用——空间变化检测
void spatialChangeDetection()
{
	/*
		八叉树可用来实现多个无序点云之间的变化检测，这些点可能在尺寸、分辨率、密度、点顺序等方面有所差异
		通过递归地比较八叉树的树结构，可以鉴定出八叉树产生的体素组成之间的区别所代表的空间变化
	*/

	//八叉树分辨率 即体素的大小
	float resolution = 0.1f;
	//初始化空间点云变化检测对象
	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree(resolution);//体素越小，约束越紧致
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloudA);
	//添加点云到八叉树，建立八叉树
	octree.setInputCloud(cloudA);
	octree.addPointsFromInputCloud();
	//交换八叉树缓存，但是cloudA对应的八叉树仍在内存中
	octree.switchBuffers();//pcl八叉树双缓冲技术，随着时间的推移高效的处理多个点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloudB);
	//添加cloudB到八叉树
	octree.setInputCloud(cloudB);
	octree.addPointsFromInputCloud();
	vector<int>newPointIdxVector;
	//获取前一cloudA对应的八叉树在cloudB对应八叉树中没有的体素--求两个点云在八叉树结构下，在最小体素单位为resolution下，点云的差异
	octree.getPointIndicesFromNewVoxels(newPointIdxVector);//第一次求，是在求cloudB；在求一次才是求cloudA
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_change(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloudB, newPointIdxVector, *cloud_change);
	//打印输出点
	cout << "Output from getPointIndicesFromNewVoxels:" << endl;
	for (size_t i = 0; i<newPointIdxVector.size(); ++i)
		cout << i << "# Index:" << newPointIdxVector[i]
		<< "  Point:" << cloudB->points[newPointIdxVector[i]].x << " "
		<< cloudB->points[newPointIdxVector[i]].y << " "
		<< cloudB->points[newPointIdxVector[i]].z << endl;

	// 初始化点云可视化对象
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
	viewer->setBackgroundColor(0, 0, 0);
	// 对cloudA点云着色可视化 (白色).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloudA_color(cloudA, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloudA, cloudA_color, "cloudA");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudA");
	// 对cloudB点云着色可视化 (绿色).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloudB_color(cloudB, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloudB, cloudB_color, "cloudB");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudB");
	// 对检测出来的变化区域可视化.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>change_color(cloud_change, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_change, change_color, "cloud_change");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_change");
	// 等待直到可视化窗口关闭
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}
}

//PCL 求八叉树的体素中心
using AlignedPointTVector = std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>;
//分配内存对齐Eigen::MatrixXf;用来储存计算到的中心点
void computeVoxelCenter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	//查找体素格的中心
	AlignedPointTVector voxel_center;
	voxel_center.clear();      //初始化
	float resolution = 0.25f;   //设置体素格的边长
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	int treeDepth_ = octree.getTreeDepth();
	cout << "the depth of treeDepth_ is : " << treeDepth_ << endl;
	int occupiedVoxelCenters = octree.getOccupiedVoxelCenters(voxel_center);//求八叉树的体素中心
	cout << "the number of occupiedVoxelCenters are : " << occupiedVoxelCenters << endl;//输出体素格中心个数
	cout << "the number of voxel are : " << voxel_center.size() << endl;   //输出体素中心的个数
	cout << "voxel0 is : " << voxel_center[0].x << voxel_center[0].y << endl;
}

//PCL 基于八叉树的体素滤波--使用八叉树的体素中心点来精简原始点云（此中心点可能不是点云中的点）
using AlignedPointT = Eigen::aligned_allocator<pcl::PointXYZ>;
void octreeVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octree_filter_cloud)
{
	/*
		pcl实现基于体素的滤波方式对点云进行下采样
		八叉树同样也是建立体素，因此同样也可以对点云进行下采样
		最简单方法：用八叉树的体素中心来代替每一个体素内的点，实现点云下采样
		此法与ApproximateVoxelGrid基本相同，都是以中心点代替体素内的点
		唯一区别：ApproximateVoxelGrid可以自由设置体素的长宽高，而八叉树只能是构建正方体的体素
		改进：用距离体素中心最近的点来代替体素中心点，从而使得下采样之后的点还是原始点云中的数据
	*/

	float m_resolution = 0.002;
	pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(m_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<pcl::PointXYZ, AlignedPointT> voxel_centers;
	octree.getOccupiedVoxelCenters(voxel_centers);

	octree_filter_cloud->width = voxel_centers.size();
	octree_filter_cloud->height = 1;
	octree_filter_cloud->points.resize(octree_filter_cloud->height * octree_filter_cloud->width);
	for (size_t i = 0; i < voxel_centers.size() - 1; i++)
	{
		octree_filter_cloud->points[i].x = voxel_centers[i].x;
		octree_filter_cloud->points[i].y = voxel_centers[i].y;
		octree_filter_cloud->points[i].z = voxel_centers[i].z;
	}
	std::cout << "体素中心点滤波后点云个数为：" << voxel_centers.size() << std::endl;
}

#include <pcl/kdtree/kdtree_flann.h>
//PCL 基于八叉树的体素滤波--使用八叉树的体素中心点的最近邻点来精简原始点云（最终获取的点云都还是原始点云数据中的点）
void octreeVoxelFilterCenterKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octknn_filter_cloud)
{
	float m_resolution = 0.002;
	pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(m_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<pcl::PointXYZ, AlignedPointT> voxel_centers;
	octree.getOccupiedVoxelCenters(voxel_centers);
	//-----------K最近邻搜索------------
	//根据下采样的结果，选择距离采样点最近的点作为最终的下采样点
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointIndicesPtr inds = boost::shared_ptr<pcl::PointIndices>(new pcl::PointIndices());//采样后根据最邻近点提取的样本点下标索引
	for (size_t i = 0; i < voxel_centers.size(); ++i) 
	{
		pcl::PointXYZ searchPoint;
		searchPoint.x = voxel_centers[i].x;
		searchPoint.y = voxel_centers[i].y;
		searchPoint.z = voxel_centers[i].z;
		int K = 1;//最近邻搜索
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
		{
			inds->indices.push_back(pointIdxNKNSearch[0]);
		}
	}
	pcl::copyPointCloud(*cloud, inds->indices, *octknn_filter_cloud);
	std::cout << "体素中心最近邻点滤波后点云个数为：" << octknn_filter_cloud->points.size() << std::endl;
};

//------------------------------------------------------------------------------------------------//

//显示点云--双窗口
void visualizeOCCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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

void main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloud) == -1)
	{
		PCL_ERROR("Cloudn't read file!");
		return;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);

	//------------------------------------------------------------------------------------------------//
	
	//octreeUSE(cloud);
	//pointCloudCompression(cloud, cloud_filtered);
	//spatialChangeDetection();
	//computeVoxelCenter(cloud);
	//octreeVoxelFilter(cloud, cloud_filtered);
	octreeVoxelFilterCenterKNN(cloud, cloud_filtered);

	//------------------------------------------------------------------------------------------------//

	visualizeOCCloud(cloud, cloud_filtered);
	return;
}