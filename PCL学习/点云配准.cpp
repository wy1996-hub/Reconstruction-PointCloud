#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>   // 控制台计算时间
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

//------------------------------------------------------------------------------------------------//
void visualizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
	// 初始化点云可视化对象
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("显示点云"));
	viewer->setBackgroundColor(0, 0, 0);
	// 对目标点云着色可视化 (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
	// 对源点云着色可视化 (blue).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input_color(source_cloud, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(source_cloud, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "input cloud");
	// 对转换后的源点云着色 (green)可视化.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(output_cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output cloud");
	// 等待直到可视化窗口关闭
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}
}
//-----------------------------------------粗配准------------------------------------------//

//PCL 4PCS算法实现点云粗配准
#include <pcl/registration/ia_fpcs.h> // 4PCS算法
void _4PCS(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
{
	pcl::console::TicToc time;
	time.tic();
	//--------------初始化4PCS配准对象-------------------
	pcl::registration::FPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> fpcs;
	fpcs.setInputSource(source_cloud);  // 源点云
	fpcs.setInputTarget(target_cloud);  // 目标点云
	fpcs.setApproxOverlap(0.7);         // 设置源和目标之间的近似重叠度。
	fpcs.setDelta(0.01);                // 设置配准后对应点之间的距离（以米为单位）。
	fpcs.setNumberOfSamples(100);       // 设置验证配准效果时要使用的采样点数量
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcs(new pcl::PointCloud<pcl::PointXYZ>);
	fpcs.align(*pcs);                   // 计算变换矩阵
	cout << "FPCS配准用时： " << time.toc() << " ms" << endl;
	cout << "变换矩阵：" << fpcs.getFinalTransformation() << endl;
	// 使用创建的变换对为输入点云进行变换
	pcl::transformPointCloud(*source_cloud, *pcs, fpcs.getFinalTransformation());
	visualizeCloud(target_cloud, source_cloud, pcs);
}

//PCL K4PCS算法实现点云粗配准
#include <pcl/registration/ia_kfpcs.h> //K4PCS算法头文件
void K4PCS(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		1 利用滤波器进行点云下采样，然后进行harris\DoG关键点检测
		2 通过4PCS算法，使用关键点集合，而非原始点云进行数据的匹配，提高运算效率
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------------K4PCS算法进行配准------------------------------
	pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
	kfpcs.setInputSource(source);  // 源点云
	kfpcs.setInputTarget(target);  // 目标点云
	kfpcs.setApproxOverlap(0.7);   // 源和目标之间的近似重叠。
	kfpcs.setLambda(0.5);          // 平移矩阵的加权系数。(暂时不知道是干什么用的)
	kfpcs.setDelta(0.002, false);  // 配准后源点云和目标点云之间的距离
	kfpcs.setNumberOfThreads(6);   // OpenMP多线程加速的线程数
	kfpcs.setNumberOfSamples(200); // 配准时要使用的随机采样点数量
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpcs(new pcl::PointCloud<pcl::PointXYZ>);
	kfpcs.align(*kpcs);

	cout << "KFPCS配准用时： " << time.toc() << " ms" << endl;
	cout << "变换矩阵：\n" << kfpcs.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::transformPointCloud(*source, *kpcs, kfpcs.getFinalTransformation());
	visualizeCloud(target, source, kpcs);
}

//PCL 改进的RANSAC算法实现点云粗配准
#include <pcl/registration/sample_consensus_prerejective.h>//　随机采样一致性配准
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
typedef pcl::PointCloud<pcl::Normal> pointnormal;
typedef pcl::PointCloud<pcl::FPFHSignature33> fpfhFeature;
typedef pcl::PointCloud<pcl::PointXYZ> pointcloud;
fpfhFeature::Ptr compute_fpfh_feature(pointcloud::Ptr input_cloud)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	//-------------------------法向量估计-----------------------
	pointnormal::Ptr normals(new pointnormal);
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(input_cloud);
	n.setNumberOfThreads(8);        // 设置openMP的线程数
	n.setSearchMethod(tree);        // 搜索方式
	n.setKSearch(10);               // K近邻点个数
	n.compute(*normals);           
	//-------------------------FPFH估计-------------------------
	fpfhFeature::Ptr fpfh(new fpfhFeature);
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fest;
	fest.setNumberOfThreads(8);     //指定8核计算
	fest.setInputCloud(input_cloud);//输入点云
	fest.setInputNormals(normals);  //输入法线
	fest.setSearchMethod(tree);     //搜索方式
	fest.setKSearch(10);            //K近邻点个数
	//fest.setRadiusSearch(0.025);  //搜索半径
	fest.compute(*fpfh);            //计算FPFH
	return fpfh;
}
void RANSACEx(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		RANSAC:random sampling consensus 随机采样一致性，常用于点云配准
		点云配准的目的就是估计：刚体变换矩阵RT
	*/
	//---------------计算源点云和目标点云的FPFH----------------------
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);
	//--------------------RANSAC点云配准-----------------------------
	pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> r_sac;
	r_sac.setInputSource(source);            // 源点云
	r_sac.setInputTarget(target);            // 目标点云
	r_sac.setSourceFeatures(source_fpfh);    // 源点云FPFH特征
	r_sac.setTargetFeatures(target_fpfh);    // 目标点云FPFH特征
	r_sac.setCorrespondenceRandomness(5);    // 在选择随机特征对应时，设置要使用的邻居的数量,数值越大，特征匹配的随机性越大。
	r_sac.setInlierFraction(0.5f);           // 所需的(输入的)inlier分数
	r_sac.setNumberOfSamples(3);             // 每次迭代中使用的采样点数量
	r_sac.setSimilarityThreshold(0.1f);      // 将底层多边形对应拒绝器对象的边缘长度之间的相似阈值设置为[0,1]，其中1为完全匹配。
	r_sac.setMaxCorrespondenceDistance(1.0f);// 内点，阈值 Inlier threshold
	r_sac.setMaximumIterations(100);         // RANSAC 　最大迭代次数
	pointcloud::Ptr align(new pointcloud);
	r_sac.align(*align);
	pcl::transformPointCloud(*source, *align, r_sac.getFinalTransformation());
	cout << "变换矩阵：\n" << r_sac.getFinalTransformation() << endl;

	//-------------------可视化------------------------------------
	visualizeCloud(source, target, align);
}

//PCL SAC-IA 初始配准算法-->求变换矩阵
#include <pcl/registration/ia_ransac.h>//sac_ia算法
#include <pcl/filters/voxel_grid.h>//体素下采样滤波
void SAC_IA(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
{
	clock_t start, end, time;
	start = clock();
	//---------------------------去除源点云的NAN点------------------------
	vector<int> indices_src; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*source_cloud, *source_cloud, indices_src);
	//-------------------------源点云下采样滤波-------------------------
	pcl::VoxelGrid<pcl::PointXYZ> vs;
	vs.setLeafSize(0.005, 0.005, 0.005);
	vs.setInputCloud(source_cloud);
	pointcloud::Ptr source(new pointcloud);
	vs.filter(*source);
	cout << "down size *source_cloud from " << source_cloud->size() << " to " << source->size() << endl;
	//--------------------------去除目标点云的NAN点--------------------
	vector<int> indices_tgt; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*target_cloud, *target_cloud, indices_tgt);
	//----------------------目标点云下采样滤波-------------------------
	pcl::VoxelGrid<pcl::PointXYZ> vt;
	vt.setLeafSize(0.005, 0.005, 0.005);
	vt.setInputCloud(target_cloud);
	pointcloud::Ptr target(new pointcloud);
	vt.filter(*target);
	cout << "down size *target_cloud from " << target_cloud->size() << " to " << target->size() << endl;
	//---------------计算源点云和目标点云的FPFH------------------------
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);
	//--------------采样一致性SAC_IA初始配准----------------------------
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(source);
	sac_ia.setSourceFeatures(source_fpfh);
	sac_ia.setInputTarget(target);
	sac_ia.setTargetFeatures(target_fpfh);
	sac_ia.setMinSampleDistance(0.1);//设置样本之间的最小距离
	sac_ia.setCorrespondenceRandomness(6); //在选择随机特征对应时，设置要使用的邻居的数量;
										   //也就是计算协方差时选择的近邻点个数，该值越大，协防差越精确，但是计算效率越低.(可省)
	pointcloud::Ptr align(new pointcloud);
	sac_ia.align(*align);
	end = clock();
	pcl::transformPointCloud(*source_cloud, *align, sac_ia.getFinalTransformation());
	cout << "calculate time is: " << float(end - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "\nSAC_IA has converged, score is " << sac_ia.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << sac_ia.getFinalTransformation() << endl;
	//-------------------可视化------------------------------------
	visualizeCloud(source_cloud, target_cloud, align);
}

//PCL 刚性目标的鲁棒姿态估计
//将一个刚性物体与一个带有杂波和遮挡的场景对齐

//PCA 实现点云粗配准
//通过线性变换提取数据的主要特征分量，常用于降维
//利用点云数据的主轴方向进行配准-->通过主轴求出旋转矩阵，计算点云中心坐标的偏移
//需要进行初始RT校正
#include<pcl/registration/correspondence_estimation.h>
// 计算点云质心与特征向量
void ComputeEigenVectorPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
	Eigen::Vector4f& pcaCentroid, Eigen::Matrix3f& eigenVectorsPCA)
{
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	eigenVectorsPCA = eigen_solver.eigenvectors();
}
// PCA求解变换矩阵
Eigen::Matrix4f PCARegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr& P_cloud, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& X_cloud)
{
	Eigen::Vector4f Cp;                  // P_cloud的质心
	Eigen::Matrix3f Up;                  // P_cloud的特征向量
	ComputeEigenVectorPCA(P_cloud, Cp, Up);// 计算P_cloud的质心和特征向量
	Eigen::Vector4f Cx;                  // X_cloud的质心
	Eigen::Matrix3f Ux;                  // X_cloud的特征向量
	ComputeEigenVectorPCA(X_cloud, Cx, Ux);// 计算X_cloud的质心和特征向量
										   // 分别讨论主轴对应的8种情况，选择误差最小时对应的变换矩阵
	float error[8] = {};
	vector<Eigen::Matrix4f>MF;
	Eigen::Matrix4f final_RT = Eigen::Matrix4f::Identity();// 定义最终的变换矩阵
	Eigen::Matrix3f Upcopy = Up;
	int sign1[8] = { 1, -1,1,1,-1,-1,1,-1 };
	int sign2[8] = { 1, 1,-1,1,-1,1,-1,-1 };
	int sign3[8] = { 1, 1, 1,-1,1,-1,-1,-1 };
	for (int nn = 0; nn < 8; nn++) {
		Up.col(0) = sign1[nn] * Upcopy.col(0);
		Up.col(1) = sign2[nn] * Upcopy.col(1);
		Up.col(2) = sign3[nn] * Upcopy.col(2);
		Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
		R = (Up * Ux.inverse()).transpose(); // 计算旋转矩阵
		Eigen::Matrix<float, 3, 1> T;
		T = Cx.head<3>() - R * (Cp.head<3>());// 计算平移向量
		Eigen::Matrix4f RT = Eigen::Matrix4f::Identity();// 初始化齐次坐标4X4变换矩阵
		RT.block<3, 3>(0, 0) = R;// 构建4X4变换矩阵的旋转部分
		RT.block<3, 1>(0, 3) = T;// 构建4X4变换矩阵的平移部分
		pcl::PointCloud<pcl::PointXYZ>::Ptr t_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::transformPointCloud(*P_cloud, *t_cloud, RT);
		// 计算每一种主轴对应的平均均方误差
		pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>core;
		core.setInputSource(t_cloud);
		core.setInputTarget(X_cloud);
		boost::shared_ptr<pcl::Correspondences> cor(new pcl::Correspondences);
		core.determineReciprocalCorrespondences(*cor);//双向K近邻搜索获取最近点对
		double mean = 0.0, stddev = 0.0;
		pcl::registration::getCorDistMeanStd(*cor, mean, stddev);
		error[nn] = mean;
		MF.push_back(RT);
	}
	// 获取误差最小时所对应的索引
	int min_index = distance(begin(error), min_element(error, error + 8));
	// 误差最小时对应的变换矩阵即为正确变换矩阵
	final_RT = MF[min_index];
	return final_RT;
}
void  visualize_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr& source, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target, pcl::PointCloud<pcl::PointXYZ>::Ptr& regist)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Registration"));
	int v1 = 0;
	int v2 = 1;
	viewer->setWindowName("PCA配准结果");
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->createViewPort(0.5, 0, 1, 1, v2);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->setBackgroundColor(0.05, 0, 0, v2);
	viewer->addText("Raw point clouds", 10, 10, "v1_text", v1);
	viewer->addText("Registed point clouds", 10, 10, "v2_text", v2);
	//原始点云绿色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(source, 0, 255, 0);
	//目标点云蓝色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(target, 0, 0, 255);
	//转换后的源点云红色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> transe(regist, 255, 0, 0);
	viewer->addPointCloud(source, src_h, "source cloud", v1);
	viewer->addPointCloud(target, tgt_h, "target cloud", v1);
	viewer->addPointCloud(target, tgt_h, "target cloud1", v2);
	viewer->addPointCloud(regist, transe, "pcs cloud", v2);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}
}
void PCA(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	pcl::console::TicToc time;
	time.tic();
	//--------------PCA计算变换矩阵------------------
	Eigen::Matrix4f PCATransform = Eigen::Matrix4f::Identity();
	PCATransform = PCARegistration(source, target);
	cout << "PCA计算变换矩阵用时： " << time.toc() / 1000 << " s" << endl;
	cout << "变换矩阵为：\n" << PCATransform << endl;
	//-----------------完成配准----------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr PCARegisted(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*source, *PCARegisted, PCATransform);
	//----------------可视化结果---------------------
	visualize_registration(source, target, PCARegisted);
}

//-----------------------------------------精配准------------------------------------------//

//----------------------点到点的ICP算法----------------------
//最小化源点云与目标点云对应点之间的距离--配准准则

//PCL ICP算法实现点云精配准
#include <pcl/registration/icp.h> // icp算法
void ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		点云配准（Point Cloud Registration）指的是输入两幅点云(source)和(target) ，
		输出一个变换T使得T(Ps)和T(Pt)的重合程度尽可能高；Ps和Pt是源点云和目标点云中的对应点。
		变换T可以是刚性的(rigid)，也可以不是，一般只考虑刚性变换，即变换只包括旋转、平移。
		点云配准可以分为粗配准（Coarse Registration）和精配准（Fine Registration）两步。
		粗配准指的是在两幅点云之间的变换完全未知的情况下进行较为粗糙的配准，目的主要是为精配准提供较好的变换初值；
		精配准则是给定一个初始变换，进一步优化得到更精确的变换。
		目前应用最广泛的点云精配准算法是迭代最近点算法（Iterative Closest Point, ICP）及各种变种 ICP 算法。
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------初始化ICP对象--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//----------------------icp核心代码--------------------
	icp.setInputSource(source);            // 源点云
	icp.setInputTarget(target);            // 目标点云
	icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(1);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值，停止迭代；
	icp.setMaximumIterations(35);           // 最大迭代次数
	icp.setUseReciprocalCorrespondences(true);//设置为true,则使用相互对应关系
											  // 计算需要的刚体变换以便将输入的源点云匹配到目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入源点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
}

//PCL KD-ICP实现点云精配准,构建双向KD树，避免点存在一对多的问题
void KDICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		通过KD树实现近邻算法，对粗配准后的点进行近邻搜索。
		以欧氏距离为判断标准，提出欧氏距离大于阈值的配准关键点，保存配准精度高的点
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------初始化ICP对象--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//---------------------KD树加速搜索--------------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(target);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);
	//----------------------icp核心代码--------------------
	icp.setInputSource(source);            // 源点云
	icp.setInputTarget(target);            // 目标点云
	icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(1);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.05);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	icp.setMaximumIterations(35);           // 最大迭代次数
											//icp.setUseReciprocalCorrespondences(true);//使用相互对应关系
											// 计算需要的刚体变换以便将输入的源点云匹配到目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入源点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
}

//PCL 交互式迭代最近点配准--通过鼠标键盘控制算法迭代次数

//PCL 多幅点云配准

//【论文复现】——基于SAC_IA和NDT融合的点云配准方法
#include <pcl/registration/ndt.h>      // NDT配准算法
pointcloud::Ptr voxel_grid_fiter(pointcloud::Ptr & inCloud)
{
	pcl::VoxelGrid<pcl::PointXYZ> vs;
	vs.setLeafSize(0.005, 0.005, 0.005);
	vs.setInputCloud(inCloud);
	pointcloud::Ptr outCloud(new pointcloud);
	vs.filter(*outCloud);
	return outCloud;
}
void SACIA_NDT(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
{
	pointcloud::Ptr source(new pointcloud);
	pointcloud::Ptr target(new pointcloud);
	source = voxel_grid_fiter(source_cloud); // 下采样滤波
	target = voxel_grid_fiter(target_cloud); // 下采样滤波

	// 1、计算源点云和目标点云的FPFH
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);

	// 2、采样一致性SAC_IA初始配准
	clock_t start, end;
	start = clock();
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(source);
	sac_ia.setSourceFeatures(source_fpfh);
	sac_ia.setInputTarget(target);
	sac_ia.setTargetFeatures(target_fpfh);
	sac_ia.setMinSampleDistance(0.01);       // 设置样本之间的最小距离
	sac_ia.setMaxCorrespondenceDistance(0.1);// 设置对应点对之间的最大距离
	sac_ia.setNumberOfSamples(200);          // 设置每次迭代计算中使用的样本数量（可省）,可节省时间
	sac_ia.setCorrespondenceRandomness(6);   // 设置在6个最近特征对应中随机选取一个
	pointcloud::Ptr align(new pointcloud);
	sac_ia.align(*align);
	Eigen::Matrix4f initial_RT = Eigen::Matrix4f::Identity();// 定义初始变换矩阵
	initial_RT = sac_ia.getFinalTransformation();
	cout << "\nSAC_IA has converged, score is " << sac_ia.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << initial_RT << endl;

	// 3、正态分布变换（NDT）
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setInputSource(source);	             // 设置要配准的点云
	ndt.setInputTarget(target);              // 设置点云配准目标
	ndt.setStepSize(4);                      // 为More-Thuente线搜索设置最大步长
	ndt.setResolution(0.01);                 // 设置NDT网格结构的分辨率（VoxelGridCovariance）
	ndt.setMaximumIterations(35);            // 设置匹配迭代的最大次数
	ndt.setTransformationEpsilon(0.01);      // 为终止条件设置最小转换差异
	pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*output_cloud, initial_RT);    // align()函数有第二个参数，输入的是初始变换的估计参数
	end = clock();
	cout << "NDT has converged:" << ndt.hasConverged()
		<< " score: " << ndt.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << ndt.getFinalTransformation() << endl;
	cout << "运行时间: " << float(end - start) / CLOCKS_PER_SEC << "s" << endl;

	// 4、使用变换矩阵对未进行滤波的原始源点云进行变换
	pcl::transformPointCloud(*source_cloud, *output_cloud, ndt.getFinalTransformation());

	// 5、可视化
	visualize_registration(source_cloud, target_cloud, output_cloud);
}

//【论文复现】——基于NDT与ICP结合的点云配准算法
typedef pcl::PointCloud<PointT> PointCloud;
// 预处理过程
void pretreat(PointCloud::Ptr& pcd_cloud, PointCloud::Ptr& pcd_down, float LeafSize = 0.4) 
{
	//去除NAN点
	std::vector<int> indices_src; //保存去除的点的索引
	pcl::removeNaNFromPointCloud(*pcd_cloud, *pcd_cloud, indices_src);
	std::cout << "remove *cloud_source nan" << endl;
	//下采样滤波
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid.setInputCloud(pcd_cloud);
	voxel_grid.filter(*pcd_down);
	cout << "down size *cloud from " << pcd_cloud->size() << "to" << pcd_down->size() << endl;
}
// 由旋转平移矩阵计算旋转角度
void matrix2angle(Eigen::Matrix4f &result_trans, Eigen::Vector3f &result_angle)
{
	double ax, ay, az;
	if (result_trans(2, 0) == 1 || result_trans(2, 0) == -1)
	{
		az = 0;
		double dlta;
		dlta = atan2(result_trans(0, 1), result_trans(0, 2));
		if (result_trans(2, 0) == -1)
		{
			ay = M_PI / 2;
			ax = az + dlta;
		}
		else
		{
			ay = -M_PI / 2;
			ax = -az + dlta;
		}
	}
	else
	{
		ay = -asin(result_trans(2, 0));
		ax = atan2(result_trans(2, 1) / cos(ay), result_trans(2, 2) / cos(ay));
		az = atan2(result_trans(1, 0) / cos(ay), result_trans(0, 0) / cos(ay));
	}
	result_angle << ax, ay, az;
	cout << "x轴旋转角度：" << ax << endl;
	cout << "y轴旋转角度：" << ay << endl;
	cout << "z轴旋转角度：" << az << endl;
}
void NDT_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source)
{
	clock_t start = clock();
	PointCloud::Ptr cloud_src(new PointCloud);
	PointCloud::Ptr cloud_tar(new PointCloud);
	pretreat(cloud_source, cloud_src);
	pretreat(cloud_target, cloud_tar);
	//NDT配准
	pcl::NormalDistributionsTransform<PointT, PointT> ndt;
	PointCloud::Ptr cloud_ndt(new PointCloud);
	ndt.setInputSource(cloud_src);
	ndt.setInputTarget(cloud_tar);
	ndt.setStepSize(0.5);              // 为More-Thuente线搜索设置最大步长
	ndt.setResolution(2);              // 设置NDT网格结构的分辨率（VoxelGridCovariance）
	ndt.setMaximumIterations(35);      // 设置匹配迭代的最大次数
	ndt.setTransformationEpsilon(0.01);// 为终止条件设置最小转换差异
	ndt.align(*cloud_ndt);
	clock_t ndt_t = clock();
	cout << "ndt time" << (double)(ndt_t - start) / CLOCKS_PER_SEC << endl;
	Eigen::Matrix4f ndt_trans = ndt.getFinalTransformation();

	//icp配准算法
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	PointCloud::Ptr cloud_icp_registration(new PointCloud);
	//设置参数
	icp.setInputSource(cloud_src);
	icp.setInputTarget(cloud_tar);
	icp.setMaxCorrespondenceDistance(10);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(0.1);
	icp.setMaximumIterations(50);
	icp.align(*cloud_icp_registration, ndt_trans);
	clock_t end = clock();
	cout << "icp time" << (double)(end - ndt_t) / CLOCKS_PER_SEC << endl;
	cout << "total time" << (double)(end - start) / CLOCKS_PER_SEC << endl;
	Eigen::Matrix4f icp_trans = icp.getFinalTransformation();
	cout << icp_trans << endl;
	pcl::transformPointCloud(*cloud_source, *cloud_icp_registration, icp_trans);

	//计算误差
	Eigen::Vector3f ANGLE_origin;
	Eigen::Vector3f TRANS_origin;
	ANGLE_origin << 0, 0, M_PI / 4;
	TRANS_origin << 0, 0.3, 0.2;
	double a_error_x, a_error_y, a_error_z;
	double t_error_x, t_error_y, t_error_z;
	Eigen::Vector3f ANGLE_result; // 由IMU得出的变换矩阵

	matrix2angle(icp_trans, ANGLE_result);
	a_error_x = fabs(ANGLE_result(0)) - fabs(ANGLE_origin(0));
	a_error_y = fabs(ANGLE_result(1)) - fabs(ANGLE_origin(1));
	a_error_z = fabs(ANGLE_result(2)) - fabs(ANGLE_origin(2));
	cout << "点云实际旋转角度:\n" << ANGLE_origin << endl;
	cout << "x轴旋转误差 : " << a_error_x << "  y轴旋转误差 : " << a_error_y << "  z轴旋转误差 : " << a_error_z << endl;

	cout << "点云实际平移距离:\n" << TRANS_origin << endl;
	t_error_x = fabs(icp_trans(0, 3)) - fabs(TRANS_origin(0));
	t_error_y = fabs(icp_trans(1, 3)) - fabs(TRANS_origin(1));
	t_error_z = fabs(icp_trans(2, 3)) - fabs(TRANS_origin(2));
	cout << "计算得到的平移距离" << endl << "x轴平移" << icp_trans(0, 3) << endl << "y轴平移" << icp_trans(1, 3) << endl << "z轴平移" << icp_trans(2, 3) << endl;
	cout << "x轴平移误差 : " << t_error_x << "  y轴平移误差 : " << t_error_y << "  z轴平移误差 : " << t_error_z << endl;
	//可视化
	visualize_registration(cloud_source, cloud_target, cloud_icp_registration);
}

//【论文复现】——利用特征点采样一致性改进ICP算法点云配准方法
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint,
	float LeafSize = 0.02, float radius = 0.04, float threshold = 5) // 参数分别为：体素格网的大小，法向量计算半径，夹角阈值（度）
{
	//体素下采样
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_down(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid.setInputCloud(cloud);
	voxel_grid.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid.filter(*pcd_down);
	//计算每一个点的法向量
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(pcd_down);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);
	//设置KD树搜索半径
	n.setKSearch(10);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);

	float Angle = 0.0;
	float Average_Sum_AngleK = 0.0;//定义邻域内K个点法向量夹角的平均值
	vector<int>indexes;
	//--------------计算法向量夹角及夹角均值----------------
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //建立kdtree对象
	kdtree.setInputCloud(pcd_down); //设置需要建立kdtree的点云指针
	vector<int> pointIdxRadiusSearch;  //保存每个近邻点的索引
	vector<float> pointRadiusSquaredDistance;  //保存每个近邻点与查找点之间的欧式距离平方
	pcl::PointXYZ searchPoint;
	for (size_t i = 0; i < pcd_down->points.size(); ++i) {
		searchPoint = pcd_down->points[i];
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			float Sum_AngleK = 0.0;//定义K个邻近的点法向夹角之和
			 /*计算法向量的夹角*/
			for (size_t m = 0; m < pointIdxRadiusSearch.size(); ++m) 
			{
				Eigen::Vector3f
					v1(normals->points[i].data_n[0],
						normals->points[i].data_n[1],
						normals->points[i].data_n[2]),
					v2(normals->points[pointIdxRadiusSearch[m]].data_n[0],
						normals->points[pointIdxRadiusSearch[m]].data_n[1],
						normals->points[pointIdxRadiusSearch[m]].data_n[2]);
				Angle = pcl::getAngle3D(v1, v2, true);
				Sum_AngleK += Angle;//邻域夹角之和
			}		
			Average_Sum_AngleK = Sum_AngleK / pointIdxRadiusSearch.size();															  
			//提取特征点
			float t = pcl::deg2rad(threshold);
			if (Average_Sum_AngleK > t) {
				indexes.push_back(i);
			}
		}
	}
	pcl::copyPointCloud(*pcd_down, indexes, *keypoint);
	cout << "提取的特征点个数:" << keypoint->points.size() << endl;
};
void SACIA_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	//1、 提取特征点
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);

	//2、计算源点云和目标点云的FPFH
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	//3、SAC配准
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> scia;
	scia.setInputSource(s_k);
	scia.setInputTarget(t_k);
	scia.setSourceFeatures(sk_fpfh);
	scia.setTargetFeatures(tk_fpfh);
	scia.setMinSampleDistance(0.007);
	scia.setNumberOfSamples(100);
	scia.setCorrespondenceRandomness(6);
	pcl::PointCloud<pcl::PointXYZ>::Ptr sac_result(new pcl::PointCloud<pcl::PointXYZ>);
	scia.align(*sac_result);
	std::cout << "sac has converged:" << scia.hasConverged() << "  score: " << scia.getFitnessScore() << endl;
	Eigen::Matrix4f sac_trans;
	sac_trans = scia.getFinalTransformation();
	std::cout << sac_trans << endl;

	//4、KD树改进的ICP配准
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//kdTree 加速搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(s_k);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(t_k);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);
	icp.setInputSource(s_k);
	icp.setInputTarget(t_k);
	icp.setMaxCorrespondenceDistance(0.1);
	icp.setMaximumIterations(35);
	icp.setTransformationEpsilon(1e-10);
	icp.setEuclideanFitnessEpsilon(0.01);
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_result(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_result, sac_trans);
	pcl::transformPointCloud(*source, *icp_result, icp.getFinalTransformation());
	// 5、可视化
	visualize_registration(source, target, icp_result);
}

//----------------------点到面的ICP算法----------------------
//最小化源点云中的点到目标点云对应点所在平面的距离
//更能体现点云的空间结构，更好的抵抗错误对应点对，迭代收敛速度更快

//PCL 点到面的ICP算法
#include <pcl/registration/icp.h> // icp算法
void cloud_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
	pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_normals)
{
	//-----------------拼接点云数据与法线信息---------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//建立kdtree来进行近邻点集搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setNumberOfThreads(10);//设置openMP的线程数
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);//点云法向计算时，需要所搜的近邻点大小
	n.compute(*normals);//开始进行法向计					
	pcl::concatenateFields(*cloud, *normals, *cloud_normals);//将点云数据与法向信息拼接
}
void ICP_class(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------拼接点云与法线信息-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//----------------点到面的icp（经典版）-----------------
	pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>p_icp;
	p_icp.setInputSource(source_with_normals);
	p_icp.setInputTarget(target_with_normals);
	p_icp.setTransformationEpsilon(1e-10);    // 为终止条件设置最小转换差异
	p_icp.setMaxCorrespondenceDistance(10);   // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	p_icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	p_icp.setMaximumIterations(35);           // 最大迭代次数		

	//p_icp.setUseSymmetricObjective(true);若设置为true则变为另一个算法	
	pcl::PointCloud<pcl::PointNormal>::Ptr p_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	p_icp.align(*p_icp_cloud);
	cout << "\nICP has converged, score is " << p_icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << p_icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, p_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//PCL 点到面的ICP精配准（LS线性最小二乘优化）
void ICP_LS(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------拼接点云与法线信息-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------点到面的ICP-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
	/*点到面的距离函数构造方法一*/
	pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>);
	/*点到面的距离函数构造方法二*/
	//typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;
	// boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(PointToPlane);
	icp.setInputSource(source_with_normals);
	icp.setInputTarget(target_with_normals);
	icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(10);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	icp.setMaximumIterations(35);           // 最大迭代次数
	pcl::PointCloud<pcl::PointNormal>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	icp.align(*icp_cloud);

	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//PCL 点到面的ICP算法（LM非线性最小二乘优化）
#include <pcl/registration/transformation_estimation_point_to_plane.h>
void ICP_LM(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------拼接点云与法线信息-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------点到面的icp（非线性优化）-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal>lm_icp;
	pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>);
	lm_icp.setTransformationEstimation(PointToPlane);
	lm_icp.setInputSource(source_with_normals);
	lm_icp.setInputTarget(target_with_normals);
	lm_icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	lm_icp.setMaxCorrespondenceDistance(10);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	lm_icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	lm_icp.setMaximumIterations(35);           // 最大迭代次数
	pcl::PointCloud<pcl::PointNormal>::Ptr lm_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	lm_icp.align(*lm_icp_cloud);
	cout << "\nICP has converged, score is " << lm_icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << lm_icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, lm_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//----------------------改进的ICP算法----------------------

//PCL Trimmed ICP
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
//“transform”: 未找到匹配的重载函数;“aux”: 不是类或命名空间名称
void print4x4Matrix(const Eigen::Matrix4d & matrix)
{
	printf("Rotation matrix :\n");
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(0, 0), matrix(0, 1), matrix(0, 2));
	printf("R = | %6.3f %6.3f %6.3f | \n", matrix(1, 0), matrix(1, 1), matrix(1, 2));
	printf("    | %6.3f %6.3f %6.3f | \n", matrix(2, 0), matrix(2, 1), matrix(2, 2));
	printf("Translation vector :\n");
	printf("t = < %6.3f, %6.3f, %6.3f >\n\n", matrix(0, 3), matrix(1, 3), matrix(2, 3));
}
void ICP_Trimmed(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud)
{
	int iterations = 35;
	pcl::console::TicToc time;
	time.tic();
	Eigen::Matrix4d transformation_matrix = Eigen::Matrix4d::Identity();
	//icp实现
	time.tic();
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaximumIterations(iterations);
	icp.setMaxCorrespondenceDistance(15);   //设置最大的对应点距离
	icp.setTransformationEpsilon(1e-10);      //设置精度
	icp.setEuclideanFitnessEpsilon(0.01);
	icp.setInputSource(source_cloud);
	icp.setInputTarget(target_cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << iterations << "ICP iteration(s) in " << time.toc() / 1000 << "s" << endl;
	//输出RT
	if (icp.hasConverged())
	{
		cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
		cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << endl;
		transformation_matrix = icp.getFinalTransformation().cast<double>();
		print4x4Matrix(transformation_matrix);
	}
	else
		PCL_ERROR("\nICP has not converged.\n");
	//trimmed icp实现
	time.tic();
	pcl::recognition::TrimmedICP<pcl::PointXYZ, double> Tricp;
	Tricp.init(target_cloud);     // target
	float sigma = 0.96;
	int Np = source_cloud->size();// num_source_points
	int Npo = Np * sigma;         // num_source_points_to_use
	Tricp.setNewToOldEnergyRatio(sigma);//数字越大配准越准确
	Eigen::Matrix4d transformation_matrix1 = Eigen::Matrix4d::Identity();
	Tricp.align(*icp_cloud, Npo, transformation_matrix1);
	cout << "Applied Trimmed ICP iteration(s) in " << time.toc() / 1000 << "s" << endl;
	cout << "Trimmed icp pose:" << endl;
	print4x4Matrix(transformation_matrix1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr Tricp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*icp_cloud, *Tricp_cloud, transformation_matrix1);
	visualize_registration(source_cloud, target_cloud, Tricp_cloud);
}

//PCL 使用GICP对点云配准
#include <pcl/registration/gicp.h>  
//--通过协方差矩阵起到类似于权重的作用，消除不好的对应点在求解过程中的作用
//--可以退化为ICP，且存在唯一解
void GICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	pcl::console::TicToc time;
	time.tic();
	//-----------------初始化GICP对象-------------------------
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
	//-----------------KD树加速搜索---------------------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(target);
	gicp.setSearchMethodSource(tree1);
	gicp.setSearchMethodTarget(tree2);
	//-----------------设置GICP相关参数-----------------------
	gicp.setInputSource(source);  //源点云
	gicp.setInputTarget(target);  //目标点云
	gicp.setMaxCorrespondenceDistance(100); //设置对应点对之间的最大距离
	gicp.setTransformationEpsilon(1e-10);   //为终止条件设置最小转换差异
	gicp.setEuclideanFitnessEpsilon(0.001);  //设置收敛条件是均方误差和小于阈值，停止迭代
	gicp.setMaximumIterations(35); 
	// 计算需要的刚体变换以便将输入的源点云匹配到目标点云
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	gicp.align(*icp_cloud);
	//---------------输出必要信息到显示--------------------
	cout << "Applied " << 35 << " GICP iterations in " << time.toc() / 1000 << " s" << endl;
	cout << "\nGICP has converged, score is " << gicp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << gicp.getFinalTransformation() << endl;
	// 使用变换矩阵对为输入点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, gicp.getFinalTransformation());
	visualize_registration(source, target, icp_cloud);
}

//PCL 目标函数对称的ICP算法
#if 0
void ICP_symmetric(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------拼接点云与法线信息-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------点到面的symm_icp-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> symm_icp;
	//PCL1.11.1
	pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>);
	symm_icp.setTransformationEstimation(PointToPlane);
	symm_icp.setInputSource(source_with_normals);
	symm_icp.setInputTarget(target_with_normals);
	symm_icp.setTransformationEpsilon(1e-10);   // 为终止条件设置最小转换差异
	symm_icp.setMaxCorrespondenceDistance(10);  // 设置对应点对之间的最大距离（此值对配准结果影响较大）。
	symm_icp.setEuclideanFitnessEpsilon(0.001);  // 设置收敛条件是均方误差和小于阈值， 停止迭代；
	symm_icp.setMaximumIterations(50);           // 最大迭代次数
	pcl::PointCloud<pcl::PointNormal>::Ptr symm_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	symm_icp.align(*symm_icp_cloud);
	cout << "\nsymPlaneICP has converged, score is " << symm_icp.getFitnessScore() << endl;
	cout << "变换矩阵：\n" << symm_icp.getFinalTransformation() << endl;
	// 使用创建的变换对为输入的源点云进行变换
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, symm_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}
#endif

//非线性加权最小二乘优化的点到面ICP算法
using point_normal = pcl::PointCloud<pcl::PointNormal>;
//================计算法线=====================
point_normal::Ptr wf(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//建立kdtree来进行近邻点集搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setNumberOfThreads(8);
	n.setInputCloud(input_cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);
	//将点云数据与法向信息拼接
	pcl::PointCloud<pcl::PointNormal>::Ptr input_cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*input_cloud, *normals, *input_cloud_normals);
	return input_cloud_normals;
}
#include <pcl/registration/icp_nl.h>//LM-ICP、非线性
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
void ICP_NoneLLS(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	pcl::console::TicToc time;
	point_normal::Ptr tn = wf(target);
	point_normal::Ptr sn = wf(source);
	cout << "法线计算完毕！！！" << endl;
	time.tic();
	//=======================初始化=====================
	pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> icp;
	//=======PointToPlaneWeighted计算点到面的距离================
	typedef pcl::registration::TransformationEstimationPointToPlaneWeighted <pcl::PointNormal, pcl::PointNormal> PointToPlane;
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	//================参数设置=================
	icp.setTransformationEstimation(point_to_plane);
	icp.setInputSource(sn);
	icp.setInputTarget(tn);
	icp.setTransformationEpsilon(1e-10);   //为终止条件设置最小转换差异
	icp.setMaxCorrespondenceDistance(10); //设置对应点对之间的最大距离（此值对配准结果影响较大）。
	icp.setEuclideanFitnessEpsilon(0.0001);  //设置收敛条件是均方误差和小于阈值， 停止迭代；
	icp.setMaximumIterations(35); //最大迭代次数；  
	pcl::PointCloud<pcl::PointNormal>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() / 1000 << "s" << endl;
	cout << "变换矩阵：\n" << icp.getFinalTransformation() << endl;
	pcl::transformPointCloud(*sn, *icp_cloud, icp.getFinalTransformation());
	//=================可视化对象====================
	boost::shared_ptr<pcl::visualization::PCLVisualizer>
		viewer_final(new pcl::visualization::PCLVisualizer("配准结果"));
	viewer_final->setBackgroundColor(0, 0, 0); 
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		target_color(target, 255, 0, 0);
	viewer_final->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "target cloud");
	// 对源点云着色可视化 (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		input_color(source, 0, 255, 0);
	viewer_final->addPointCloud<pcl::PointXYZ>(source, input_color, "input cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "input cloud");
	// 对转换后的源点云着色 (blue)可视化.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal>
		output_color(icp_cloud, 0, 0, 255);
	viewer_final->addPointCloud<pcl::PointNormal>(icp_cloud, output_color, "output cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "output cloud");
	while (!viewer_final->wasStopped())
	{
		viewer_final->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//LM-ICP 实现点云精配准
#include <pcl/filters/random_sample.h>//采取固定数量的点云
#include <pcl/registration/icp_nl.h> //LM-ICP
void  random_sample_point(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_rsf, double count)
{
	pcl::RandomSample<pcl::PointXYZ> rs_src;
	rs_src.setInputCloud(cloud);
	rs_src.setSample(count);
	rs_src.filter(*cloud_rsf);
}
void LMICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	pcl::console::TicToc time;
	// ----------------------随机采样特征点--------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	random_sample_point(source, s_k, 3000);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	random_sample_point(target, t_k, 3000);
	time.tic();
	// ------------------------LM-ICP--------------------------------
	pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> lmicp;
	lmicp.setInputSource(s_k);
	lmicp.setInputTarget(t_k);
	lmicp.setTransformationEpsilon(1e-10);    //为终止条件设置最小转换差异
	lmicp.setMaxCorrespondenceDistance(10);   //设置对应点对之间的最大距离（此值对配准结果影响较大）。
	lmicp.setEuclideanFitnessEpsilon(0.0001); //设置收敛条件是均方误差和小于阈值， 停止迭代；
	lmicp.setMaximumIterations(35);           //最大迭代次数；  
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	lmicp.align(*icp_cloud);
	cout << "Applied " << 35 << " LM-ICP iterations in " << time.toc() / 1000 << "s" << endl;
	cout << "变换矩阵：\n" << lmicp.getFinalTransformation() << endl;
	// 对源点云进行变换
	pcl::transformPointCloud(*source, *icp_cloud, lmicp.getFinalTransformation());
	visualize_registration(source, target, icp_cloud);
}

//----------------------基于概率模型的算法----------------------

//PCL 3D-NDT 算法实现点云配准--构建多维变量的正态分布
#include <pcl/registration/ndt.h>               // NDT配准
#include <pcl/filters/approximate_voxel_grid.h> // 体素滤波
void _3DNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud)
{
	pcl::console::TicToc time;
	if (source_cloud->empty() || target_cloud->empty())
	{
		cout << "请确认点云文件名称是否正确" << endl;
		return;
	}
	else {
		cout << "从目标点云读取 " << target_cloud->size() << " 个点" << endl;
		cout << "从源点云中读取 " << source_cloud->size() << " 个点" << endl;
	}

	//将输入的源点云过滤到原始尺寸的大概10%以提高匹配的速度。
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
	approximate_voxel_filter.setInputCloud(source_cloud);
	approximate_voxel_filter.setLeafSize(0.1, 0.1, 0.1);
	approximate_voxel_filter.filter(*filtered_cloud);
	cout << "Filtered cloud contains " << filtered_cloud->size() << " data points " << endl;
	// -------------NDT进行配准--------------
	time.tic();
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setStepSize(4);                 // 为More-Thuente线搜索设置最大步长
	ndt.setResolution(0.1);             // 设置NDT网格结构的分辨率（VoxelGridCovariance）
	ndt.setMaximumIterations(35);       // 设置匹配迭代的最大次数
	ndt.setInputSource(filtered_cloud);	// 设置要配准的点云
	ndt.setInputTarget(target_cloud);   // 设置点云配准目标
	ndt.setTransformationEpsilon(0.01); // 为终止条件设置最小转换差异
	pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*output_cloud);
	cout << "NDT has converged:" << ndt.hasConverged()
		<< " score: " << ndt.getFitnessScore() << endl;
	cout << "Applied " << ndt.getMaximumIterations() << " NDT iterations in " << time.toc() << " ms" << endl;
	cout << "变换矩阵：\n" << ndt.getFinalTransformation() << endl;
	//使用变换矩阵对未过滤的源点云进行变换
	pcl::transformPointCloud(*source_cloud, *output_cloud, ndt.getFinalTransformation());
	visualize_registration(source_cloud, target_cloud, output_cloud);
}


//------------------------------------------------------------------------------------------------//

void main()
{
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("../roorm.pcd", *cloud) == -1)
	{
		PCL_ERROR("Cloudn't read file!");
		return;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>);

	//------------------------------------------------------------------------------------------------//
	//_4PCS(cloud, cloud);
	//K4PCS(cloud, cloud);
	//RANSACEx(cloud, cloud);
	//SAC_IA(cloud, cloud);
	//PCA(cloud, cloud);
	//ICP(cloud, cloud);
	//KDICP(cloud, cloud);
	//SACIA_NDT(cloud, cloud);
	//NDT_ICP(cloud, cloud);
	//SACIA_ICP(cloud, cloud);
	//ICP_class(cloud, cloud);
	//ICP_LS(cloud, cloud);
	//ICP_LM(cloud, cloud);
	//ICP_Trimmed(cloud, cloud);
	//GICP(cloud, cloud);
	//ICP_NoneLLS(cloud, cloud);
	//LMICP(cloud, cloud);
	_3DNDT(cloud, cloud);
	//------------------------------------------------------------------------------------------------//
	system("pause");
	return;
}