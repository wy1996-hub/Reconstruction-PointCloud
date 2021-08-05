#include <iostream>
#include <pcl/io/pcd_io.h>  
#include <pcl/point_cloud.h>  
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>
#include <pcl/common/common_headers.h>
#include <Eigen/Eigenvalues>
#include <Eigen/Dense>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl\features\normal_3d.h>

using namespace std;
using namespace Eigen;

/*
	PCL特征模块：
	包含了用于（点云数据估计三维特征）的数据结构和功能函数，
	三维特征是空间中某个三维点或者位置的表示，它是基于点周围的可用信息来描述几何的图形的一种表示。
	在三维空间中，查询点周围的方法一般是K领域查找。
	三维空间的特征点物理意义上与图像类似，都是使用一些具有显著特征的点来表示整个点云。

	eigen:本征,即物质本身的特征
	在线性代数中，里面有特征值，此特征值也称本征值

	PCL点云特征描述与提取：
	3D点云特征描述与提取是点云信息处理中最基础也是最关键的一部分。
	点云的识别、分割、重采样、配准曲面重建等处理，大部分算法都依赖特征描述符提取的结果。
	从尺度上来分，一般分为局部特征的描述和全局特征的描述，
	例如局部的法线等几何形状特征的描述，全局的拓朴特征的描述，都属于3D点云特征描述与提取的范畴。

	常用的特征描述算法有：
	1. 法线和曲率计算 normal_3d_feature 、
	2. 特征值分析、
	3. PFH  点特征直方图描述子 （统计点法线角度差值row pitch yaw）   n*k^2、
	4. FPFH 快速点特征直方图描述子 FPFH是PFH的简化形式  n*k
	5. 3D Shape Context（3D形状内容描述子）
	pcl::ShapeContext3DEstimation< PointInT, PointNT, PointOutT >
	实现3D形状内容描述子算法
	6. 纹理特征， 2d-3d点对  特征描述子（orb可以）
	7. Spin Image
	8. VFH  视点特征直方图(Viewpoint Feature Histogram 视角方向与点法线方向夹角)
	9. NARF 关键点特征  pcl::NarfKeypoint narf特征 pcl::NarfDescriptor(深度图边缘)
	10. RoPs 特征(Rotational Projection Statistics)
	11. (GASD）全局一致的空间分布描述子特征 Globally Aligned Spatial Distribution (GASD) descriptors
	12. 旋转图像（spin iamge）
	旋转图像最早是由johnson提出的特征描述子，主要用于3D场景中的曲面匹配和模型识别。

	PCL 描述三维特征相关基础：
	在原始表示形式下，点的定义是用笛卡尔坐标系坐标 x, y, z 相对于一个给定的原点来简单表示的三维映射系统的概念。
	假定坐标系的原点不随着时间而改变，这里有两个点p1和p2分别在时间t1和t2捕获，有着相同的坐标，
	对这两个点作比较其实是属于不适定问题（ill-posed problem），
	因为虽然相对于一些距离测度它们是相等的，但是它们取样于完全不同的表面，
	因此当把它们和临近的其他环境中点放在一起时，它们表达着完全不同的信息，
	这是因为在t1和t2之间局部环境有可能发生改变。一些获取设备也许能够提供取样点的额外数据，
	例如强度或表面反射率等，甚至颜色，然而那并不能完全解决问题。单从两个点之间来，对比仍然是不适定问题。
	由于各种不同需求需要进行对比以便能够区分曲面空间的分布情况，应用软件要求有更好的特征度量方式，
	因此作为一个单一实体的三维点概念和笛卡尔坐标系被淘汰了，出现了一个新的概念取而代之：局部描述子（local descriptor）。
	文献中对这一概念的描述有许多种不同的命名，如：形状描述子（shape descriptors）或几何特征（geometric features），
	可都统称为点特征表示。
	通过包括周围的领域，特征描述子能够表征采样表面的几何性质，
	它有助于解决不适定的对比问题，理想情况下相同或相似表面上的点的特征值将非常相似（相对特定度量准则），
	而不同表面上的点的特征描述子将有明显差异。

	下面几个条件，通过能否获得相同的局部表面特征值，可以判定点特征表示方式的优劣：
	1、平移选转不变性（刚体变换）：即三维旋转和三维平移变化 不会影响特征向量F估计
	2、抗密度干扰性（改变采样密度）：原则上，一个局部表面小块的采样密度无论是大还是小，都应该有相同的特征向量值
	3、对噪声具有稳定性（噪声）：数据中有轻微噪声的情况下，点特征表示在它的特征向量中必须保持相同或者极其相似的值
	对于点云局部特征描述子的质量评估主要体现在以下五个方面：
	1、刚体变换不变性—当目标或者传感器进行了旋转和平移等刚体运动时，在对应点提取的局部特征应当具有不变性；
	2、描述性—因为点和点之间是一一对应的，局部特征需要具有一定的鉴别能力来区分相同目标之内或不同目标之间的局部几何模式；
	3、鲁棒性—局部特征需要对由于自遮挡、遮挡、嘈杂、测量距离变化等引起的孔洞、噪声、数据分辨率变化等干扰具有鲁棒性；
	4、时效性—机器人等实时性高的应用平台对于局部特征计算的速度有着严格的要求；
	5、紧凑性—局部特征的紧凑性取决于其占用的内存资源，主要受特征维度和特征数据类型影响，紧凑性高的局部特征有着维度低、二值化等特点。

	然而，现有的点云局部特征仍难以充分满足上述需求，而且对于数据模态和应用场景的变化较为敏感。
	对于特征匹配的质量评估一方面体现在抗离群点（错误匹配）的干扰能力，另一方面体现在匹配的效率。
	评估研究表明大部分现有特征匹配算法存在耗时长、对高离群点率敏感的缺陷。

	通常，PCL中特征向量利用快速 kd-tree 查询 ，使用近似法来计算查询点的最近邻元素，
	通常有两种查询类型：K邻域查询，半径搜索两中方法

	三维点匹配问题，以源点云和目标点云来命名两组待匹配的点云序列，
	当两组点云之间发生了旋转、平移等刚性变换时，称之为刚性点云匹配数据，常为传感器在不同视角或不同场景下捕获的刚体目标点云数据序列；
	当两组点云之间发生了伸缩、仿射、变形等非刚性变换时，则称之为非刚性点云匹配数据。
	对于刚性点云匹配数据之间的点对应关系建立通常包含两步：局部几何特征描述和特征匹配。
	局部几何特征描述的目的是用一个特征向量来表征隐含在局部曲面内的空间信息和几何信息；
	特征匹配的目的是通过度量两组点云特征集中任意两个特征的相似性确定初始匹配并筛选出正确的匹配子集。
	当点云数据体量较大时，在初始点云中检测若干稀疏且具有区分能力的关键点是一种常用的预处理方案。

	二十一世纪初期，曾有一系列点云关键点检测子被陆续提出，
	例如三维 Harris 检测子（Harris 3D，H3D）和本征形状签名（intrinsic shape signature，ISS）。
	但最近关于点云关键点检测子评估研究指出现有的检测子在现实应用场景数据中可重复性低且较为耗时。
	另外，取代传统关键点检测子的均匀采样和随机采样方法在目标识别以及点云配准应用中均被证明为一种快速有效的关键点选取方式。
	因此，当前的研究热点主要集中在点云局部特征描述和特征匹配两方面。
*/

//---------------------------------点云的属性-------------------------------------//

//PCL 特征(pcl::Feature)模块使用方法
void pclFeatureUSE(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	/*
		1  输入数据
		setinputcloud输入点云数据
		setindices输入点云中带索引的点云数据
		setsearchsurface点云搜索面
		eg:当有一个非常密集的输入点云时，我们不想对所有的点进行特征估计，而是（找关键点/体素滤波下采样点云）；
		然后在处理过后的点云数据中进行特征估计。即把过滤后的点云传递给特征估计算法，把原始数据设为搜索面，提高效率。

		表面法线是一个表面的重要属性。
	*/

#if 0
	//创建normal estimation类，并将输入数据集传递给它
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	//创建一个空的kdtree表示，并将其传递给normal estimation对象
	//它的内容将基于给定的输入数据集填充到对象内部（因为没有给出其他搜索面）
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	//输出数据集
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	//使用半径为3cm的球体中的所有邻居
	ne.setRadiusSearch(0.03);
	//计算特征
	ne.compute(*cloud_normals);
	//法向量点云大小应与输入点云具有相同的大小
#endif

#if 0
	//创建一组要使用的索引。为了简单起见，我们将使用云中前10%的点
	std::vector<int> indices(std::floor(cloud->size() / 10));
	for (std::size_t i = 0; i < indices.size(); ++i) indices[i] = i;
	//创建normal estimation类，并将输入数据集传递给它
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud);
	//通过索引
	boost::shared_ptr<std::vector<int>> indicesptr(new std::vector<int>(indices));//使用indices初始化
	ne.setIndices(indicesptr);
	//创建一个空的kdtree表示，并将其传递给normal estimation对象
	//它的内容将基于给定的输入数据集填充到对象内部（因为没有给出其他搜索面）
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	//输出数据集
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.03);
	ne.compute(*cloud_normals);
#endif

#if 0
	//创建cloud的下采样版本cloud_downsampled
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_downsampled);
	//传递原始数据（下采样前）作为搜索曲面
	ne.setSearchSurface(cloud);
	// 其内容将基于给定的曲面数据集填充到对象内部
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);
	ne.setRadiusSearch(0.03);
	ne.compute(*cloud_normals);
#endif


}

//PCA 计算点云的特征值
void calPCAFeature(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr KeyPoint)
{
	/*
		基于局部表面拟合的方法进行法向量估计：
		点云采样表面处处光滑的情况下，任何点的局部邻域都可以用平面进行很好的拟合；
		对于点云中每个扫描点，搜索与其最近邻的K个相邻点，计算这些点最小二乘意义上的局部平面；
		可以认为K个最近邻点拟合出的平面的法向量即为当前扫描点的法向量；
		平面的法向量可以有主成分分析PCA得到。
		协方差矩阵的特征值l0<=l1<=l2;则该点的表面曲率为l0/(l0+l1+l2)。
	*/

	cout << "Number of points in the  Input cloud is:" << cloud->size() << endl;

	// K近邻搜索
	int KNumbersNeighbor = 10; 
	std::vector<int> NeighborsKNSearch(KNumbersNeighbor);
	std::vector<float> NeighborsKNSquaredDistance(KNumbersNeighbor);
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointXYZ searchPoint;

	int* NumbersNeighbor = new  int[cloud->points.size()];
	double* SmallestEigen = new  double[cloud->points.size()];
	double* MiddleEigen = new  double[cloud->points.size()];
	double* LargestEigen = new  double[cloud->points.size()];
	double* DLS = new  double[cloud->points.size()];
	double* DLM = new  double[cloud->points.size()];
	double* DMS = new  double[cloud->points.size()];
	double* Sigma = new  double[cloud->points.size()];

	for (size_t i = 0; i < cloud->points.size(); i++)
	{
		searchPoint.x = cloud->points[i].x;
		searchPoint.y = cloud->points[i].y;
		searchPoint.z = cloud->points[i].z;

		if (kdtree.nearestKSearch(searchPoint, KNumbersNeighbor, NeighborsKNSearch, NeighborsKNSquaredDistance) > 0)
			NumbersNeighbor[i] = NeighborsKNSearch.size();
		else
			NumbersNeighbor[i] = 0;

		//计算协方差矩阵

		//计算均值
		float Xmean; float Ymean; float Zmean;
		float sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += cloud->points[NeighborsKNSearch[ii]].x;
		}
		Xmean = sum / NumbersNeighbor[i];
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += cloud->points[NeighborsKNSearch[ii]].y;
		}
		Ymean = sum / NumbersNeighbor[i];
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) {
			sum += cloud->points[NeighborsKNSearch[ii]].z;
		}
		Zmean = sum / NumbersNeighbor[i];

		//计算方差
		float CovXX; float CovXY; float CovXZ; float CovYX; float CovYY; float CovYZ; float CovZX; float CovZY; float CovZZ;
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].x - Xmean));
		}
		CovXX = sum / (NumbersNeighbor[i] - 1);
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
		}
		CovXY = sum / (NumbersNeighbor[i] - 1);
		CovYX = CovXY;//相等
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].x - Xmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovXZ = sum / (NumbersNeighbor[i] - 1);
		CovZX = CovXZ;
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].y - Ymean));
		}
		CovYY = sum / (NumbersNeighbor[i] - 1);
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].y - Ymean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovYZ = sum / (NumbersNeighbor[i] - 1);
		CovZY = CovYZ;
		sum = 0.00;
		for (size_t ii = 0; ii < NeighborsKNSearch.size(); ++ii) 
		{
			sum += ((cloud->points[NeighborsKNSearch[ii]].z - Zmean) * (cloud->points[NeighborsKNSearch[ii]].z - Zmean));
		}
		CovZZ = sum / (NumbersNeighbor[i] - 1);

		
		Matrix3f Cov;
		Cov << CovXX, CovXY, CovXZ, CovYX, CovYY, CovYZ, CovZX, CovZY, CovZZ;

		//计算特征值和特征向量
		SelfAdjointEigenSolver<Matrix3f> eigensolver(Cov);
		if (eigensolver.info() != Success) abort();

		double EigenValue1 = eigensolver.eigenvalues()[0];
		double EigenValue2 = eigensolver.eigenvalues()[1];
		double EigenValue3 = eigensolver.eigenvalues()[2];
		double Smallest = 0.00; double Middle = 0.00; double Largest = 0.00;
		Smallest = min(EigenValue3, min(EigenValue1, EigenValue2));

		if (EigenValue1 <= EigenValue2 && EigenValue1 <= EigenValue3) 
		{
			Smallest = EigenValue1;
			if (EigenValue2 <= EigenValue3) { Middle = EigenValue2; Largest = EigenValue3; }
			else { Middle = EigenValue3; Largest = EigenValue2; }
		}
		if (EigenValue1 >= EigenValue2 && EigenValue1 >= EigenValue3)
		{
			Largest = EigenValue1;
			if (EigenValue2 <= EigenValue3) { Smallest = EigenValue2; Middle = EigenValue3; }
			else { Smallest = EigenValue3; Middle = EigenValue2; }
		}
		if ((EigenValue1 >= EigenValue2 && EigenValue1 <= EigenValue3) || (EigenValue1 <= EigenValue2 && EigenValue1 >= EigenValue3))
		{
			Middle = EigenValue1;
			if (EigenValue2 >= EigenValue3) { Largest = EigenValue2; Smallest = EigenValue3; }
			else { Largest = EigenValue3; Smallest = EigenValue2; }
		}

		SmallestEigen[i] = Smallest;//最小特征值 lamda1
		MiddleEigen[i] = Middle;    //中间特征值 lamda2
		LargestEigen[i] = Largest;  //最大特征值 lamda3
		DLS[i] = std::abs(SmallestEigen[i] / LargestEigen[i]);          // lamda1/lamda3 ;
		DLM[i] = std::abs(MiddleEigen[i] / LargestEigen[i]);            // lamda2/lamda3 ;
		DMS[i] = std::abs(SmallestEigen[i] / MiddleEigen[i]);           // lamda1/lamda2 ;
		Sigma[i] = (SmallestEigen[i]) / (SmallestEigen[i] + MiddleEigen[i] + LargestEigen[i]); //曲率
		if (DMS[i] > 0.002)
			KeyPoint->push_back(cloud->points[i]);
	}
	cout << " KeyPoints个数为 " << KeyPoint->size() << endl;
}

//点云的曲率及计算
/*
	数学概念上的曲率：曲线弯曲程度的度量
	主曲率、平均曲率、高斯曲率（描述曲面凹凸性质的量，越高越不光滑）

	点云中任意一点都存在某曲面逼近该点的邻域点云，一点处的曲率可用该点及其领域点拟合的局部曲面的曲率来表征。
	通过最小二乘，可以用二次曲面来表征局部区域。

	表面曲率：点云数据表面的特征值来描述点云表面变化程度的一个概念，与数学意义上的曲率不同
	计算：
	二次曲面拟合求点云曲率
	利用相邻点的法向量求一点的曲率

	不同符号的平均曲率H和高斯曲率K代表不同的曲面凹凸形状

	pcl中只有两种计算曲率的调用函数：
	1、PrincipalCurvaturesEstimation:类似利用相邻点的法向量估计一个点的主曲率的方法--PrincipalCurvaturesEstimationCAN
	2、NormalEstimation:基于PCA主成分分析实现的，具体实现细节参照：pcl计算点云法向量并显示
	关于主曲率的主方向：pcl中实现的计算主曲率的方法，只能计算出一个主曲率方向，
	若想得到两个主曲率方向，只能自己写，具体实现细节参照：利用相邻点的法向量估计一个点的曲率
*/

//二次曲面拟合求点云的曲率
void curvatureByQuadricSurfaceFitting(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	//---------------矩阵运算相关参数-------------------
	MatrixXd Mean_curvature;//平均曲率
	Mean_curvature.resize(cloud->size(), 1);    //初始化矩阵 cloud->size * 1；
	MatrixXd Gauss_curvature;//高斯曲率
	Gauss_curvature.resize(cloud->size(), 1);
	Matrix<double, 6, 6>Q;                      //求解方程组系数矩阵
	Matrix<double, 6, 6>Q_Inverse;		        //系数矩阵的逆矩阵
	Matrix<double, 6, 1>B;					    //方程组右值矩阵

	Q.setZero();//初始化矩阵元素全为0
	Q_Inverse.setZero();
	B.setZero();
	double a = 0, b = 0, c = 0, d = 0, e = 0, f = 0;//二次曲面方程系数
	double u = 0, v = 0;							//二次曲面参数方程参数
	double E = 0, G = 0, F = 0, L = 0, M = 0, N = 0;//曲面第一、第二基本量
	double Meancurvature = 0, Gausscurvature = 0;	//平均曲率、高斯曲率

	//------------K近邻搜索查找最近点----------------
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	for (size_t i_p = 0; i_p < cloud->points.size(); ++i_p) 
	{
		pcl::PointXYZ searchPoint = cloud->points[i_p]; //设置查找点
		int K = 10;                                //设置需要查找的近邻点个数
		vector<int> pNK(K);                       // 保存每个近邻点的索引
		vector<float> pointNKNSquaredDistance(K); // 保存每个近邻点与查找点之间的欧式距离平方

		if (kdtree.nearestKSearch(searchPoint, K, pNK, pointNKNSquaredDistance) > 0)
		{
			for (size_t i = 0; i < pNK.size(); ++i) 
			{
				cloud->points[pNK[i]];
				//---------------构建最小二乘平差计算矩阵-------------------
				Q(0, 0) += pow(cloud->points[pNK[i]].x, 4);
				Q(1, 0) = Q(0, 1) += pow(cloud->points[pNK[i]].x, 3) * cloud->points[pNK[i]].y;
				Q(2, 0) = Q(0, 2) += pow(cloud->points[pNK[i]].x * cloud->points[pNK[i]].y, 2);
				Q(3, 0) = Q(0, 3) += pow(cloud->points[pNK[i]].x, 3);
				Q(4, 0) = Q(0, 4) += pow(cloud->points[pNK[i]].x, 2) * cloud->points[pNK[i]].y;
				Q(5, 0) = Q(0, 5) += pow(cloud->points[pNK[i]].x, 2);
				Q(1, 1) += pow(cloud->points[pNK[i]].x * cloud->points[pNK[i]].y, 2);
				Q(2, 1) = Q(1, 2) += pow(cloud->points[pNK[i]].y, 3) * cloud->points[pNK[i]].x;
				Q(3, 1) = Q(1, 3) += pow(cloud->points[pNK[i]].x, 2) * cloud->points[pNK[i]].y;
				Q(4, 1) = Q(1, 4) += pow(cloud->points[pNK[i]].y, 2) * cloud->points[pNK[i]].x;
				Q(5, 1) = Q(1, 5) += cloud->points[pNK[i]].x * cloud->points[pNK[i]].y;
				Q(2, 2) += pow(cloud->points[pNK[i]].y, 4);
				Q(3, 2) = Q(2, 3) += pow(cloud->points[pNK[i]].y, 2) * cloud->points[pNK[i]].x;
				Q(4, 2) = Q(2, 4) += pow(cloud->points[pNK[i]].y, 3);
				Q(5, 2) = Q(2, 5) += pow(cloud->points[pNK[i]].y, 2);
				Q(3, 3) += pow(cloud->points[pNK[i]].x, 2);
				Q(4, 3) = Q(3, 4) += cloud->points[pNK[i]].x * cloud->points[pNK[i]].y;
				Q(5, 3) = Q(3, 5) += cloud->points[pNK[i]].x;
				Q(4, 4) += pow(cloud->points[pNK[i]].y, 2);
				Q(5, 4) = Q(4, 5) += cloud->points[pNK[i]].y;
				Q(5, 5) += 1;

				B(0, 0) += pow(cloud->points[pNK[i]].x, 2) * cloud->points[pNK[i]].z;
				B(1, 0) += cloud->points[pNK[i]].x * cloud->points[pNK[i]].y * cloud->points[pNK[i]].z;
				B(2, 0) += pow(cloud->points[pNK[i]].y, 2) * cloud->points[pNK[i]].z;
				B(3, 0) += cloud->points[pNK[i]].x * cloud->points[pNK[i]].z;
				B(4, 0) += cloud->points[pNK[i]].y * cloud->points[pNK[i]].z;
				B(5, 0) += cloud->points[pNK[i]].z;

				//---------------求解矩阵------------------
				Q_Inverse = Q.inverse();
				for (int j = 0; j < 6; ++j)
				{
					a += Q_Inverse(0, j) * B(j, 0);
					b += Q_Inverse(1, j) * B(j, 0);
					c += Q_Inverse(2, j) * B(j, 0);
					d += Q_Inverse(3, j) * B(j, 0);
					e += Q_Inverse(4, j) * B(j, 0);
					f += Q_Inverse(5, j) * B(j, 0);
				}
				// 根据所求曲面方程的系数计算曲面第一第二基本量
				u = 2 * a * cloud->points[pNK[i]].x + b * cloud->points[pNK[i]].y + d;
				v = 2 * c * cloud->points[pNK[i]].y + b * cloud->points[pNK[i]].x + e;
				E = 1 + u * u;
				F = u * v;
				G = 1 + v * v;
				double u_v = sqrt(1 + u * u + v * v);
				L = (2 * a) / u_v;
				M = b / u_v;
				N = (2 * c) / u_v;
				// 高斯曲率
				Gausscurvature = (L * N - M * M) / (E * G - F * F);
				Gauss_curvature(i_p, 0) = Gausscurvature;
				// 平均曲率
				Meancurvature = (E * N - 2 * F * M + G * L) / (2 * E * G - 2 * F * F);
				Mean_curvature(i_p, 0) = Meancurvature;
			}
		}
	}
	cout << "高斯曲率为：" << Gauss_curvature(4, 0) << endl;
}

//PCL 计算点云的主曲率--PrincipalCurvaturesEstimation
#include <pcl/features/principal_curvatures.h> //计算主曲率
//Error: no override found for 'vtkActor'.
#include <vtkAutoInit.h>
VTK_MODULE_INIT(vtkRenderingOpenGL);
void calPrincipalCurvature(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	/*
		PointCloud可以理解为一个容器、类模板

		注意:
		1、pcl::KdTreeFLANN<pcl::PointXYZ>基于FLANN库实现的KDTree,
		除了可以搜索点还能搜索特征pcl::KdTreeFLANN<FeatureT>,一般用于寻找匹配点对
		2、其他操作方法中使用到KDtree的时候，通常使用搜索模块封装的Kdtree类，
		即pcl::search::KdTree< PointT, Tree >,用于在Kdtree结构中执行搜索方法，
		此类与FLANN实现的Kdtree类均继承了pcl::Kdtree，
		另外搜索模块封装的Kdtree还继承了搜索类pcl::search::Search< PointT >，
		这也是其与FLANN库实现的Kdtree的区别。
		使用方面基本一致，只是实现方式不同，FLANN目的更多是单独使用时加速。
	*/

	// --------------------计算点云的法线------------------
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);      //设置邻域点搜索方式
	// n.setRadiusSearch (0.03);  //设置KD树搜索半径
	n.setKSearch(10);
	//定义一个新的点云储存含有法线的值
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);         //计算出来法线的值

	//-----------------------主曲率计算---------------------
	pcl::PrincipalCurvaturesEstimation<pcl::PointXYZ, pcl::Normal, pcl::PrincipalCurvatures> p;
	p.setInputCloud(cloud);      //提供原始点云(没有法线)
	p.setInputNormals(normals);  //为点云提供法线
	p.setSearchMethod(tree);     //使用与法线估算相同的KdTree
	//p.setRadiusSearch(1.0);
	p.setKSearch(10);
	// 计算主曲率
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr pri(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	p.compute(*pri);
	cout << "output points.size: " << pri->points.size() << endl;
	// 显示和检索第0点的主曲率。
	cout << "最大曲率;" << pri->points[0].pc1 << endl;//输出最大曲率
	cout << "最小曲率:" << pri->points[0].pc2 << endl;//输出最小曲率
	//输出主曲率方向（最大特征值对应的特征向量）
	cout << "主曲率方向;" << endl;
	cout << pri->points[0].principal_curvature_x << endl;
	cout << pri->points[0].principal_curvature_y << endl;
	cout << pri->points[0].principal_curvature_z << endl;
	//-----------------------可视化-------------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	viewer->setBackgroundColor(0.3, 0.3, 0.3);     //设置背景颜色
	viewer->addText("Curvatures", 10, 10, "text"); //设置显示文字
	viewer->setWindowName("计算主曲率");           //设置窗口名字
	viewer->addCoordinateSystem(0.1);              //添加坐标系
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0, 225, 0); //设置点云颜色
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud"); //设置点云大小
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "cloud"); //添加点云到可视化窗口
	//添加需要显示的点云法向。cloud为原始点云模型，normal为法向信息，20表示需要显示法向的点云间隔，即每20个点显示一次法向，2表示法向长度。
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals, 20, 0.002, "normals");
	// 添加需要显示的点云主曲率。cloud为原始点云模型，normal为法向信息，pri为点云主曲率，
	// 10表示需要显示曲率的点云间隔，即每10个点显示一次主曲率，10表示法向长度。
	// 目前addPointCloudPrincipalCurvatures只接受<pcl::PointXYZ>和<pcl::Normal>两个参数，未能实现曲率的可视化。
	viewer->addPointCloudPrincipalCurvatures<pcl::PointXYZ, pcl::Normal>(cloud, normals, pri, 10, 10, "Curvatures");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//利用相邻点的法向量估计一个点的主曲率--PrincipalCurvaturesEstimationCAN
#if 0
#include "PrincipalCurvaturesEstimationCAN.h"
#define R 2
#define r 1
#define N 100
#define THETA_SCALE (M_PI / 2)
#define THETA_OFFSET (M_PI / 2)
#define PHI_SCALE (M_PI / 4)
#define SEARCH_RADIUS 0.1
void calPriCurvatByNormalVectorofAdjacentPt()
{
	/*
		点云表面上特定点的所有相邻点决定了局部形状。
		如果通过曲面拟合来估计曲率，可能会产生较大误差。
		应该考虑法向量的贡献。为了估计一个点的曲率，我们将只考虑一个相邻点的贡献，这一贡献被转换为一条正截面曲线的构造。
		将构造一个法向截面圆，并根据两个点（目标点和他的一个邻居点）的位置和法向向量来估计法向曲率。
	*/
	Eigen::VectorXf theta = THETA_SCALE * Eigen::VectorXf::Random(N) + Eigen::VectorXf::Constant(N, THETA_OFFSET);
	Eigen::VectorXf phi = PHI_SCALE * Eigen::VectorXf::Random(N);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
	for (int i = 0; i < N; i++) 
	{
		pcl::PointXYZ point;
		point.x = (R + r * cos(theta(i))) * cos(phi(i));
		point.y = (R + r * cos(theta(i))) * sin(phi(i));
		point.z = r * sin(theta(i));
		cloud->points.push_back(point);
	}
	// 1 NormalEstimation
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimation;
	normal_estimation.setInputCloud(cloud);
	normal_estimation.setSearchMethod(tree);
	normal_estimation.setRadiusSearch(SEARCH_RADIUS);
	normal_estimation.setViewPoint(0, 0, std::numeric_limits<float>::infinity());
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>());
	normal_estimation.compute(*normals);
	// 2 PrincipalCurvaturesEstimationCAN
	PrincipalCurvaturesEstimationCAN curvature_estimation;
	curvature_estimation.setInputCloud(cloud);
	curvature_estimation.setInputNormals(normals);
	curvature_estimation.setSearchMethod(tree);
	curvature_estimation.setRadiusSearch(SEARCH_RADIUS);
	pcl::PointCloud<pcl::PrincipalCurvatures>::Ptr curvatures(new pcl::PointCloud<pcl::PrincipalCurvatures>());
	curvature_estimation.compute(*curvatures);
	for (int i = 0; i < N; i++) 
	{
		pcl::PrincipalCurvatures curve_point = curvatures.get()->points[i];
		std::cout
			<< theta(i) << " "
			<< phi(i) << " "
			<< curve_point.pc1 * curve_point.pc2 << " "
			<< cos(theta(i)) / (r * (R + r * cos(theta(i))))
			<< std::endl;
	}
}
#endif

//PCL 计算点云法向量并显示
#if 0
#define BOOST_TYPEOF_EMULATION
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d_omp.h>
void calculateNormalVector(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	/*
		点云的采样表面处处光滑的情况下，任何点的局部邻域都可以用平面很好的拟合
		可以认为由k个最近点拟合出的平面的法向量即当前扫描点的法向量

		compute(*normals)的计算结果：法向量的xyz坐标和表面曲率curvature

		1 计算输入点云中所有点的法线
		2 计算输入点云中一个子集的法线
		3 使用另一个数据集（比较完整）估计其最邻近点，估算输入数据集（比较稀疏）中所有点的一组曲面法线--适用于下采样后的点云
	*/

	//------------下采样----------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::VoxelGrid<pcl::PointXYZ> sor;
	sor.setInputCloud(cloud);
	sor.setLeafSize(0.005f, 0.005f, 0.05f);
	sor.filter(*cloud_downsampled);
	//-------------计算法线-------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_downsampled);
	ne.setNumberOfThreads(4);
	ne.setSearchSurface(cloud);
	ne.setSearchMethod(tree);
	ne.setRadiusSearch(0.01);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*normals);
	//------------------可视化-----------------------
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Normal viewer"));
	//设置背景颜色
	viewer->setBackgroundColor(0.3, 0.3, 0.3);
	viewer->addText("faxian", 10, 10, "text");
	//设置点云颜色
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud_downsampled, 255, 0, 0);
	//添加坐标系
	//viewer->addCoordinateSystem(0.1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_downsampled, single_color, "sample cloud");
	viewer->addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud_downsampled, normals, 10, 0.02, "normals");
	//设置点云大小
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}
#endif

//PCL 使用积分图进行法线估计
#include <pcl/features/integral_image_normal.h>//积分图发现估计
void normalEstimationByIntegralGraph(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		使用积分图计算有序点云的法线：注意，只适用于有序点云

		可使用的法线估计方法
		enum NormalEstimationMethod
		{
		COVARIANCE_MATRIX, 从最近邻的协方差矩阵创建了9个积分图去计算一个点的法线
		AVERAGE_3D_GRADIENT, 创建了6个积分图去计算3D梯度里面竖直和水平方向的光滑部分，同时利用两个梯度的卷积来计算法线。
		AVERAGE_DEPTH_CHANGE 造了一个单一的积分图，从平均深度的变化中来计算法线。
		};
	*/

	//估计法线
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::IntegralImageNormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setNormalEstimationMethod(ne.AVERAGE_3D_GRADIENT);//设置估计方法
	ne.setMaxDepthChangeFactor(0.02f);//最大深度变化系数
	ne.setNormalSmoothingSize(10.0f);//优化法线方向时考虑邻域大小
	ne.setInputCloud(cloud);//输入点云，必须是有序点云
	ne.compute(*normals);//执行法线估计

	//法线可视化
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setBackgroundColor(0, 0, 0.5);
	viewer.addPointCloudNormals<pcl::PointXYZ, pcl::Normal>(cloud, normals);
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
	return;
}

//PCL MLS平滑点云并计算法向量
#if 0
#include <pcl/surface/mls.h>
void MLSnormalEstimation(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		pcl::MovingLeastSquares移动最小二乘法
		对每个采样点的邻域拟合移动最小二乘曲面，根据局部曲面的信息来计算采样点的法失和曲率
		协方差分析法精度不如MLS，但效率更高
		对平面进行拟合，有种类似平滑点云的作用
	*/

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	// 输出文件中有PointNormal类型，用来存储移动最小二乘法算出的法线
	pcl::PointCloud<pcl::PointNormal>::Ptr mls_points_normal(new pcl::PointCloud<pcl::PointNormal>);
	// 定义对象 (第二种定义类型是为了存储法线, 即使用不到也需要定义出来)
	pcl::MovingLeastSquares<pcl::PointXYZ, pcl::PointNormal> mls;
	mls.setInputCloud(cloud);
	mls.setComputeNormals(true);     // 是否计算法线，设置为ture则计算法线
	mls.setPolynomialFit(true);      // 设置为true则在平滑过程中采用多项式拟合来提高精度
	mls.setPolynomialOrder(2);       // 设置MLS拟合的阶数，默认是2
	mls.setSearchMethod(tree);       // 邻域点搜索的方式
	mls.setSearchRadius(0.005);      // 邻域搜索半径
	mls.process(*mls_points_normal); // 曲面重建
	cout << "mls poits size is: " << mls_points_normal->size() << endl;

	// 计算结果可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("CloudShow"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->setWindowName("MLS计算点云法向量并显示");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointNormal> v(mls_points_normal, 0, 255, 0);
	viewer->addPointCloud<pcl::PointNormal>(mls_points_normal, v, "point");
	viewer->addPointCloudNormals<pcl::PointNormal>(mls_points_normal, 10, 0.1, "normal");

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}

}
#endif

//PCL 基于法向量夹角的点云特征点提取
void extractBasedOnNormalVectorAngle(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudOut)
{
	/*
		pcl::getAngle3D计算两个空间向量的夹角
		Eigen::Vector3f v1(2, 4, 7),v2(7, 8, 9);
		pcl::getAngle3D(v1,v2,false);//false: 计算的夹角结果为弧度制，true：计算的夹角结果为角度制
		double radian_angle = atan2(v1.cross(v2).norm(), v1.transpose() * v2); 
		//计算结果取值范围为[0,PI]，即用弧度制表示夹角

		cosO=ab/(|a||b|);O=arccos{ab/(|a||b|)};
	*/

	//计算每一个点的法向量
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(cloud);
	//设置邻域点搜索方式
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);
	// n.setRadiusSearch (0.03);
	n.setKSearch(10);
	//定义一个新的点云储存含有法线的值
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//计算出来法线的值
	n.compute(*normals);

	float Angle = 0.0;
	float Average_Sum_AngleK = 0.0;//定义邻域内K个点法向量夹角的平均值
	vector<int>indexes;
	float threshold = 15;//夹角阈值

	//计算法向量夹角及夹角均值
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //建立kdtree对象
	kdtree.setInputCloud(cloud); //设置需要建立kdtree的点云指针
	int K = 10;  //设置需要查找的近邻点个数
	vector<int> pointIdxNKNSearch(K);  //保存每个近邻点的索引
	vector<float> pointNKNSquaredDistance(K); //保存每个近邻点与查找点之间的欧式距离平方
	pcl::PointXYZ searchPoint;
	for (size_t i = 0; i < cloud->points.size(); ++i) 
	{
		searchPoint = cloud->points[i];
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0)
		{
			float Sum_AngleK = 0.0;//定义K个邻近的点法向夹角之和
			Eigen::Vector3f v1( normals->points[i].data_n[0],
								normals->points[i].data_n[1],
								normals->points[i].data_n[2]);
			for (size_t m = 0; m < pointIdxNKNSearch.size(); ++m) 
			{			
				Eigen::Vector3f v2(normals->points[pointIdxNKNSearch[m]].data_n[0],
						normals->points[pointIdxNKNSearch[m]].data_n[1],
						normals->points[pointIdxNKNSearch[m]].data_n[2]);
				//计算夹角
				Angle = pcl::getAngle3D(v1, v2, true);
				Sum_AngleK += Angle;//邻域夹角之和
			}		
			Average_Sum_AngleK = Sum_AngleK / pointIdxNKNSearch.size();//邻域夹角均值
		}
		//提取特征点
		if (Average_Sum_AngleK > threshold)
			indexes.push_back(i);
	}
	pcl::copyPointCloud(*cloud, indexes, *cloudOut);
}

//PCL 计算点云包围盒
#include <pcl/features/moment_of_inertia_estimation.h>
void computePtCloudBoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		pcl::MomentOfInertiaEstimation< PointT >
		1、获取基于惯性矩与偏心率的描述子
		2、提取有向包围盒OBB和坐标轴对齐包围盒AABB
		此处的有向包围盒OBB不一定是最小的包围盒
	*/

	//创建MomentOfInertiaEstimation类
	pcl::MomentOfInertiaEstimation <pcl::PointXYZ> feature_extractor;
	feature_extractor.setInputCloud(cloud);
	feature_extractor.compute();

	//声明存储描述符和边框所需的所有必要变量
	vector <float> moment_of_inertia;//存储惯性矩的特征向量
	vector <float> eccentricity;//存储偏心率的特征向量	
	pcl::PointXYZ min_point_AABB;
	pcl::PointXYZ max_point_AABB;
	pcl::PointXYZ min_point_OBB;
	pcl::PointXYZ max_point_OBB;
	pcl::PointXYZ position_OBB;
	Eigen::Matrix3f rotational_matrix_OBB;
	float major_value, middle_value, minor_value;
	Eigen::Vector3f major_vector, middle_vector, minor_vector;
	Eigen::Vector3f mass_center;
	//计算
	feature_extractor.getMomentOfInertia(moment_of_inertia);//惯性矩特征
	feature_extractor.getEccentricity(eccentricity);//偏心率特征
	feature_extractor.getAABB(min_point_AABB, max_point_AABB);//AABB对应的左下角和右上角坐标
	feature_extractor.getOBB(min_point_OBB, max_point_OBB, position_OBB, rotational_matrix_OBB);//OBB对应的相关参数
	feature_extractor.getEigenValues(major_value, middle_value, minor_value);//三个特征值
	feature_extractor.getEigenVectors(major_vector, middle_vector, minor_vector);//三个特征向量
	feature_extractor.getMassCenter(mass_center);//点云中心坐标
	
	//可视化
	cout << "蓝色的包围盒为OBB,红色的包围盒为AABB" << endl;
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("基于惯性矩与偏心率的描述子"));
	viewer->setBackgroundColor(0, 0, 0);
	viewer->addCoordinateSystem(1.0);
	viewer->initCameraParameters();
	viewer->addPointCloud<pcl::PointXYZ>(cloud, pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>(cloud, 0, 255, 0), "sample cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "sample cloud");
	viewer->addCube(min_point_AABB.x, max_point_AABB.x, min_point_AABB.y, max_point_AABB.y, min_point_AABB.z, max_point_AABB.z, 1.0, 0.0, 0.0, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "AABB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "AABB");
	Eigen::Vector3f position(position_OBB.x, position_OBB.y, position_OBB.z);

	cout << "position_OBB: " << position_OBB << endl;
	cout << "mass_center: " << mass_center << endl;
	Eigen::Quaternionf quat(rotational_matrix_OBB);
	viewer->addCube(position, quat, max_point_OBB.x - min_point_OBB.x, max_point_OBB.y - min_point_OBB.y, max_point_OBB.z - min_point_OBB.z, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 0, 1, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY, 0.5, "OBB");
	viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_LINE_WIDTH, 3, "OBB");
	viewer->setRepresentationToWireframeForAllActors();
	pcl::PointXYZ center(mass_center(0), mass_center(1), mass_center(2));
	pcl::PointXYZ x_axis(major_vector(0) + mass_center(0), major_vector(1) + mass_center(1), major_vector(2) + mass_center(2));
	pcl::PointXYZ y_axis(middle_vector(0) + mass_center(0), middle_vector(1) + mass_center(1), middle_vector(2) + mass_center(2));
	pcl::PointXYZ z_axis(minor_vector(0) + mass_center(0), minor_vector(1) + mass_center(1), minor_vector(2) + mass_center(2));
	viewer->addLine(center, x_axis, 1.0f, 0.0f, 0.0f, "major eigen vector");
	viewer->addLine(center, y_axis, 0.0f, 1.0f, 0.0f, "middle eigen vector");
	viewer->addLine(center, z_axis, 0.0f, 0.0f, 1.0f, "minor eigen vector");
	cout << "size of cloud :" << cloud->points.size() << endl;
	cout << "moment_of_inertia :" << moment_of_inertia.size() << endl;
	cout << "eccentricity :" << eccentricity.size() << endl;

	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//PCL PCA构建点云包围盒
#include <Eigen/Core>
#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
void PCABoundingBox(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		包围盒也叫外接最小矩形,是一种求解离散点集最优包围空间的算法，
		基本思想是用体积稍大且特性简单的几何体（称为包围盒）来近似地代替复杂的几何对象。
  		常见的包围盒算法有AABB包围盒、包围球、方向包围盒OBB以及固定方向凸包FDH

		最小包围盒的计算过程大致如下:
		1.利用PCA主元分析法获得点云的三个主方向，获取质心，计算协方差，获得协方差矩阵，
			求取协方差矩阵的特征值和特长向量，特征向量即为主方向。
		2.利用1中获得的主方向和质心，将输入点云转换至原点，且主方向与坐标系方向重回，建立变换到原点的点云的包围盒。
		3.给输入点云设置主方向和包围盒，通过输入点云到原点点云变换的逆变换实现。
	*/

	// 直接通过Eigen,计算点云质心和协方差矩阵
	Eigen::Vector4f pcaCentroid;
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	// 协方差矩阵分解求特征值特征向量
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	Eigen::Matrix3f eigenVectorsPCA = eigen_solver.eigenvectors();
	Eigen::Vector3f eigenValuesPCA = eigen_solver.eigenvalues();
	// 校正主方向间垂直
	eigenVectorsPCA.col(2) = eigenVectorsPCA.col(0).cross(eigenVectorsPCA.col(1));
	eigenVectorsPCA.col(0) = eigenVectorsPCA.col(1).cross(eigenVectorsPCA.col(2));
	eigenVectorsPCA.col(1) = eigenVectorsPCA.col(2).cross(eigenVectorsPCA.col(0));

	cout << "特征值va(3x1):\n" << eigenValuesPCA << endl; // Eigen计算出来的特征值默认是从小到大排列
	cout << "特征向量ve(3x3):\n" << eigenVectorsPCA << endl;
	cout << "质心点(4x1):\n" << pcaCentroid << endl;

	/*
	// 另一种计算点云协方差矩阵特征值和特征向量的方式:通过pcl中的pca接口，如下，这种情况得到的特征向量相似特征向量
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudPCAprojection (new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PCA<pcl::PointXYZ> pca;
	pca.setInputCloud(cloudSegmented);
	pca.project(*cloudSegmented, *cloudPCAprojection);
	std::cerr << std::endl << "EigenVectors: " << pca.getEigenVectors() << std::endl;//计算特征向量
	std::cerr << std::endl << "EigenValues: " << pca.getEigenValues() << std::endl;//计算特征值
	*/

	// 将输入点云转换至原点
	Eigen::Matrix4f tm = Eigen::Matrix4f::Identity();     // 定义变换矩阵 
	Eigen::Matrix4f tm_inv = Eigen::Matrix4f::Identity(); // 定义变换矩阵的逆
	tm.block<3, 3>(0, 0) = eigenVectorsPCA.transpose();   // 旋转矩阵R.
	tm.block<3, 1>(0, 3) = -1.0f * (eigenVectorsPCA.transpose()) * (pcaCentroid.head<3>());// 平移向量 -R*t
	tm_inv = tm.inverse();

	std::cout << "变换矩阵tm(4x4):\n" << tm << std::endl;
	std::cout << "逆变矩阵tm'(4x4):\n" << tm_inv << std::endl;

	pcl::PointCloud<pcl::PointXYZ>::Ptr transformedCloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*cloud, *transformedCloud, tm);

	pcl::PointXYZ min_p1, max_p1;
	Eigen::Vector3f c1, c;
	pcl::getMinMax3D(*transformedCloud, min_p1, max_p1);
	c1 = 0.5f * (min_p1.getVector3fMap() + max_p1.getVector3fMap());

	cout << "型心c1(3x1):\n" << c1 << endl;

	Eigen::Affine3f tm_inv_aff(tm_inv);
	pcl::transformPoint(c1, c, tm_inv_aff);
	Eigen::Vector3f whd, whd1;
	whd1 = max_p1.getVector3fMap() - min_p1.getVector3fMap();
	whd = whd1;
	float sc1 = (whd1(0) + whd1(1) + whd1(2)) / 3;  //点云平均尺度，用于设置主方向箭头大小

	cout << "width1=" << whd1(0) << endl;
	cout << "heght1=" << whd1(1) << endl;
	cout << "depth1=" << whd1(2) << endl;
	cout << "scale1=" << sc1 << endl;

	const Eigen::Quaternionf bboxQ1(Eigen::Quaternionf::Identity());
	const Eigen::Vector3f    bboxT1(c1);
	const Eigen::Quaternionf bboxQ(tm_inv.block<3, 3>(0, 0));
	const Eigen::Vector3f    bboxT(c);

	// 变换到原点的点云主方向
	pcl::PointXYZ op;
	op.x = 0.0;
	op.y = 0.0;
	op.z = 0.0;
	Eigen::Vector3f px, py, pz;
	Eigen::Affine3f tm_aff(tm);
	pcl::transformVector(eigenVectorsPCA.col(0), px, tm_aff);
	pcl::transformVector(eigenVectorsPCA.col(1), py, tm_aff);
	pcl::transformVector(eigenVectorsPCA.col(2), pz, tm_aff);
	pcl::PointXYZ pcaX;
	pcaX.x = sc1 * px(0);
	pcaX.y = sc1 * px(1);
	pcaX.z = sc1 * px(2);
	pcl::PointXYZ pcaY;
	pcaY.x = sc1 * py(0);
	pcaY.y = sc1 * py(1);
	pcaY.z = sc1 * py(2);
	pcl::PointXYZ pcaZ;
	pcaZ.x = sc1 * pz(0);
	pcaZ.y = sc1 * pz(1);
	pcaZ.z = sc1 * pz(2);

	// 初始点云的主方向
	pcl::PointXYZ cp;
	cp.x = pcaCentroid(0);
	cp.y = pcaCentroid(1);
	cp.z = pcaCentroid(2);
	pcl::PointXYZ pcX;
	pcX.x = sc1 * eigenVectorsPCA(0, 0) + cp.x;
	pcX.y = sc1 * eigenVectorsPCA(1, 0) + cp.y;
	pcX.z = sc1 * eigenVectorsPCA(2, 0) + cp.z;
	pcl::PointXYZ pcY;
	pcY.x = sc1 * eigenVectorsPCA(0, 1) + cp.x;
	pcY.y = sc1 * eigenVectorsPCA(1, 1) + cp.y;
	pcY.z = sc1 * eigenVectorsPCA(2, 1) + cp.z;
	pcl::PointXYZ pcZ;
	pcZ.x = sc1 * eigenVectorsPCA(0, 2) + cp.x;
	pcZ.y = sc1 * eigenVectorsPCA(1, 2) + cp.y;
	pcZ.z = sc1 * eigenVectorsPCA(2, 2) + cp.z;

	// 可视化
	pcl::visualization::PCLVisualizer viewer;
	viewer.setBackgroundColor(1.0, 1.0, 1.0);
	viewer.setWindowName("PCA获取点云包围盒");
	//输入的初始点云
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> color_handler(cloud, 255, 0, 0);
	viewer.addPointCloud(cloud, color_handler, "cloud");
	viewer.addCube(bboxT, bboxQ, whd(0), whd(1), whd(2), "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "bbox");

	viewer.addArrow(pcX, cp, 1.0, 0.0, 0.0, false, "arrow_x");
	viewer.addArrow(pcY, cp, 0.0, 1.0, 0.0, false, "arrow_y");
	viewer.addArrow(pcZ, cp, 0.0, 0.0, 1.0, false, "arrow_z");

	//转换到原点的点云
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tc_handler(transformedCloud, 0, 255, 0);
	viewer.addPointCloud(transformedCloud, tc_handler, "transformCloud");
	viewer.addCube(bboxT1, bboxQ1, whd1(0), whd1(1), whd1(2), "bbox1");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox1");
	viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0.0, 1.0, 0.0, "bbox1");

	viewer.addArrow(pcaX, op, 1.0, 0.0, 0.0, false, "arrow_X");
	viewer.addArrow(pcaY, op, 0.0, 1.0, 0.0, false, "arrow_Y");
	viewer.addArrow(pcaZ, op, 0.0, 0.0, 1.0, false, "arrow_Z");
	viewer.addCoordinateSystem(0.5f * sc1);

	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(10000));
	}

}

//PCL 点云边界提取
#include <pcl/features/boundary.h> //边界提取
void boundaryExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		基于发现法线实现点云边缘提取
		边缘点往往位于最外围：其周围的点大多在边缘点的同一侧
	*/

	//计算法向量
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setRadiusSearch(0.01);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);
	//边界特征估计
	pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundEst;
	boundEst.setInputCloud(cloud);
	boundEst.setInputNormals(normals);
	boundEst.setRadiusSearch(0.02);
	boundEst.setAngleThreshold(M_PI / 2);//边界判断时的角度阈值
	boundEst.setSearchMethod(tree);
	pcl::PointCloud<pcl::Boundary> boundaries;
	boundEst.compute(boundaries);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_boundary(new pcl::PointCloud<pcl::PointXYZ>);
	for (int i = 0; i < cloud->points.size(); i++)
		if (boundaries[i].boundary_point > 0)
			cloud_boundary->push_back(cloud->points[i]);
	cout << "边界点个数:" << cloud_boundary->points.size() << endl;

	//可视化
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("ShowCloud"));
	int v1(0);
	viewer->setWindowName("边界提取");
	viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->addText("Raw point clouds", 10, 10, "v1_text", v1);
	int v2(0);
	viewer->createViewPort(0.5, 0.0, 1, 1.0, v2);
	viewer->setBackgroundColor(0.5, 0.5, 0.5, v2);
	viewer->addText("Boudary point clouds", 10, 10, "v2_text", v2);

	viewer->addPointCloud<pcl::PointXYZ>(cloud, "sample cloud", v1);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_boundary, "cloud_boundary", v2);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "sample cloud", v1);
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloud_boundary", v2);
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100000));
	}
}

//PCL alpha shapes平面点云边界特征提取
#if 0
#include <pcl/surface/concave_hull.h> // 提取平面点云边界
#include <pcl/console/time.h> // TicToc
void planeBoundaryExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_hull)
{
	pcl::console::TicToc time;
	time.tic();

	pcl::ConcaveHull<pcl::PointXYZ> chull;
	chull.setInputCloud(cloud); // 输入点云为投影后的点云
	chull.setAlpha(0.1);        // 设置alpha值为0.1
	chull.reconstruct(*cloud_hull);
	cout << "提取边界点个数为: " << cloud_hull->points.size() << endl;

	cout << "提取边界点用时： " << time.toc() / 1000 << " 秒" << endl;
}
#endif

//---------------------------------关键点提取-------------------------------------//

//PCL ISS关键点提取
#include <pcl/keypoints/iss_3d.h>
#include <boost/thread/thread.hpp>
void ISSExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		ISS内部形状描述子：表示立体几何形状的方法，含有丰富的几何特征信息，可完成高质量点云配准
	*/

	pcl::ISSKeypoint3D<pcl::PointXYZ, pcl::PointXYZ> iss;
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

	//传递索引
	//vector <int>(point_indices);
	//pcl::IndicesPtr indices(new vector <int>(point_indices));
	iss.setInputCloud(cloud);
	iss.setSearchMethod(tree);
	iss.setNumberOfThreads(4);     //初始化调度器并设置要使用的线程数
	iss.setSalientRadius(1.0f);  // 设置用于计算协方差矩阵的球邻域半径
	iss.setNonMaxRadius(1.5f);   // 设置非极大值抑制应用算法的半径
	iss.setThreshold21(0.65);     // 设定第二个和第一个特征值之比的上限
	iss.setThreshold32(0.5);     // 设定第三个和第二个特征值之比的上限
	iss.setMinNeighbors(10);       // 在应用非极大值抑制算法时，设置必须找到的最小邻居数
	iss.compute(*keypoints);

	/*for (size_t ii = 0; ii < keypoints->points.size();++ii) {
		point_indices.push_back(iss.getKeypointsIndices()->indices[ii]);
	};*/
	cout << "ISS_3D points 的提取结果为 " << keypoints->points.size() << endl;

	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("3D ISS"));
	viewer->setBackgroundColor(255, 255, 255);
	viewer->setWindowName("ISS关键点提取");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> single_color(cloud, 0.0, 255, 0.0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud, single_color, "sample cloud");
	viewer->addPointCloud<pcl::PointXYZ>(keypoints, "key cloud");//特征点
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "key cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1.0, 0.0, 0.0, "key cloud");
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}
}

//PCL Harris3D关键点提取
#if 0
#include <pcl/keypoints/harris_3d.h> // Harris3D关键点检测
void Harris3DExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		注意此处PCL的point类型设置为<pcl::PointXYZI>,即除了x、y、z坐标还必须包含强度信息
		因为Harris的评估值保存在输出点云的(I)分量中，Harris输出点云中的(I)非传统点云中的
		强度信息，因此后续在保存和可视化输出点云的时候需要通过点的索引来重新获取。
	*/

	pcl::StopWatch watch; // 计时器
	pcl::HarrisKeypoint3D <pcl::PointXYZ, pcl::PointXYZI> harris;
	pcl::PointCloud<pcl::PointXYZI>::Ptr Harris_keypoints(new pcl::PointCloud<pcl::PointXYZI>);

	harris.setInputCloud(cloud);     // 提供指向输入数据集的指针
	harris.setMethod(harris.LOWE);   // 设置要计算响应的方法（可以不设置）
	harris.setRadius(0.02);          // 设置法线估计和非极大值抑制的半径。
	harris.setRadiusSearch(0.01);    // 设置用于关键点检测的最近邻居的球半径
	harris.setNonMaxSupression(true);// 是否应该应用非最大值抑制
	harris.setThreshold(0.002);     // 设置角点检测阈值，只有当非极大值抑制设置为true时才有效
	harris.setRefine(true);          // 检测到的关键点是否需要细化，设置为true时，关键点为点云中的点
	harris.setNumberOfThreads(6);    // 初始化调度程序并设置要使用的线程数
	harris.compute(*Harris_keypoints);

	cout << "算法运行:" << watch.getTimeSeconds() << "秒" << endl;
	cout << "提取关键点" << Harris_keypoints->points.size() << "个" << endl;
	pcl::PointCloud<pcl::PointXYZ>::Ptr keypoints(new pcl::PointCloud<pcl::PointXYZ>);
	//-----------------------获取关键点的索引-------------------------
	pcl::PointIndicesConstPtr keypoints2_indices = harris.getKeypointsIndices();
	//pcl::PointXYZI格式无法正确保存和显示，这里通过索引从原始点云中获取位于点云中特征点
	pcl::copyPointCloud(*cloud, *keypoints2_indices, *keypoints);
}
#endif

//PCL 3D - SIFT关键点检测(Z方向梯度约束)
#if 0
#include <pcl/keypoints/sift_keypoint.h>
namespace pcl
{
	template<>
	struct SIFTKeypointFieldSelector<PointXYZ>
	{
		inline float
			operator () (const PointXYZ &p) const
		{
			return p.z;
		}
	};
}
void SIFTZExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz)
{
	pcl::StopWatch watch; // 计时器
	//-----------------------------SIFT算法参数----------------------------------
	const float min_scale = 0.001f;           // 设置尺度空间中最小尺度的标准偏差          
	const int n_octaves = 3;                  // 设置尺度空间层数，越小则特征点越多           
	const int n_scales_per_octave = 15;       // 设置尺度空间中计算的尺度个数
	const float min_contrast = 0.0001f;       // 设置限制关键点检测的阈值  

	//----------------------------SIFT关键点检测---------------------------------
	pcl::SIFTKeypoint<pcl::PointXYZ, pcl::PointWithScale> sift;//创建sift关键点检测对象
	pcl::PointCloud<pcl::PointWithScale> result;
	sift.setInputCloud(cloud_xyz);            // 设置输入点云
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	sift.setSearchMethod(tree);               // 创建一个空的kd树对象tree，并把它传递给sift检测对象
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);//指定搜索关键点的尺度范围
	sift.setMinimumContrast(min_contrast);    // 设置限制关键点检测的阈值
	sift.compute(result);                     // 执行sift关键点检测，保存结果在result
	cout << "Extracted " << result.size() << " keypoints" << endl;
	cout << "SIFT关键点提取用时： " << watch.getTimeSeconds() << "秒" << endl;

	//为了可视化需要将点类型pcl::PointWithScale的数据转换为点类型pcl::PointXYZ的数据
	//方法一
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	copyPointCloud(result, *cloud_temp);

	//方法二
	/*pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>& cloud_sift = *cloud_temp;
	pcl::PointXYZ point;
	for (int i = 0; i < result.size(); i++)
	{
	point.x = result.at(i).x;
	point.y = result.at(i).y;
	point.z = result.at(i).z;
	cloud_sift.push_back(point);
	}*/

	//---------------------可视化输入点云和关键点----------------------------
	pcl::visualization::PCLVisualizer viewer("Sift keypoint");
	viewer.setWindowName("SIFT关键点检测");
	viewer.setBackgroundColor(255, 255, 255);
	viewer.addPointCloud(cloud_xyz, "cloud");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "cloud");
	viewer.addPointCloud(cloud_temp, "keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2, "keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 1, 0, 0, "keypoints");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce();
	}
}
#endif

//PCL 3D - SIFT关键点检测(曲率不变特征约束,基于法向梯度，用曲率代替强度梯度)
#if 0
#include <pcl/features/normal_3d.h>
#include <pcl/keypoints/sift_keypoint.h>
void SIFTNormExtraction(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_xyz)
{
	// --------------------SIFT算法参数----------------------------
	const float min_scale = 0.001f;     // 设置尺度空间中最小尺度的标准偏差 
	const int n_octaves = 3;            // 设置尺度空间层数，越小则特征点越多 
	const int n_scales_per_octave = 4;  // 设置尺度空间中计算的尺度个数
	const float min_contrast = 0.0001f; // 设置限制关键点检测的阈值       

	// ----------计算cloud_xyz的法向量和表面曲率-------------------
	pcl::NormalEstimation<pcl::PointXYZ, pcl::PointNormal> ne;
	pcl::PointCloud<pcl::PointNormal>::Ptr cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree_n(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setInputCloud(cloud_xyz);
	ne.setSearchMethod(tree_n);
	ne.setRadiusSearch(0.01);
	ne.compute(*cloud_normals);
	// 从cloud_xyz复制xyz信息，并将其添加到cloud_normals中，因为PointNormals估计中的xyz字段为零  
	for (std::size_t i = 0; i < cloud_normals->size(); ++i)
	{
		(*cloud_normals)[i].x = (*cloud_xyz)[i].x;
		(*cloud_normals)[i].y = (*cloud_xyz)[i].y;
		(*cloud_normals)[i].z = (*cloud_xyz)[i].z;
	}
	// -----------------使用法线值作为强度变量估计SIFT关键点-----------------------------
	pcl::SIFTKeypoint<pcl::PointNormal, pcl::PointWithScale> sift;
	pcl::PointCloud<pcl::PointWithScale> result;
	pcl::search::KdTree<pcl::PointNormal>::Ptr tree(new pcl::search::KdTree<pcl::PointNormal>());
	sift.setSearchMethod(tree);
	sift.setScales(min_scale, n_octaves, n_scales_per_octave);
	sift.setMinimumContrast(min_contrast);
	sift.setInputCloud(cloud_normals);
	sift.compute(result);
	cout << "No of SIFT points in the result are " << result.size() << endl;
	//将点云转为pointXYZ进行可视化
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_temp(new pcl::PointCloud<pcl::PointXYZ>);
	copyPointCloud(result, *cloud_temp);
	cout << "SIFT points in the cloud_temp are " << cloud_temp->size() << endl;
	//可视化输入点云和关键点
	pcl::visualization::PCLVisualizer viewer("PCL Viewer");
	viewer.setWindowName("SIFT关键点检测");
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> keypoints_color_handler(cloud_temp, 0, 255, 0);
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> cloud_color_handler(cloud_xyz, 255, 0, 0);
	viewer.setBackgroundColor(0.0, 0.0, 0.0);
	viewer.addPointCloud(cloud_xyz, cloud_color_handler, "cloud");
	viewer.addPointCloud(cloud_temp, keypoints_color_handler, "keypoints");
	viewer.setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 3, "keypoints");
	while (!viewer.wasStopped())
	{
		viewer.spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(100));
	}
}
#endif

//---------------------------------特征描述子-------------------------------------//

//PCL 计算PFH并可视化
#include <pcl/features/pfh.h>
#include <pcl/visualization/pcl_plotter.h>// 直方图的可视化 方法2
void PFHDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_ptr)
{
	// 计算法线========创建法线估计类
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	ne.setInputCloud(cloud_ptr);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	ne.setSearchMethod(tree);//设置近邻搜索算法

	// 输出点云 带有法线描述
	pcl::PointCloud<pcl::Normal>::Ptr cloud_normals_ptr(new pcl::PointCloud<pcl::Normal>);
	pcl::PointCloud<pcl::Normal>& cloud_normals = *cloud_normals_ptr;
	ne.setRadiusSearch(0.03);//半价内搜索临近点 3cm
	ne.compute(cloud_normals);

	//创建PFH估计对象pfh，并将输入点云数据集cloud和法线normals传递给它
	pcl::PFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::PFHSignature125> pfh;// phf特征估计其器
	pfh.setInputCloud(cloud_ptr);
	pfh.setInputNormals(cloud_normals_ptr);
	//如果点云是类型为PointNormal,则执行pfh.setInputNormals (cloud);
	//创建一个空的kd树表示法，并把它传递给PFH估计对象。
	//基于已给的输入数据集，建立kdtree
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
	//pcl::KdTreeFLANN<pcl::PointXYZ>::Ptr tree2 (new pcl::KdTreeFLANN<pcl::PointXYZ> ()); //-- older call for PCL 1.5-
	pfh.setSearchMethod(tree2);//设置近邻搜索算法
	pcl::PointCloud<pcl::PFHSignature125>::Ptr pfh_fe_ptr(new pcl::PointCloud<pcl::PFHSignature125>());//phf特征

	//使用半径在5厘米范围内的所有邻元素。
	//注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
	pfh.setRadiusSearch(0.05);
	//计算pfh特征值
	pfh.compute(*pfh_fe_ptr);
	cout << "phf feature size : " << pfh_fe_ptr->points.size() << endl;
	// 应该与input cloud->points.size ()有相同的大小，即每个点都有一个pfh特征向量

	// ========直方图可视化=============================
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*pfh_fe_ptr, 300); //设置的很坐标长度，该值越大，则显示的越细致
	plotter.plot();

}

//PCL 计算FPFH并可视化
#include <pcl/features/fpfh.h>
#include <pcl/features/normal_3d_omp.h>//使用OMP需要添加的头文件
#include <pcl/visualization/histogram_visualizer.h> //直方图的可视化
#include <pcl/visualization/pcl_plotter.h>// 直方图的可视化 方法2
void FPFHDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	//计算法向量
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	n.setInputCloud(cloud);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setSearchMethod(tree);//设置近邻搜索算法
	n.setNumberOfThreads(4);//设置openMP的线程数
	n.setKSearch(30);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);
	//计算FPFH
	pcl::FPFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fpfh;
	fpfh.setInputCloud(cloud);
	//fpfh.setSearchSurface(cloud);
	fpfh.setInputNormals(normals);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
	fpfh.setSearchMethod(tree2);//设置近邻搜索算法

	pcl::PointCloud<pcl::FPFHSignature33>::Ptr fpfh_fe(new pcl::PointCloud<pcl::FPFHSignature33>());
	//注意：此处使用的半径必须要大于估计表面法线时使用的半径!!!
	fpfh.setRadiusSearch(0.5);
	//计算pfh特征值
	fpfh.compute(*fpfh_fe);
	cout << "phf feature size : " << fpfh_fe->points.size() << endl;
	// 应该与input cloud->points.size ()有相同的大小，即每个点都有一个pfh特征向量

	// ---------------直方图可视化-----------------

	//定义绘图器
	pcl::visualization::PCLPlotter* plotter = new pcl::visualization::PCLPlotter("My Plotter");
	//设置特性
	plotter->setTitle("FPFH");
	plotter->setShowLegend(true);
	cout << pcl::getFieldsList<pcl::FPFHSignature33>(*fpfh_fe);
	plotter->addFeatureHistogram<pcl::FPFHSignature33>(*fpfh_fe, "fpfh", 5, "one_fpfh");
	/*第2个参数为点云类型的field name，5表示可视化第五个点的FPFH特征
	该参数可通过getFieldsList()返回，并且只限定于注册过的点云类型*/
	plotter->setWindowSize(800, 600);
	plotter->spinOnce(30000000);
	plotter->clearPlots();
	//方法2
	// pcl::visualization::PCLPlotter plotter;
	//plotter.addFeatureHistogram(*fpfh_fe_ptr, 300); //设置的根坐标长度，该值越大，则显示的越细致
	//plotter.plot();
}

//PCL 估计一点云的VFH特征
#include <pcl/features/vfh.h>//vfh
void VFHDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	//计算法线
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	// 添加搜索算法 kdtree search  最近的几个点 估计平面 协方差矩阵PCA分解 求解法线
	ne.setInputCloud(cloud);
	ne.setSearchMethod(tree);//设置近邻搜索算法 
	ne.setRadiusSearch(0.03);//半价内搜索临近点 3cm
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	ne.compute(*normals);//计算法线
	
	//创建VFH估计对象vfh，并把输入数据集cloud和法线normals传递给它------
	pcl::VFHEstimation<pcl::PointXYZ, pcl::Normal, pcl::VFHSignature308> vfh;
	vfh.setInputCloud(cloud);
	vfh.setInputNormals(normals);
	//创建一个空的kd树对象，并把它传递给VFH估计对象。
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>());
	vfh.setSearchMethod(tree2);//设置近邻搜索算法 
	pcl::PointCloud<pcl::VFHSignature308>::Ptr vfh_fe_ptr(new pcl::PointCloud<pcl::VFHSignature308>());
	vfh.compute(*vfh_fe_ptr);
	cout << "vfh feature size : " << vfh_fe_ptr->points.size() << endl; // 应该 等于 1

	//可视化直方图
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*vfh_fe_ptr, 300); //设置的很坐标长度，该值越大，则显示的越细致
	plotter.plot();
}

//PCL Spin Image旋转图像
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/spin_image.h>
void spinImageDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		通过围绕某个轴（法线）旋转来累积点的二维直方图
		对遮挡和背景干扰具有很强的稳健性，在点云配准和目标识别中广泛应用
	*/

	//------------------下采样滤波-----------------
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setLeafSize(0.012, 0.012, 0.012);
	voxel_grid.setInputCloud(cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud1(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid.filter(*cloud1);
	cout << "down size *cloud_src_o from " << cloud->size() << "to" << cloud1->size() << endl;

	//--------------------计算法线------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//建立kdtree来进行近邻点集搜索
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setNumberOfThreads(6);//设置openMP的线程数
	n.setInputCloud(cloud1);
	n.setSearchMethod(tree);
	//n.setKSearch(10);//点云法向计算时，需要所搜的近邻点大小
	n.setRadiusSearch(0.3);//半径搜素
	n.compute(*normals);//开始进行法向计

	//自旋图像计算
	pcl::SpinImageEstimation<pcl::PointXYZ, pcl::Normal, pcl::Histogram<153> > spin_image_descriptor(8, 0.05, 10);
	//三个数字分别表示：旋转图像分辨率；最小允许输入点与搜索曲面点之间的法线夹角的余弦值，以便在支撑中保留该点；
	//小支持点数，以正确估计自旋图像。如果在某个点支持包含的点较少，则抛出异常
	spin_image_descriptor.setInputCloud(cloud1);
	spin_image_descriptor.setInputNormals(normals);
	// 使用法线计算的KD树
	spin_image_descriptor.setSearchMethod(tree);
	pcl::PointCloud<pcl::Histogram<153> >::Ptr spin_images(new pcl::PointCloud<pcl::Histogram<153> >);
	spin_image_descriptor.setRadiusSearch(0.2);
	// 计算自旋图像
	spin_image_descriptor.compute(*spin_images);
	cout << "SI output points.size (): " << spin_images->points.size() << endl;

	// 显示和检索第一点的自旋图像描述符向量。
	pcl::Histogram<153> first_descriptor = spin_images->points[0];
	cout << first_descriptor << endl;
	//自旋图像描述符可视化
	pcl::visualization::PCLPlotter plotter;
	plotter.addFeatureHistogram(*spin_images, 300); //设置的横坐标长度，该值越大，则显示的越细致
	plotter.plot();
}

//PCL SHOT352描述子
#include <pcl/features/shot_omp.h>    //描述子
#include <pcl/keypoints/uniform_sampling.h>//均匀采样
void SHOTDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	//-------------------计算法线-----------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setNumberOfThreads(4);//设置openMP的线程数
	n.setViewPoint(0,0,0);//设置视点，默认为（0，0，0）
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);//点云法向计算时，需要所搜的近邻点大小
	n.compute(*normals);//开始进行法向计

	// -------均匀采样提取关键点-------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::UniformSampling<pcl::PointXYZ> US;
	US.setInputCloud(cloud);
	US.setRadiusSearch(0.01f);
	US.filter(*cloud_filtered);
	cout << "均匀采样之后点云的个数：" << cloud_filtered->points.size() << endl;

	//------------为关键点计算描述子----------------
	pcl::PointCloud<pcl::SHOT352>::Ptr model_descriptors(new pcl::PointCloud<pcl::SHOT352>()); //描述子
	pcl::SHOTEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::SHOT352> descr_est;
	descr_est.setRadiusSearch(2);  //设置搜索半径
	descr_est.setInputCloud(cloud_filtered);  //输入模型的关键点
	descr_est.setInputNormals(normals);  //输入模型的法线
	descr_est.setSearchSurface(cloud);         //输入的点云
	descr_est.setNumberOfThreads(4);	//设置openMP的线程数，所有OMP处都需要加上
	descr_est.compute(*model_descriptors);     //计算描述子
}

//PCL 计算3DSC并可视化
#include <pcl/features/3dsc.h> //3D形状描述子
void _3DSCDescribe(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud)
{
	/*
		ShapeContext3DEstimation：3D形状上下文描述符
		扩展图像中的形状上下文特征
	*/

	//------------------计算法线----------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP加速
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setNumberOfThreads(4);//设置openMP的线程数
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);

	//-------------------计算3dsc-----------------------
	pcl::ShapeContext3DEstimation<pcl::PointXYZ, pcl::Normal, pcl::ShapeContext1980> sc;
	sc.setInputCloud(cloud);
	sc.setInputNormals(normals);
	sc.setSearchMethod(tree);
	pcl::PointCloud<pcl::ShapeContext1980>::Ptr sps_src(new pcl::PointCloud<pcl::ShapeContext1980>());
	sc.setMinimalRadius(0.02);     //搜索球面(Rmin)的最小半径值。
	sc.setRadiusSearch(0.03);      //设置用于确定用于特征估计的最近邻居的球体半径。
	sc.setPointDensityRadius(0.02);//这个半径用于计算局部点密度=这个半径内的点数。
	sc.compute(*sps_src);

	// ---------------直方图可视化-----------------

	pcl::visualization::PCLPlotter* plotter = new pcl::visualization::PCLPlotter("My Plotter");
	plotter->setTitle("3DSC");
	plotter->setShowLegend(true);
	cout << pcl::getFieldsList<pcl::ShapeContext1980>(*sps_src);
	plotter->addFeatureHistogram<pcl::ShapeContext1980>(*sps_src, "shape_context", 5, "one_3dsc");
	/*第2个参数为点云类型的field name，5表示可视化第五个点的特征
	该参数可通过getFieldsList()返回，并且只限定于注册过的点云类型*/
	plotter->setWindowSize(800, 600);
	plotter->spinOnce(30000000);
	plotter->clearPlots();
}

//------------------------------------------------------------------------------------------------//

//显示点云--双窗口
void visualizeFeatCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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
	if (pcl::io::loadPCDFile<pcl::PointXYZ>("../roorm.pcd", *cloud) == -1)
	{
		PCL_ERROR("Cloudn't read file!");
		return;
	}
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudOut(new pcl::PointCloud<pcl::PointXYZ>);

	//------------------------------------------------------------------------------------------------//

	//pclFeatureUSE(cloud);
	//calPCAFeature(cloud, cloudOut);
	//curvatureByQuadricSurfaceFitting(cloud);
	//calPrincipalCurvature(cloud);
	//calPriCurvatByNormalVectorofAdjacentPt();
	//calculateNormalVector(cloud);
	//normalEstimationByIntegralGraph(cloud);
	//MLSnormalEstimation(cloud);
	//extractBasedOnNormalVectorAngle(cloud, cloudOut);
	//computePtCloudBoundingBox(cloud);
	//PCABoundingBox(cloud);
	//boundaryExtraction(cloud);
	//planeBoundaryExtraction(cloud, cloudOut);

	//------------------------------------------------------------------------------------------------//

	//ISSExtraction(cloud);
	//Harris3DExtraction(cloud);
	//SIFTZExtraction(cloud);
	//SIFTNormExtraction(cloud);

	//------------------------------------------------------------------------------------------------//

	//PFHDescribe(cloud);
	//FPFHDescribe(cloud);
	//VFHDescribe(cloud);
	//spinImageDescribe(cloud);
	//SHOTDescribe(cloud);
	_3DSCDescribe(cloud);

	//------------------------------------------------------------------------------------------------//

	//visualizeFeatCloud(cloud, cloudOut);
	system("pause");
	return;
}