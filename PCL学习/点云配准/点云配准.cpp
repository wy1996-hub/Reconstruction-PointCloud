#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/console/time.h>   // ����̨����ʱ��
#include <boost/thread/thread.hpp>
#include <pcl/visualization/pcl_visualizer.h>

using namespace std;

//------------------------------------------------------------------------------------------------//
void visualizeCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& output_cloud)
{
	// ��ʼ�����ƿ��ӻ�����
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("��ʾ����"));
	viewer->setBackgroundColor(0, 0, 0);
	// ��Ŀ�������ɫ���ӻ� (red).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> target_color(target_cloud, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(target_cloud, target_color, "target cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "target cloud");
	// ��Դ������ɫ���ӻ� (blue).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> input_color(source_cloud, 0, 0, 255);
	viewer->addPointCloud<pcl::PointXYZ>(source_cloud, input_color, "input cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "input cloud");
	// ��ת�����Դ������ɫ (green)���ӻ�.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> output_color(output_cloud, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(output_cloud, output_color, "output cloud");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "output cloud");
	// �ȴ�ֱ�����ӻ����ڹر�
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}
}
//-----------------------------------------����׼------------------------------------------//

//PCL 4PCS�㷨ʵ�ֵ��ƴ���׼
#include <pcl/registration/ia_fpcs.h> // 4PCS�㷨
void _4PCS(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
{
	pcl::console::TicToc time;
	time.tic();
	//--------------��ʼ��4PCS��׼����-------------------
	pcl::registration::FPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> fpcs;
	fpcs.setInputSource(source_cloud);  // Դ����
	fpcs.setInputTarget(target_cloud);  // Ŀ�����
	fpcs.setApproxOverlap(0.7);         // ����Դ��Ŀ��֮��Ľ����ص��ȡ�
	fpcs.setDelta(0.01);                // ������׼���Ӧ��֮��ľ��루����Ϊ��λ����
	fpcs.setNumberOfSamples(100);       // ������֤��׼Ч��ʱҪʹ�õĲ���������
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcs(new pcl::PointCloud<pcl::PointXYZ>);
	fpcs.align(*pcs);                   // ����任����
	cout << "FPCS��׼��ʱ�� " << time.toc() << " ms" << endl;
	cout << "�任����" << fpcs.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ������ƽ��б任
	pcl::transformPointCloud(*source_cloud, *pcs, fpcs.getFinalTransformation());
	visualizeCloud(target_cloud, source_cloud, pcs);
}

//PCL K4PCS�㷨ʵ�ֵ��ƴ���׼
#include <pcl/registration/ia_kfpcs.h> //K4PCS�㷨ͷ�ļ�
void K4PCS(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		1 �����˲������е����²�����Ȼ�����harris\DoG�ؼ�����
		2 ͨ��4PCS�㷨��ʹ�ùؼ��㼯�ϣ�����ԭʼ���ƽ������ݵ�ƥ�䣬�������Ч��
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------------K4PCS�㷨������׼------------------------------
	pcl::registration::KFPCSInitialAlignment<pcl::PointXYZ, pcl::PointXYZ> kfpcs;
	kfpcs.setInputSource(source);  // Դ����
	kfpcs.setInputTarget(target);  // Ŀ�����
	kfpcs.setApproxOverlap(0.7);   // Դ��Ŀ��֮��Ľ����ص���
	kfpcs.setLambda(0.5);          // ƽ�ƾ���ļ�Ȩϵ����(��ʱ��֪���Ǹ�ʲô�õ�)
	kfpcs.setDelta(0.002, false);  // ��׼��Դ���ƺ�Ŀ�����֮��ľ���
	kfpcs.setNumberOfThreads(6);   // OpenMP���̼߳��ٵ��߳���
	kfpcs.setNumberOfSamples(200); // ��׼ʱҪʹ�õ��������������
	pcl::PointCloud<pcl::PointXYZ>::Ptr kpcs(new pcl::PointCloud<pcl::PointXYZ>);
	kfpcs.align(*kpcs);

	cout << "KFPCS��׼��ʱ�� " << time.toc() << " ms" << endl;
	cout << "�任����\n" << kfpcs.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ�����Դ���ƽ��б任
	pcl::transformPointCloud(*source, *kpcs, kfpcs.getFinalTransformation());
	visualizeCloud(target, source, kpcs);
}

//PCL �Ľ���RANSAC�㷨ʵ�ֵ��ƴ���׼
#include <pcl/registration/sample_consensus_prerejective.h>//���������һ������׼
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
	//-------------------------����������-----------------------
	pointnormal::Ptr normals(new pointnormal);
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(input_cloud);
	n.setNumberOfThreads(8);        // ����openMP���߳���
	n.setSearchMethod(tree);        // ������ʽ
	n.setKSearch(10);               // K���ڵ����
	n.compute(*normals);           
	//-------------------------FPFH����-------------------------
	fpfhFeature::Ptr fpfh(new fpfhFeature);
	pcl::FPFHEstimationOMP<pcl::PointXYZ, pcl::Normal, pcl::FPFHSignature33> fest;
	fest.setNumberOfThreads(8);     //ָ��8�˼���
	fest.setInputCloud(input_cloud);//�������
	fest.setInputNormals(normals);  //���뷨��
	fest.setSearchMethod(tree);     //������ʽ
	fest.setKSearch(10);            //K���ڵ����
	//fest.setRadiusSearch(0.025);  //�����뾶
	fest.compute(*fpfh);            //����FPFH
	return fpfh;
}
void RANSACEx(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		RANSAC:random sampling consensus �������һ���ԣ������ڵ�����׼
		������׼��Ŀ�ľ��ǹ��ƣ�����任����RT
	*/
	//---------------����Դ���ƺ�Ŀ����Ƶ�FPFH----------------------
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);
	//--------------------RANSAC������׼-----------------------------
	pcl::SampleConsensusPrerejective<PointT, PointT, pcl::FPFHSignature33> r_sac;
	r_sac.setInputSource(source);            // Դ����
	r_sac.setInputTarget(target);            // Ŀ�����
	r_sac.setSourceFeatures(source_fpfh);    // Դ����FPFH����
	r_sac.setTargetFeatures(target_fpfh);    // Ŀ�����FPFH����
	r_sac.setCorrespondenceRandomness(5);    // ��ѡ�����������Ӧʱ������Ҫʹ�õ��ھӵ�����,��ֵԽ������ƥ��������Խ��
	r_sac.setInlierFraction(0.5f);           // �����(�����)inlier����
	r_sac.setNumberOfSamples(3);             // ÿ�ε�����ʹ�õĲ���������
	r_sac.setSimilarityThreshold(0.1f);      // ���ײ����ζ�Ӧ�ܾ�������ı�Ե����֮���������ֵ����Ϊ[0,1]������1Ϊ��ȫƥ�䡣
	r_sac.setMaxCorrespondenceDistance(1.0f);// �ڵ㣬��ֵ Inlier threshold
	r_sac.setMaximumIterations(100);         // RANSAC ������������
	pointcloud::Ptr align(new pointcloud);
	r_sac.align(*align);
	pcl::transformPointCloud(*source, *align, r_sac.getFinalTransformation());
	cout << "�任����\n" << r_sac.getFinalTransformation() << endl;

	//-------------------���ӻ�------------------------------------
	visualizeCloud(source, target, align);
}

//PCL SAC-IA ��ʼ��׼�㷨-->��任����
#include <pcl/registration/ia_ransac.h>//sac_ia�㷨
#include <pcl/filters/voxel_grid.h>//�����²����˲�
void SAC_IA(pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud)
{
	clock_t start, end, time;
	start = clock();
	//---------------------------ȥ��Դ���Ƶ�NAN��------------------------
	vector<int> indices_src; //����ȥ���ĵ������
	pcl::removeNaNFromPointCloud(*source_cloud, *source_cloud, indices_src);
	//-------------------------Դ�����²����˲�-------------------------
	pcl::VoxelGrid<pcl::PointXYZ> vs;
	vs.setLeafSize(0.005, 0.005, 0.005);
	vs.setInputCloud(source_cloud);
	pointcloud::Ptr source(new pointcloud);
	vs.filter(*source);
	cout << "down size *source_cloud from " << source_cloud->size() << " to " << source->size() << endl;
	//--------------------------ȥ��Ŀ����Ƶ�NAN��--------------------
	vector<int> indices_tgt; //����ȥ���ĵ������
	pcl::removeNaNFromPointCloud(*target_cloud, *target_cloud, indices_tgt);
	//----------------------Ŀ������²����˲�-------------------------
	pcl::VoxelGrid<pcl::PointXYZ> vt;
	vt.setLeafSize(0.005, 0.005, 0.005);
	vt.setInputCloud(target_cloud);
	pointcloud::Ptr target(new pointcloud);
	vt.filter(*target);
	cout << "down size *target_cloud from " << target_cloud->size() << " to " << target->size() << endl;
	//---------------����Դ���ƺ�Ŀ����Ƶ�FPFH------------------------
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);
	//--------------����һ����SAC_IA��ʼ��׼----------------------------
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(source);
	sac_ia.setSourceFeatures(source_fpfh);
	sac_ia.setInputTarget(target);
	sac_ia.setTargetFeatures(target_fpfh);
	sac_ia.setMinSampleDistance(0.1);//��������֮�����С����
	sac_ia.setCorrespondenceRandomness(6); //��ѡ�����������Ӧʱ������Ҫʹ�õ��ھӵ�����;
										   //Ҳ���Ǽ���Э����ʱѡ��Ľ��ڵ��������ֵԽ��Э����Խ��ȷ�����Ǽ���Ч��Խ��.(��ʡ)
	pointcloud::Ptr align(new pointcloud);
	sac_ia.align(*align);
	end = clock();
	pcl::transformPointCloud(*source_cloud, *align, sac_ia.getFinalTransformation());
	cout << "calculate time is: " << float(end - start) / CLOCKS_PER_SEC << "s" << endl;
	cout << "\nSAC_IA has converged, score is " << sac_ia.getFitnessScore() << endl;
	cout << "�任����\n" << sac_ia.getFinalTransformation() << endl;
	//-------------------���ӻ�------------------------------------
	visualizeCloud(source_cloud, target_cloud, align);
}

//PCL ����Ŀ���³����̬����
//��һ������������һ�������Ӳ����ڵ��ĳ�������

//PCA ʵ�ֵ��ƴ���׼
//ͨ�����Ա任��ȡ���ݵ���Ҫ���������������ڽ�ά
//���õ������ݵ����᷽�������׼-->ͨ�����������ת���󣬼���������������ƫ��
//��Ҫ���г�ʼRTУ��
#include<pcl/registration/correspondence_estimation.h>
// ���������������������
void ComputeEigenVectorPCA(const pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
	Eigen::Vector4f& pcaCentroid, Eigen::Matrix3f& eigenVectorsPCA)
{
	pcl::compute3DCentroid(*cloud, pcaCentroid);
	Eigen::Matrix3f covariance;
	pcl::computeCovarianceMatrixNormalized(*cloud, pcaCentroid, covariance);
	Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> eigen_solver(covariance, Eigen::ComputeEigenvectors);
	eigenVectorsPCA = eigen_solver.eigenvectors();
}
// PCA���任����
Eigen::Matrix4f PCARegistration(pcl::PointCloud<pcl::PointXYZ>::Ptr& P_cloud, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& X_cloud)
{
	Eigen::Vector4f Cp;                  // P_cloud������
	Eigen::Matrix3f Up;                  // P_cloud����������
	ComputeEigenVectorPCA(P_cloud, Cp, Up);// ����P_cloud�����ĺ���������
	Eigen::Vector4f Cx;                  // X_cloud������
	Eigen::Matrix3f Ux;                  // X_cloud����������
	ComputeEigenVectorPCA(X_cloud, Cx, Ux);// ����X_cloud�����ĺ���������
										   // �ֱ����������Ӧ��8�������ѡ�������Сʱ��Ӧ�ı任����
	float error[8] = {};
	vector<Eigen::Matrix4f>MF;
	Eigen::Matrix4f final_RT = Eigen::Matrix4f::Identity();// �������յı任����
	Eigen::Matrix3f Upcopy = Up;
	int sign1[8] = { 1, -1,1,1,-1,-1,1,-1 };
	int sign2[8] = { 1, 1,-1,1,-1,1,-1,-1 };
	int sign3[8] = { 1, 1, 1,-1,1,-1,-1,-1 };
	for (int nn = 0; nn < 8; nn++) {
		Up.col(0) = sign1[nn] * Upcopy.col(0);
		Up.col(1) = sign2[nn] * Upcopy.col(1);
		Up.col(2) = sign3[nn] * Upcopy.col(2);
		Eigen::Matrix3f R = Eigen::Matrix3f::Identity();
		R = (Up * Ux.inverse()).transpose(); // ������ת����
		Eigen::Matrix<float, 3, 1> T;
		T = Cx.head<3>() - R * (Cp.head<3>());// ����ƽ������
		Eigen::Matrix4f RT = Eigen::Matrix4f::Identity();// ��ʼ���������4X4�任����
		RT.block<3, 3>(0, 0) = R;// ����4X4�任�������ת����
		RT.block<3, 1>(0, 3) = T;// ����4X4�任�����ƽ�Ʋ���
		pcl::PointCloud<pcl::PointXYZ>::Ptr t_cloud(new pcl::PointCloud<pcl::PointXYZ>());
		pcl::transformPointCloud(*P_cloud, *t_cloud, RT);
		// ����ÿһ�������Ӧ��ƽ���������
		pcl::registration::CorrespondenceEstimation<pcl::PointXYZ, pcl::PointXYZ>core;
		core.setInputSource(t_cloud);
		core.setInputTarget(X_cloud);
		boost::shared_ptr<pcl::Correspondences> cor(new pcl::Correspondences);
		core.determineReciprocalCorrespondences(*cor);//˫��K����������ȡ������
		double mean = 0.0, stddev = 0.0;
		pcl::registration::getCorDistMeanStd(*cor, mean, stddev);
		error[nn] = mean;
		MF.push_back(RT);
	}
	// ��ȡ�����Сʱ����Ӧ������
	int min_index = distance(begin(error), min_element(error, error + 8));
	// �����Сʱ��Ӧ�ı任����Ϊ��ȷ�任����
	final_RT = MF[min_index];
	return final_RT;
}
void  visualize_registration(pcl::PointCloud<pcl::PointXYZ>::Ptr& source, 
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target, pcl::PointCloud<pcl::PointXYZ>::Ptr& regist)
{
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Registration"));
	int v1 = 0;
	int v2 = 1;
	viewer->setWindowName("PCA��׼���");
	viewer->createViewPort(0, 0, 0.5, 1, v1);
	viewer->createViewPort(0.5, 0, 1, 1, v2);
	viewer->setBackgroundColor(0, 0, 0, v1);
	viewer->setBackgroundColor(0.05, 0, 0, v2);
	viewer->addText("Raw point clouds", 10, 10, "v1_text", v1);
	viewer->addText("Registed point clouds", 10, 10, "v2_text", v2);
	//ԭʼ������ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> src_h(source, 0, 255, 0);
	//Ŀ�������ɫ
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ> tgt_h(target, 0, 0, 255);
	//ת�����Դ���ƺ�ɫ
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
	//--------------PCA����任����------------------
	Eigen::Matrix4f PCATransform = Eigen::Matrix4f::Identity();
	PCATransform = PCARegistration(source, target);
	cout << "PCA����任������ʱ�� " << time.toc() / 1000 << " s" << endl;
	cout << "�任����Ϊ��\n" << PCATransform << endl;
	//-----------------�����׼----------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr PCARegisted(new pcl::PointCloud<pcl::PointXYZ>());
	pcl::transformPointCloud(*source, *PCARegisted, PCATransform);
	//----------------���ӻ����---------------------
	visualize_registration(source, target, PCARegisted);
}

//-----------------------------------------����׼------------------------------------------//

//----------------------�㵽���ICP�㷨----------------------
//��С��Դ������Ŀ����ƶ�Ӧ��֮��ľ���--��׼׼��

//PCL ICP�㷨ʵ�ֵ��ƾ���׼
#include <pcl/registration/icp.h> // icp�㷨
void ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		������׼��Point Cloud Registration��ָ����������������(source)��(target) ��
		���һ���任Tʹ��T(Ps)��T(Pt)���غϳ̶Ⱦ����ܸߣ�Ps��Pt��Դ���ƺ�Ŀ������еĶ�Ӧ�㡣
		�任T�����Ǹ��Ե�(rigid)��Ҳ���Բ��ǣ�һ��ֻ���Ǹ��Ա任�����任ֻ������ת��ƽ�ơ�
		������׼���Է�Ϊ����׼��Coarse Registration���;���׼��Fine Registration��������
		����׼ָ��������������֮��ı任��ȫδ֪������½��н�Ϊ�ֲڵ���׼��Ŀ����Ҫ��Ϊ����׼�ṩ�Ϻõı任��ֵ��
		����׼���Ǹ���һ����ʼ�任����һ���Ż��õ�����ȷ�ı任��
		ĿǰӦ����㷺�ĵ��ƾ���׼�㷨�ǵ���������㷨��Iterative Closest Point, ICP�������ֱ��� ICP �㷨��
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------��ʼ��ICP����--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//----------------------icp���Ĵ���--------------------
	icp.setInputSource(source);            // Դ����
	icp.setInputTarget(target);            // Ŀ�����
	icp.setTransformationEpsilon(1e-10);   // Ϊ��ֹ����������Сת������
	icp.setMaxCorrespondenceDistance(1);  // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	icp.setEuclideanFitnessEpsilon(0.001);  // �������������Ǿ�������С����ֵ��ֹͣ������
	icp.setMaximumIterations(35);           // ����������
	icp.setUseReciprocalCorrespondences(true);//����Ϊtrue,��ʹ���໥��Ӧ��ϵ
											  // ������Ҫ�ĸ���任�Ա㽫�����Դ����ƥ�䵽Ŀ�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "�任����\n" << icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ����Դ���ƽ��б任
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
}

//PCL KD-ICPʵ�ֵ��ƾ���׼,����˫��KD������������һ�Զ������
void KDICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	/*
		ͨ��KD��ʵ�ֽ����㷨���Դ���׼��ĵ���н���������
		��ŷ�Ͼ���Ϊ�жϱ�׼�����ŷ�Ͼ��������ֵ����׼�ؼ��㣬������׼���ȸߵĵ�
	*/
	pcl::console::TicToc time;
	time.tic();
	//--------------------��ʼ��ICP����--------------------
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//---------------------KD����������--------------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(target);
	icp.setSearchMethodSource(tree1);
	icp.setSearchMethodTarget(tree2);
	//----------------------icp���Ĵ���--------------------
	icp.setInputSource(source);            // Դ����
	icp.setInputTarget(target);            // Ŀ�����
	icp.setTransformationEpsilon(1e-10);   // Ϊ��ֹ����������Сת������
	icp.setMaxCorrespondenceDistance(1);  // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	icp.setEuclideanFitnessEpsilon(0.05);  // �������������Ǿ�������С����ֵ�� ֹͣ������
	icp.setMaximumIterations(35);           // ����������
											//icp.setUseReciprocalCorrespondences(true);//ʹ���໥��Ӧ��ϵ
											// ������Ҫ�ĸ���任�Ա㽫�����Դ����ƥ�䵽Ŀ�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() << " ms" << endl;
	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "�任����\n" << icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ����Դ���ƽ��б任
	pcl::transformPointCloud(*source, *icp_cloud, icp.getFinalTransformation());
}

//PCL ����ʽ�����������׼--ͨ�������̿����㷨��������

//PCL ���������׼

//�����ĸ��֡���������SAC_IA��NDT�ںϵĵ�����׼����
#include <pcl/registration/ndt.h>      // NDT��׼�㷨
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
	source = voxel_grid_fiter(source_cloud); // �²����˲�
	target = voxel_grid_fiter(target_cloud); // �²����˲�

	// 1������Դ���ƺ�Ŀ����Ƶ�FPFH
	fpfhFeature::Ptr source_fpfh = compute_fpfh_feature(source);
	fpfhFeature::Ptr target_fpfh = compute_fpfh_feature(target);

	// 2������һ����SAC_IA��ʼ��׼
	clock_t start, end;
	start = clock();
	pcl::SampleConsensusInitialAlignment<pcl::PointXYZ, pcl::PointXYZ, pcl::FPFHSignature33> sac_ia;
	sac_ia.setInputSource(source);
	sac_ia.setSourceFeatures(source_fpfh);
	sac_ia.setInputTarget(target);
	sac_ia.setTargetFeatures(target_fpfh);
	sac_ia.setMinSampleDistance(0.01);       // ��������֮�����С����
	sac_ia.setMaxCorrespondenceDistance(0.1);// ���ö�Ӧ���֮���������
	sac_ia.setNumberOfSamples(200);          // ����ÿ�ε���������ʹ�õ�������������ʡ��,�ɽ�ʡʱ��
	sac_ia.setCorrespondenceRandomness(6);   // ������6�����������Ӧ�����ѡȡһ��
	pointcloud::Ptr align(new pointcloud);
	sac_ia.align(*align);
	Eigen::Matrix4f initial_RT = Eigen::Matrix4f::Identity();// �����ʼ�任����
	initial_RT = sac_ia.getFinalTransformation();
	cout << "\nSAC_IA has converged, score is " << sac_ia.getFitnessScore() << endl;
	cout << "�任����\n" << initial_RT << endl;

	// 3����̬�ֲ��任��NDT��
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setInputSource(source);	             // ����Ҫ��׼�ĵ���
	ndt.setInputTarget(target);              // ���õ�����׼Ŀ��
	ndt.setStepSize(4);                      // ΪMore-Thuente������������󲽳�
	ndt.setResolution(0.01);                 // ����NDT����ṹ�ķֱ��ʣ�VoxelGridCovariance��
	ndt.setMaximumIterations(35);            // ����ƥ�������������
	ndt.setTransformationEpsilon(0.01);      // Ϊ��ֹ����������Сת������
	pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*output_cloud, initial_RT);    // align()�����еڶ���������������ǳ�ʼ�任�Ĺ��Ʋ���
	end = clock();
	cout << "NDT has converged:" << ndt.hasConverged()
		<< " score: " << ndt.getFitnessScore() << endl;
	cout << "�任����\n" << ndt.getFinalTransformation() << endl;
	cout << "����ʱ��: " << float(end - start) / CLOCKS_PER_SEC << "s" << endl;

	// 4��ʹ�ñ任�����δ�����˲���ԭʼԴ���ƽ��б任
	pcl::transformPointCloud(*source_cloud, *output_cloud, ndt.getFinalTransformation());

	// 5�����ӻ�
	visualize_registration(source_cloud, target_cloud, output_cloud);
}

//�����ĸ��֡���������NDT��ICP��ϵĵ�����׼�㷨
typedef pcl::PointCloud<PointT> PointCloud;
// Ԥ�������
void pretreat(PointCloud::Ptr& pcd_cloud, PointCloud::Ptr& pcd_down, float LeafSize = 0.4) 
{
	//ȥ��NAN��
	std::vector<int> indices_src; //����ȥ���ĵ������
	pcl::removeNaNFromPointCloud(*pcd_cloud, *pcd_cloud, indices_src);
	std::cout << "remove *cloud_source nan" << endl;
	//�²����˲�
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid.setInputCloud(pcd_cloud);
	voxel_grid.filter(*pcd_down);
	cout << "down size *cloud from " << pcd_cloud->size() << "to" << pcd_down->size() << endl;
}
// ����תƽ�ƾ��������ת�Ƕ�
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
	cout << "x����ת�Ƕȣ�" << ax << endl;
	cout << "y����ת�Ƕȣ�" << ay << endl;
	cout << "z����ת�Ƕȣ�" << az << endl;
}
void NDT_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_source)
{
	clock_t start = clock();
	PointCloud::Ptr cloud_src(new PointCloud);
	PointCloud::Ptr cloud_tar(new PointCloud);
	pretreat(cloud_source, cloud_src);
	pretreat(cloud_target, cloud_tar);
	//NDT��׼
	pcl::NormalDistributionsTransform<PointT, PointT> ndt;
	PointCloud::Ptr cloud_ndt(new PointCloud);
	ndt.setInputSource(cloud_src);
	ndt.setInputTarget(cloud_tar);
	ndt.setStepSize(0.5);              // ΪMore-Thuente������������󲽳�
	ndt.setResolution(2);              // ����NDT����ṹ�ķֱ��ʣ�VoxelGridCovariance��
	ndt.setMaximumIterations(35);      // ����ƥ�������������
	ndt.setTransformationEpsilon(0.01);// Ϊ��ֹ����������Сת������
	ndt.align(*cloud_ndt);
	clock_t ndt_t = clock();
	cout << "ndt time" << (double)(ndt_t - start) / CLOCKS_PER_SEC << endl;
	Eigen::Matrix4f ndt_trans = ndt.getFinalTransformation();

	//icp��׼�㷨
	pcl::IterativeClosestPoint<PointT, PointT> icp;
	PointCloud::Ptr cloud_icp_registration(new PointCloud);
	//���ò���
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

	//�������
	Eigen::Vector3f ANGLE_origin;
	Eigen::Vector3f TRANS_origin;
	ANGLE_origin << 0, 0, M_PI / 4;
	TRANS_origin << 0, 0.3, 0.2;
	double a_error_x, a_error_y, a_error_z;
	double t_error_x, t_error_y, t_error_z;
	Eigen::Vector3f ANGLE_result; // ��IMU�ó��ı任����

	matrix2angle(icp_trans, ANGLE_result);
	a_error_x = fabs(ANGLE_result(0)) - fabs(ANGLE_origin(0));
	a_error_y = fabs(ANGLE_result(1)) - fabs(ANGLE_origin(1));
	a_error_z = fabs(ANGLE_result(2)) - fabs(ANGLE_origin(2));
	cout << "����ʵ����ת�Ƕ�:\n" << ANGLE_origin << endl;
	cout << "x����ת��� : " << a_error_x << "  y����ת��� : " << a_error_y << "  z����ת��� : " << a_error_z << endl;

	cout << "����ʵ��ƽ�ƾ���:\n" << TRANS_origin << endl;
	t_error_x = fabs(icp_trans(0, 3)) - fabs(TRANS_origin(0));
	t_error_y = fabs(icp_trans(1, 3)) - fabs(TRANS_origin(1));
	t_error_z = fabs(icp_trans(2, 3)) - fabs(TRANS_origin(2));
	cout << "����õ���ƽ�ƾ���" << endl << "x��ƽ��" << icp_trans(0, 3) << endl << "y��ƽ��" << icp_trans(1, 3) << endl << "z��ƽ��" << icp_trans(2, 3) << endl;
	cout << "x��ƽ����� : " << t_error_x << "  y��ƽ����� : " << t_error_y << "  z��ƽ����� : " << t_error_z << endl;
	//���ӻ�
	visualize_registration(cloud_source, cloud_target, cloud_icp_registration);
}

//�����ĸ��֡������������������һ���ԸĽ�ICP�㷨������׼����
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/common/common.h>
#include <pcl/common/common_headers.h>
void extract_keypoint(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& keypoint,
	float LeafSize = 0.02, float radius = 0.04, float threshold = 5) // �����ֱ�Ϊ�����ظ����Ĵ�С������������뾶���н���ֵ���ȣ�
{
	//�����²���
	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	pcl::PointCloud<pcl::PointXYZ>::Ptr pcd_down(new pcl::PointCloud<pcl::PointXYZ>);
	voxel_grid.setInputCloud(cloud);
	voxel_grid.setLeafSize(LeafSize, LeafSize, LeafSize);
	voxel_grid.filter(*pcd_down);
	//����ÿһ����ķ�����
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> n;
	n.setInputCloud(pcd_down);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setSearchMethod(tree);
	//����KD�������뾶
	n.setKSearch(10);
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	n.compute(*normals);

	float Angle = 0.0;
	float Average_Sum_AngleK = 0.0;//����������K���㷨�����нǵ�ƽ��ֵ
	vector<int>indexes;
	//--------------���㷨�����нǼ��нǾ�ֵ----------------
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;  //����kdtree����
	kdtree.setInputCloud(pcd_down); //������Ҫ����kdtree�ĵ���ָ��
	vector<int> pointIdxRadiusSearch;  //����ÿ�����ڵ������
	vector<float> pointRadiusSquaredDistance;  //����ÿ�����ڵ�����ҵ�֮���ŷʽ����ƽ��
	pcl::PointXYZ searchPoint;
	for (size_t i = 0; i < pcd_down->points.size(); ++i) {
		searchPoint = pcd_down->points[i];
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			float Sum_AngleK = 0.0;//����K���ڽ��ĵ㷨��н�֮��
			 /*���㷨�����ļн�*/
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
				Sum_AngleK += Angle;//����н�֮��
			}		
			Average_Sum_AngleK = Sum_AngleK / pointIdxRadiusSearch.size();															  
			//��ȡ������
			float t = pcl::deg2rad(threshold);
			if (Average_Sum_AngleK > t) {
				indexes.push_back(i);
			}
		}
	}
	pcl::copyPointCloud(*pcd_down, indexes, *keypoint);
	cout << "��ȡ�����������:" << keypoint->points.size() << endl;
};
void SACIA_ICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& target,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& source)
{
	//1�� ��ȡ������
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	extract_keypoint(source, s_k);
	extract_keypoint(target, t_k);

	//2������Դ���ƺ�Ŀ����Ƶ�FPFH
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr sk_fpfh = compute_fpfh_feature(s_k);
	pcl::PointCloud<pcl::FPFHSignature33>::Ptr tk_fpfh = compute_fpfh_feature(t_k);

	//3��SAC��׼
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

	//4��KD���Ľ���ICP��׼
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	//kdTree ��������
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
	// 5�����ӻ�
	visualize_registration(source, target, icp_result);
}

//----------------------�㵽���ICP�㷨----------------------
//��С��Դ�����еĵ㵽Ŀ����ƶ�Ӧ������ƽ��ľ���
//�������ֵ��ƵĿռ�ṹ�����õĵֿ������Ӧ��ԣ����������ٶȸ���

//PCL �㵽���ICP�㷨
#include <pcl/registration/icp.h> // icp�㷨
void cloud_with_normal(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, 
	pcl::PointCloud<pcl::PointNormal>::Ptr& cloud_normals)
{
	//-----------------ƴ�ӵ��������뷨����Ϣ---------------------
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;//OMP����
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//����kdtree�����н��ڵ㼯����
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
	n.setNumberOfThreads(10);//����openMP���߳���
	n.setInputCloud(cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);//���Ʒ������ʱ����Ҫ���ѵĽ��ڵ��С
	n.compute(*normals);//��ʼ���з����					
	pcl::concatenateFields(*cloud, *normals, *cloud_normals);//�����������뷨����Ϣƴ��
}
void ICP_class(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------ƴ�ӵ����뷨����Ϣ-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//----------------�㵽���icp������棩-----------------
	pcl::IterativeClosestPointWithNormals<pcl::PointNormal, pcl::PointNormal>p_icp;
	p_icp.setInputSource(source_with_normals);
	p_icp.setInputTarget(target_with_normals);
	p_icp.setTransformationEpsilon(1e-10);    // Ϊ��ֹ����������Сת������
	p_icp.setMaxCorrespondenceDistance(10);   // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	p_icp.setEuclideanFitnessEpsilon(0.001);  // �������������Ǿ�������С����ֵ�� ֹͣ������
	p_icp.setMaximumIterations(35);           // ����������		

	//p_icp.setUseSymmetricObjective(true);������Ϊtrue���Ϊ��һ���㷨	
	pcl::PointCloud<pcl::PointNormal>::Ptr p_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	p_icp.align(*p_icp_cloud);
	cout << "\nICP has converged, score is " << p_icp.getFitnessScore() << endl;
	cout << "�任����\n" << p_icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ�����Դ���ƽ��б任
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, p_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//PCL �㵽���ICP����׼��LS������С�����Ż���
void ICP_LS(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------ƴ�ӵ����뷨����Ϣ-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------�㵽���ICP-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> icp;
	/*�㵽��ľ��뺯�����췽��һ*/
	pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>);
	/*�㵽��ľ��뺯�����췽����*/
	//typedef pcl::registration::TransformationEstimationPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal> PointToPlane;
	// boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	icp.setTransformationEstimation(PointToPlane);
	icp.setInputSource(source_with_normals);
	icp.setInputTarget(target_with_normals);
	icp.setTransformationEpsilon(1e-10);   // Ϊ��ֹ����������Сת������
	icp.setMaxCorrespondenceDistance(10);  // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	icp.setEuclideanFitnessEpsilon(0.001);  // �������������Ǿ�������С����ֵ�� ֹͣ������
	icp.setMaximumIterations(35);           // ����������
	pcl::PointCloud<pcl::PointNormal>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	icp.align(*icp_cloud);

	cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
	cout << "�任����\n" << icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ�����Դ���ƽ��б任
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//PCL �㵽���ICP�㷨��LM��������С�����Ż���
#include <pcl/registration/transformation_estimation_point_to_plane.h>
void ICP_LM(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------ƴ�ӵ����뷨����Ϣ-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------�㵽���icp���������Ż���-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal>lm_icp;
	pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationPointToPlane<pcl::PointNormal, pcl::PointNormal>);
	lm_icp.setTransformationEstimation(PointToPlane);
	lm_icp.setInputSource(source_with_normals);
	lm_icp.setInputTarget(target_with_normals);
	lm_icp.setTransformationEpsilon(1e-10);   // Ϊ��ֹ����������Сת������
	lm_icp.setMaxCorrespondenceDistance(10);  // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	lm_icp.setEuclideanFitnessEpsilon(0.001);  // �������������Ǿ�������С����ֵ�� ֹͣ������
	lm_icp.setMaximumIterations(35);           // ����������
	pcl::PointCloud<pcl::PointNormal>::Ptr lm_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	lm_icp.align(*lm_icp_cloud);
	cout << "\nICP has converged, score is " << lm_icp.getFitnessScore() << endl;
	cout << "�任����\n" << lm_icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ�����Դ���ƽ��б任
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, lm_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}

//----------------------�Ľ���ICP�㷨----------------------

//PCL Trimmed ICP
#include <pcl/recognition/ransac_based/trimmed_icp.h>
#include <pcl/recognition/ransac_based/auxiliary.h>
//��transform��: δ�ҵ�ƥ������غ���;��aux��: ������������ռ�����
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
	//icpʵ��
	time.tic();
	pcl::IterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> icp;
	icp.setMaximumIterations(iterations);
	icp.setMaxCorrespondenceDistance(15);   //�������Ķ�Ӧ�����
	icp.setTransformationEpsilon(1e-10);      //���þ���
	icp.setEuclideanFitnessEpsilon(0.01);
	icp.setInputSource(source_cloud);
	icp.setInputTarget(target_cloud);
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	icp.align(*icp_cloud);
	cout << "Applied " << iterations << "ICP iteration(s) in " << time.toc() / 1000 << "s" << endl;
	//���RT
	if (icp.hasConverged())
	{
		cout << "\nICP has converged, score is " << icp.getFitnessScore() << endl;
		cout << "\nICP transformation " << iterations << " : cloud_icp -> cloud_in" << endl;
		transformation_matrix = icp.getFinalTransformation().cast<double>();
		print4x4Matrix(transformation_matrix);
	}
	else
		PCL_ERROR("\nICP has not converged.\n");
	//trimmed icpʵ��
	time.tic();
	pcl::recognition::TrimmedICP<pcl::PointXYZ, double> Tricp;
	Tricp.init(target_cloud);     // target
	float sigma = 0.96;
	int Np = source_cloud->size();// num_source_points
	int Npo = Np * sigma;         // num_source_points_to_use
	Tricp.setNewToOldEnergyRatio(sigma);//����Խ����׼Խ׼ȷ
	Eigen::Matrix4d transformation_matrix1 = Eigen::Matrix4d::Identity();
	Tricp.align(*icp_cloud, Npo, transformation_matrix1);
	cout << "Applied Trimmed ICP iteration(s) in " << time.toc() / 1000 << "s" << endl;
	cout << "Trimmed icp pose:" << endl;
	print4x4Matrix(transformation_matrix1);
	pcl::PointCloud<pcl::PointXYZ>::Ptr Tricp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*icp_cloud, *Tricp_cloud, transformation_matrix1);
	visualize_registration(source_cloud, target_cloud, Tricp_cloud);
}

//PCL ʹ��GICP�Ե�����׼
#include <pcl/registration/gicp.h>  
//--ͨ��Э���������������Ȩ�ص����ã��������õĶ�Ӧ�����������е�����
//--�����˻�ΪICP���Ҵ���Ψһ��
void GICP(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	pcl::console::TicToc time;
	time.tic();
	//-----------------��ʼ��GICP����-------------------------
	pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZ, pcl::PointXYZ> gicp;
	//-----------------KD����������---------------------------
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree1(new pcl::search::KdTree<pcl::PointXYZ>);
	tree1->setInputCloud(source);
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree2(new pcl::search::KdTree<pcl::PointXYZ>);
	tree2->setInputCloud(target);
	gicp.setSearchMethodSource(tree1);
	gicp.setSearchMethodTarget(tree2);
	//-----------------����GICP��ز���-----------------------
	gicp.setInputSource(source);  //Դ����
	gicp.setInputTarget(target);  //Ŀ�����
	gicp.setMaxCorrespondenceDistance(100); //���ö�Ӧ���֮���������
	gicp.setTransformationEpsilon(1e-10);   //Ϊ��ֹ����������Сת������
	gicp.setEuclideanFitnessEpsilon(0.001);  //�������������Ǿ�������С����ֵ��ֹͣ����
	gicp.setMaximumIterations(35); 
	// ������Ҫ�ĸ���任�Ա㽫�����Դ����ƥ�䵽Ŀ�����
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	gicp.align(*icp_cloud);
	//---------------�����Ҫ��Ϣ����ʾ--------------------
	cout << "Applied " << 35 << " GICP iterations in " << time.toc() / 1000 << " s" << endl;
	cout << "\nGICP has converged, score is " << gicp.getFitnessScore() << endl;
	cout << "�任����\n" << gicp.getFinalTransformation() << endl;
	// ʹ�ñ任�����Ϊ������ƽ��б任
	pcl::transformPointCloud(*source, *icp_cloud, gicp.getFinalTransformation());
	visualize_registration(source, target, icp_cloud);
}

//PCL Ŀ�꺯���ԳƵ�ICP�㷨
#if 0
void ICP_symmetric(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	//-----------------ƴ�ӵ����뷨����Ϣ-------------------
	pcl::PointCloud<pcl::PointNormal>::Ptr source_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(source, source_with_normals);
	pcl::PointCloud<pcl::PointNormal>::Ptr target_with_normals(new pcl::PointCloud<pcl::PointNormal>);
	cloud_with_normal(target, target_with_normals);
	//--------------------�㵽���symm_icp-----------------------
	pcl::IterativeClosestPoint<pcl::PointNormal, pcl::PointNormal> symm_icp;
	//PCL1.11.1
	pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>::Ptr PointToPlane
	(new pcl::registration::TransformationEstimationSymmetricPointToPlaneLLS<pcl::PointNormal, pcl::PointNormal>);
	symm_icp.setTransformationEstimation(PointToPlane);
	symm_icp.setInputSource(source_with_normals);
	symm_icp.setInputTarget(target_with_normals);
	symm_icp.setTransformationEpsilon(1e-10);   // Ϊ��ֹ����������Сת������
	symm_icp.setMaxCorrespondenceDistance(10);  // ���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	symm_icp.setEuclideanFitnessEpsilon(0.001);  // �������������Ǿ�������С����ֵ�� ֹͣ������
	symm_icp.setMaximumIterations(50);           // ����������
	pcl::PointCloud<pcl::PointNormal>::Ptr symm_icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	symm_icp.align(*symm_icp_cloud);
	cout << "\nsymPlaneICP has converged, score is " << symm_icp.getFitnessScore() << endl;
	cout << "�任����\n" << symm_icp.getFinalTransformation() << endl;
	// ʹ�ô����ı任��Ϊ�����Դ���ƽ��б任
	pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::transformPointCloud(*source, *out_cloud, symm_icp.getFinalTransformation());
	visualize_registration(source, target, out_cloud);
}
#endif

//�����Լ�Ȩ��С�����Ż��ĵ㵽��ICP�㷨
using point_normal = pcl::PointCloud<pcl::PointNormal>;
//================���㷨��=====================
point_normal::Ptr wf(pcl::PointCloud<pcl::PointXYZ>::Ptr input_cloud)
{
	pcl::NormalEstimationOMP<pcl::PointXYZ, pcl::Normal> n;
	pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
	//����kdtree�����н��ڵ㼯����
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
	n.setNumberOfThreads(8);
	n.setInputCloud(input_cloud);
	n.setSearchMethod(tree);
	n.setKSearch(10);
	n.compute(*normals);
	//�����������뷨����Ϣƴ��
	pcl::PointCloud<pcl::PointNormal>::Ptr input_cloud_normals(new pcl::PointCloud<pcl::PointNormal>);
	pcl::concatenateFields(*input_cloud, *normals, *input_cloud_normals);
	return input_cloud_normals;
}
#include <pcl/registration/icp_nl.h>//LM-ICP��������
#include <pcl/registration/transformation_estimation_point_to_plane_weighted.h>
void ICP_NoneLLS(pcl::PointCloud<pcl::PointXYZ>::Ptr& source,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target)
{
	pcl::console::TicToc time;
	point_normal::Ptr tn = wf(target);
	point_normal::Ptr sn = wf(source);
	cout << "���߼�����ϣ�����" << endl;
	time.tic();
	//=======================��ʼ��=====================
	pcl::IterativeClosestPointNonLinear<pcl::PointNormal, pcl::PointNormal> icp;
	//=======PointToPlaneWeighted����㵽��ľ���================
	typedef pcl::registration::TransformationEstimationPointToPlaneWeighted <pcl::PointNormal, pcl::PointNormal> PointToPlane;
	boost::shared_ptr<PointToPlane> point_to_plane(new PointToPlane);
	//================��������=================
	icp.setTransformationEstimation(point_to_plane);
	icp.setInputSource(sn);
	icp.setInputTarget(tn);
	icp.setTransformationEpsilon(1e-10);   //Ϊ��ֹ����������Сת������
	icp.setMaxCorrespondenceDistance(10); //���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	icp.setEuclideanFitnessEpsilon(0.0001);  //�������������Ǿ�������С����ֵ�� ֹͣ������
	icp.setMaximumIterations(35); //������������  
	pcl::PointCloud<pcl::PointNormal>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointNormal>);
	icp.align(*icp_cloud);
	cout << "Applied " << 35 << " ICP iterations in " << time.toc() / 1000 << "s" << endl;
	cout << "�任����\n" << icp.getFinalTransformation() << endl;
	pcl::transformPointCloud(*sn, *icp_cloud, icp.getFinalTransformation());
	//=================���ӻ�����====================
	boost::shared_ptr<pcl::visualization::PCLVisualizer>
		viewer_final(new pcl::visualization::PCLVisualizer("��׼���"));
	viewer_final->setBackgroundColor(0, 0, 0); 
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		target_color(target, 255, 0, 0);
	viewer_final->addPointCloud<pcl::PointXYZ>(target, target_color, "target cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "target cloud");
	// ��Դ������ɫ���ӻ� (green).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>
		input_color(source, 0, 255, 0);
	viewer_final->addPointCloud<pcl::PointXYZ>(source, input_color, "input cloud");
	viewer_final->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE,
		1, "input cloud");
	// ��ת�����Դ������ɫ (blue)���ӻ�.
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

//LM-ICP ʵ�ֵ��ƾ���׼
#include <pcl/filters/random_sample.h>//��ȡ�̶������ĵ���
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
	// ----------------------�������������--------------------------
	pcl::PointCloud<pcl::PointXYZ>::Ptr s_k(new pcl::PointCloud<pcl::PointXYZ>);
	random_sample_point(source, s_k, 3000);
	pcl::PointCloud<pcl::PointXYZ>::Ptr t_k(new pcl::PointCloud<pcl::PointXYZ>);
	random_sample_point(target, t_k, 3000);
	time.tic();
	// ------------------------LM-ICP--------------------------------
	pcl::IterativeClosestPointNonLinear<pcl::PointXYZ, pcl::PointXYZ> lmicp;
	lmicp.setInputSource(s_k);
	lmicp.setInputTarget(t_k);
	lmicp.setTransformationEpsilon(1e-10);    //Ϊ��ֹ����������Сת������
	lmicp.setMaxCorrespondenceDistance(10);   //���ö�Ӧ���֮��������루��ֵ����׼���Ӱ��ϴ󣩡�
	lmicp.setEuclideanFitnessEpsilon(0.0001); //�������������Ǿ�������С����ֵ�� ֹͣ������
	lmicp.setMaximumIterations(35);           //������������  
	pcl::PointCloud<pcl::PointXYZ>::Ptr icp_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	lmicp.align(*icp_cloud);
	cout << "Applied " << 35 << " LM-ICP iterations in " << time.toc() / 1000 << "s" << endl;
	cout << "�任����\n" << lmicp.getFinalTransformation() << endl;
	// ��Դ���ƽ��б任
	pcl::transformPointCloud(*source, *icp_cloud, lmicp.getFinalTransformation());
	visualize_registration(source, target, icp_cloud);
}

//----------------------���ڸ���ģ�͵��㷨----------------------

//PCL 3D-NDT �㷨ʵ�ֵ�����׼--������ά��������̬�ֲ�
#include <pcl/registration/ndt.h>               // NDT��׼
#include <pcl/filters/approximate_voxel_grid.h> // �����˲�
void _3DNDT(pcl::PointCloud<pcl::PointXYZ>::Ptr& source_cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& target_cloud)
{
	pcl::console::TicToc time;
	if (source_cloud->empty() || target_cloud->empty())
	{
		cout << "��ȷ�ϵ����ļ������Ƿ���ȷ" << endl;
		return;
	}
	else {
		cout << "��Ŀ����ƶ�ȡ " << target_cloud->size() << " ����" << endl;
		cout << "��Դ�����ж�ȡ " << source_cloud->size() << " ����" << endl;
	}

	//�������Դ���ƹ��˵�ԭʼ�ߴ�Ĵ��10%�����ƥ����ٶȡ�
	pcl::PointCloud<pcl::PointXYZ>::Ptr filtered_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::ApproximateVoxelGrid<pcl::PointXYZ> approximate_voxel_filter;
	approximate_voxel_filter.setInputCloud(source_cloud);
	approximate_voxel_filter.setLeafSize(0.1, 0.1, 0.1);
	approximate_voxel_filter.filter(*filtered_cloud);
	cout << "Filtered cloud contains " << filtered_cloud->size() << " data points " << endl;
	// -------------NDT������׼--------------
	time.tic();
	pcl::NormalDistributionsTransform<pcl::PointXYZ, pcl::PointXYZ> ndt;
	ndt.setStepSize(4);                 // ΪMore-Thuente������������󲽳�
	ndt.setResolution(0.1);             // ����NDT����ṹ�ķֱ��ʣ�VoxelGridCovariance��
	ndt.setMaximumIterations(35);       // ����ƥ�������������
	ndt.setInputSource(filtered_cloud);	// ����Ҫ��׼�ĵ���
	ndt.setInputTarget(target_cloud);   // ���õ�����׼Ŀ��
	ndt.setTransformationEpsilon(0.01); // Ϊ��ֹ����������Сת������
	pcl::PointCloud<pcl::PointXYZ>::Ptr output_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	ndt.align(*output_cloud);
	cout << "NDT has converged:" << ndt.hasConverged()
		<< " score: " << ndt.getFitnessScore() << endl;
	cout << "Applied " << ndt.getMaximumIterations() << " NDT iterations in " << time.toc() << " ms" << endl;
	cout << "�任����\n" << ndt.getFinalTransformation() << endl;
	//ʹ�ñ任�����δ���˵�Դ���ƽ��б任
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