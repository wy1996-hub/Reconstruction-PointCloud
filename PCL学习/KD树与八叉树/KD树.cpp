#include <iostream>
#include <pcl/kdtree/kdtree_flann.h>//kdtree��������
#include <pcl/io/pcd_io.h>  //�ļ��������
#include <pcl/point_types.h>  //��������ض���
#include <pcl/visualization/pcl_visualizer.h>//���ӻ���ض���
#include <boost/thread/thread.hpp>

using namespace std;

/*
	KD-tree��һ�����ݽṹ��������֯����Kά�Ŀռ��е����ɸ��㣬��һ����������Լ���Ķ���λ������
	����1ά���ݵĲ�ѯ��ʹ��ƽ������������������ɣ�KD-tree����һ�ָ�γ���ݵĿ��ٲ�ѯ�ṹ,һ����Զ�ά���ݵ�����һά����������
	����KD������Ը�ά���ݣ���Ҫ���ÿһά�����ж���
	depth��ʾ��ǰ��KD���ĵڼ��㣬���depth��ż����ͨ�������߶Լ��Ͻ��л��֣����depth��������ͨ�������߽��л���
	kd�����ڷ�Χ�����䣩�����ö�����������ǳ�����
	һ��ֻ������ά���ƣ�����kd��������ά��
	kd���Ĳ���ʱ�临�Ӷȣ�nlogn����ͨ����ֱ�ڵ��Ƶ�һά��ƽ�棬���ռ�ݹ�ָ�Ϊ����ӿռ�ʵ�ֵ������ݿ��ټ���

	��ά������kd-tree����ϸ������̣�������λ�������ָ�ƽ�棩
	1�����ݵ��Ƶ�ȫ������ϵ�����������е��Ƶ��������Χ��
	2����ÿ����������1����������壬�����ָ�ƽ��
	3�������ָ��ֿռ�ͷָ�ƽ���ϵĵ㹹�ɷ�֧�����ӵ�
	4���ָ��ֿռ䣬�����ڲ�����������1����ִ�в���2�����ָ�
*/

//PCL KD����ʹ��
void kdtreeUSE()
{
	// 1��������ϵͳʱ��Ϊrand()���ӣ�Ȼ����������ݴ��������PointCloud
	srand(time(nullptr));
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);
	cloud->height = 1;//�����������
	cloud->width = 1000;
	cloud->points.resize(cloud->height * cloud->width);
	for (size_t i = 0; i < cloud->size(); i++)
	{
		cloud->points[i].x = 1024.f*rand() / (RAND_MAX + 1.f);
		(*cloud)[i].y = 1024.f*rand() / (RAND_MAX + 1.f);
		(*cloud)[i].z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	}
	// 2������kdtree���󣬲������������������Ϊ������ơ�
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	// 3��Ȼ��ָ��һ�����������Ϊ�������㡱
	pcl::PointXYZ searchPoint;  
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);
	// 4������һ������(����������Ϊ10)���������������ڴ������д洢�������K����
	int K = 10;                                   // ��Ҫ���ҵĽ��ڵ����
	std::vector<int> pointIdxKNNSearch(K);        // ����ÿ�����ڵ������
	std::vector<float> pointKNNSquaredDistance(K);// ����ÿ�����ڵ�����ҵ�֮���ŷʽ����ƽ��
	std::cout << "K nearest neighbor search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with K=" << K << std::endl;
	// 5����ӡ������������㡱������10������ھӵ�λ�ã���Щλ���Ѿ��洢����ǰ������������
	if (kdtree.nearestKSearch(searchPoint, K, pointIdxKNNSearch, pointKNNSquaredDistance) > 0)// > 0��ʾ�ܹ��ҵ����ڵ㣬 = 0��ʾ�Ҳ������ڵ�
	{
		for (std::size_t i = 0; i < pointIdxKNNSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxKNNSearch[i]].x
			<< " " << (*cloud)[pointIdxKNNSearch[i]].y
			<< " " << (*cloud)[pointIdxKNNSearch[i]].z
			<< " (squared distance: " << pointKNNSquaredDistance[i] << ")" << std::endl;
	}
	// 6������һ������뾶���������������ڴ������д洢�������K����
	float radius = 256.0f * rand() / (RAND_MAX + 1.0f); // ���Ұ뾶��Χ
	std::vector<int> pointIdxRadiusSearch;              // ����ÿ�����ڵ������
	std::vector<float> pointRadiusSquaredDistance;      // ����ÿ�����ڵ�����ҵ�֮���ŷʽ����ƽ��
	std::cout << "Neighbors within radius search at (" << searchPoint.x
		<< " " << searchPoint.y
		<< " " << searchPoint.z
		<< ") with radius=" << radius << std::endl;
	// 7����ӡ������������뾶���ҵ��ĵ��λ��
	if(kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
	{
		for (std::size_t i = 0; i < pointIdxRadiusSearch.size(); ++i)
			std::cout << "    " << (*cloud)[pointIdxRadiusSearch[i]].x
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].y
			<< " " << (*cloud)[pointIdxRadiusSearch[i]].z
			<< " (squared distance: " << pointRadiusSquaredDistance[i] << ")" << std::endl;
	}
}

//PCL addLine���ӻ�K����
void kdtreeVisualNearestK()
{
	pcl::PointCloud<pcl::PointXYZRGBA>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGBA>);
	pcl::PCDReader reader;
	reader.read("../bunny-color.pcd", *cloud);//��ȡPCD�ļ�
	cout << "PointCloud  has: " << cloud->points.size() << " data points." << endl;
	//����kdtree���󣬲�����ȡ���ĵ�������Ϊ���롣
	pcl::KdTreeFLANN<pcl::PointXYZRGBA> kdtree;
	kdtree.setInputCloud(cloud);
	//��ʼ��������
	pcl::PointXYZRGBA searchPoint;
	searchPoint.x = 0;
	searchPoint.y = 0;
	searchPoint.z = 0;
	//���ӻ� ע������Ŀ��ӻ��Ǵ������ӻ�������������һ��������չʾͼ���ں���Ĵ���Ż���ʾ�����ļ���
	boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer(new pcl::visualization::PCLVisualizer("Viewer plane"));
	viewer->addPointCloud<pcl::PointXYZRGBA>(cloud, "sample cloud");
	//Ϊ�˸�ֱ�۵Ŀ�������ڵ㣬�����еĵ����ѯ�����ߣ�ÿ�����߶���Ҫ���Լ�������ID�������ظ������Զ����ַ�����������ͬ��ID
	string lineId = "line";
	stringstream ss;//ͨ������ʵ���ַ��������ֵ�ת��
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
			ss << i;// ��int���͵�ֵ������������
			lineId += ss.str();//����ʹ�� str() �������� stringstream ����ת��Ϊ string ����
			//��ӵ����ѯ��֮������ߣ����ֱ�ʾ����ߵ���ɫ
			viewer->addLine<pcl::PointXYZRGBA>(cloud->points[pointIdxNKNSearch[i]], searchPoint, 1, 58, 82, lineId);
			cout << "    " << cloud->points[pointIdxNKNSearch[i]]
				<< " (squared distance: " << pointNKNSquaredDistance[i] << ")" << endl;
			ss.str("");//ֱ����clearû��Ч����clear() ���������ڽ��ж����������ת���ĳ���
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
//������Ƶ�ƽ���ܶ�
double computeCloudResolution(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	/*
		�����豸��ͬ���豸���볡��Զ����ͬ����ʹ�õ����ܶȲ�������
		���е����ܶȹ��Ʒ��������ھ���+���ڷֿ�
		���ھ��룺������Ƹ���ľ���ƽ��ֵ�����Ƶ��Ʒֲ����̶ܳȣ�ĳһ�㵽�ݴ˵��������ĵ�ľ���
	*/

	double resolution = 0.0;
	int numberOfPoints = 0;
	std::vector<int> indices(2);
	std::vector<float> squaredDistances(2);
	pcl::search::KdTree<pcl::PointXYZ> tree;//����flann�����kdtree
	tree.setInputCloud(cloud);

	for (size_t i = 0; i < cloud->size(); ++i)
	{    
		//����Ƿ������Ч��
		//pcl���ݴ���ʱ���ܶ��㷨�ῼ����Ч�㣬��Ҫͨ���ж�pointcloud���е����ݳ�Ա�Ƿ����nan
		//isFinite��������һ��bool�����ĳ��ֵ�ǲ���������ֵ���������pcl::removeNaNFromPointCloud
		if (!pcl::isFinite(cloud->points[i]))
			continue;//skip nans

		//��ͬһ�������ڽ���k��������ʱ��k=1�ĵ�Ϊ��ѯ�㱾��
		int nres = tree.nearestKSearch(i, 2, indices, squaredDistances);
		if (nres == 2)
		{
			resolution += sqrt(squaredDistances[1]);
			++numberOfPoints;
		}
	}
	if (numberOfPoints != 0)
		resolution /= numberOfPoints;
	cout << "�����ܶ�Ϊ��" << resolution << endl;

	return resolution;
}

#include <pcl/common/utils.h>
//������Ƶ�ƽ���ܶ��Ż���
double computeCloudResolutionEx(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, float max_dist, int nr_threads)
{
	/*
		����ն�����������ܶȵ�Ӱ��
		��K��������ʱ���������Լ��������³���Ը�ǿ
		max_dist:����Ϊ����ĵ��������;
		nr_threads:Ҫʹ�õ��߳���(Ĭ��ֵ=1����������OpenMP��־ʱʹ��)
	*/

	const float max_dist_sqr = max_dist * max_dist;
	const std::size_t s = cloud->points.size();

	pcl::search::KdTree <pcl::PointXYZ> tree;
	tree.setInputCloud(cloud);

	float mean_dist = 0.f;
	int num = 0;
	std::vector <int> ids(2);
	std::vector <float> dists_sqr(2);
	//--------------���̼߳��ٿ�ʼ-------------
	/*pcl::utils::ignore(nr_threads);
	#pragma omp parallel for \
	default(none) \
	shared(tree, cloud) \
	firstprivate(ids, dists_sqr) \
	reduction(+:mean_dist, num) \
	firstprivate(s, max_dist_sqr) \
	num_threads(nr_threads)*/
	//--------------���̼߳��ٽ���--------------
	for (int i = 0; i < 1000; i++)//�������1000����
	{
		tree.nearestKSearch((*cloud)[rand() % s], 2, ids, dists_sqr);
		if (dists_sqr[1] < max_dist_sqr)//����Լ������
		{
			mean_dist += std::sqrt(dists_sqr[1]);
			num++;
		}
	}
	cout << "�����ܶ�Ϊ��" << (mean_dist / num) << endl;

	return (mean_dist / num);
}

#include <pcl/filters/extract_indices.h>
//ɾ���������ص��ĵ�
void deleteOverlappedPoints(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud_filtered)
{
	/*
		��ĳһ����ĳһ������ֵ(0.000001)�����ڲ�ֹ�䱾��һ���㣬����Ϊ�����ظ��㡣
		���ظ����������¼���������ں����Դ��ظ���Ϊ��ѯ������ʱ����ʱ��һ��Ҳ�ᱻ����Ϊ�ظ��㣬
		��pointIdxRadiusSearch�ж����������еģ��ʴ�pointIdxRadiusSearch�еĵڶ������������ʼ��¼��
		�������Ա�֤����ɾ���ظ��ĵ㣬������һ����
	*/

	// 1 KD���뾶����
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	vector<int> pointIdxRadiusSearch;//����ÿ�����ڵ������
	vector<float> pointRadiusSquaredDistance;//����ÿ�����ڵ�����ҵ�֮���ŷʽ����ƽ��
	vector<int> total_index;
	//������֮��ľ���Ϊ0.000001����Ϊ���غϵ�
	float radius = 0.000001;

	// 2 ��cloud�е�ÿ�����������ڵĵ���бȽ�
	for (size_t i = 0; i < cloud->size(); ++i)
	{
		pcl::PointXYZ searchPoint = cloud->points[i];
		if (kdtree.radiusSearch(searchPoint, radius, pointIdxRadiusSearch, pointRadiusSquaredDistance) > 0)
		{
			if (pointIdxRadiusSearch.size() != 1)
			{
				for (size_t j = 1; j < pointIdxRadiusSearch.size(); j++)//��pointIdxRadiusSearch�еĵڶ������������ʼ��¼
				{
					total_index.push_back(pointIdxRadiusSearch[j]);
				}
			}
		}
	}

	// 3 ɾ���ظ�����
	sort(total_index.begin(), total_index.end());//��������������
	total_index.erase(unique(total_index.begin(), total_index.end()), total_index.end());//�������е��ظ�����ȥ��
	//��������ɾ���ظ��ĵ�
	pcl::PointIndices::Ptr outliners(new pcl::PointIndices());
	outliners->indices.resize(total_index.size());
	for (size_t i = 0; i < total_index.size(); i++)
	{
		outliners->indices[i] = total_index[i];
	}
	cout << "�ظ�����ɾ����ϣ�����" << endl;

	// 4 ��ȡɾ���ظ���֮��ĵ���
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud(cloud);
	extract.setIndices(outliners);
	extract.setNegative(true);//����Ϊtrue���ʾ��������֮��ĵ�
	extract.filter(*cloud_filtered);
	cout << "ԭʼ�����е�ĸ���Ϊ��" << cloud->points.size() << endl;
	cout << "ɾ�����ظ���ĸ���Ϊ:" << total_index.size() << endl;
	cout << "ȥ��֮���ĸ���Ϊ:" << cloud_filtered->points.size() << endl;
}

//------------------------------------------------------------------------------------------------//

//��ʾ����--˫����
void visualizeKDCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
	pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudCut)
{
	//PCL���ӻ�����
	boost::shared_ptr<pcl::visualization::PCLVisualizer> view(new pcl::visualization::PCLVisualizer("ShowClouds"));
	int v1 = 0, v2 = 0;
	view->createViewPort(.0, .0, .5, 1., v1);
	view->setBackgroundColor(0., 0., 0., v1);
	view->addText("Raw point clouds", 10, 10, "text1", v1);
	view->createViewPort(.5, .0, 1., 1., v2);
	view->setBackgroundColor(0., 0., 0., v2);
	view->addText("sampled point clouds", 10, 10, "text2", v2);

	//����z�ֶν�����Ⱦ
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildColor(cloud, "z");
	pcl::visualization::PointCloudColorHandlerGenericField<pcl::PointXYZ> fildCutColor(cloudCut, "z");
	view->addPointCloud<pcl::PointXYZ>(cloud, fildColor, "Raw point clouds", v1);
	view->addPointCloud<pcl::PointXYZ>(cloudCut, fildCutColor, "sampled point clouds", v2);
	//���õ��Ƶ���Ⱦ����,string &id = "cloud"�൱�ڴ���ID
	//����z�ֶν�����Ⱦ������Ҫ������Ⱦÿ��������ɫ
	//view->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR, 0, 1, 0, "Raw point clouds", v1);//���õ�����ɫ
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
	//addLine���ӻ�K����
	//kdtreeVisualNearestK();
	//������Ƶ�ƽ���ܶ�
	//double density = computeCloudResolution(cloud);
	//double density = computeCloudResolutionEx(cloud, .2f, 1);
	//ɾ���������ص��ĵ�
	deleteOverlappedPoints(cloud, cloud_filtered);

	visualizeKDCloud(cloud, cloud_filtered);
	return;
}