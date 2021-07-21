#include <iostream>
#include <pcl/io/pcd_io.h>  
#include <pcl/point_cloud.h>  
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread/thread.hpp>

using namespace std;

/*
	�����ռ������ڵ������ݴ������ѱ��㷺��Ӧ�ã������Ŀռ�����һ�����Զ������𼶻��ֿռ�ĸ��ֿռ������ṹ.
	�˲����ṹͨ������ά�ռ�ļ���ʵ�������Ԫ�ʷ֣�ÿ����Ԫ������ͬ��ʱ��Ϳռ临�Ӷ�;
	ͨ��ѭ���ݹ�Ļ��ַ����Դ�СΪ( 2 nx 2 n x 2 n ) ����ά�ռ�ļ��ζ�������ʷ֣��Ӷ�����һ�����и��ڵ�ķ���ͼ.
	�˲�����һ�����ڹ���ϡ��3D���Ƶ���״���ݽṹ����ʵ�֡������ڽ���������K�����������뾶�ڽ���������

	ʵ��Octree�Ĳ���:
	(1). �趨���ݹ����
	(2). �ҳ����������ߴ磬���Դ˳ߴ罨����һ��������
	(3). ���򽫵�λԪԪ�ض����ܱ�������û���ӽڵ��������
	(4). ��û�дﵽ���ݹ���ȣ��ͽ���ϸ�ְ˵ȷݣ��ٽ�����������װ�ĵ�λԪԪ��ȫ���ֵ����˸���������
	(5). �������������������䵽�ĵ�λԪԪ��������Ϊ���Ҹ�����������һ���ģ������������ֹͣϸ��;
	��Ϊ���ݿռ�ָ����ۣ�ϸ�ֵĿռ����õ��ķ���ض����٣�����һ����Ŀ��������ô����Ŀ����һ��������������и�����Ρ�
	(6). �ظ�3��ֱ���ﵽ���ݹ���ȡ�

	PCL��octree ��ѹ���������ݷ���Ӧ��:
	�����ɺ��������ݼ���ɣ���Щ����ͨ�����롢��ɫ�����ߵȸ�����Ϣ�������ռ���ά�㡣
	���⣬�������Էǳ��ߵ����ʱ����������������Ҫռ���൱��Ĵ洢��Դ��
	һ��������Ҫ�洢����ͨ�����������Ƶ�ͨ���ŵ����д��䣬�ṩ����������ݵ�ѹ�������ͱ��ʮ�����á�
	PCL���ṩ�˵���ѹ�����ܣ����������ѹ���������͵ĵ��ƣ�����������ƣ�
	�������޲ο���ͱ仯�ĵ�ĳߴ硢�ֱ��ʡ��ֲ��ܶȺ͵�˳��Ƚṹ������
	���ң��ײ��octree���ݽṹ����Ӽ�������Դ��Ч�غϲ��������ݡ�

	�˲�����k-d���Ƚ�:
	(1).�˲����㷨���㷨ʵ�ּ򵥣��������������������£��Ƚ����ѵ�����С���ȣ�Ҷ�ڵ㣩��ȷ����
	���Ƚϴ�ʱ���еĽڵ������������ԱȽϴ󣬺�����ѯЧ���ԱȽϵͣ���֮�����Ƚ�С���˲�����������ӣ�
	��Ҫ���ڴ�ռ�Ҳ�Ƚϴ�ÿ����Ҷ�ӽڵ���Ҫ�˸�ָ�룩��Ч��Ҳ���͡�
	���ȷֵĻ������ݣ�ʹ��������������ƫб������£��ܻ���������ƣ���Ч�ʲ���̫�ߡ�
	(2).k-d����������ϱȽ������ƣ����ڴ�������������£����������Ƚ�Сʱ�������Ŀ���Ҳ�ϴ󣬵��Ȱ˲������Щ��
	��С������������£�������Ч�ʱȽϸߣ��������������������£���Ч�ʻ���һ�����½���һ�������������Ĺ��ɡ�
	(3).Ҳ�н��˲�����k-d�����������Ӧ�ã�Ӧ�ð˲������д����ȵĻ��ֺͲ��ң�����ʹ��k-d������ϸ�֣�
	Ч�ʻ���һ������������������Ч�ʱ仯Ҳ���������ı仯��һ�����Թ�ϵ��

	OCtoMap������һ�ָ�Ч�Ŀ��Ժܺõ�ѹ�����ƽ�ʡ�洢�ռ䣬��ʵʱ���µ�ͼ�������÷ֱ��ʵİ˲�����ͼ��
*/

//------------------------------------------------------------------------------------------------//

//�˲�����ʹ��
#include <pcl/octree/octree_search.h>
void octreeUSE(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& mcloud)
{
	srand((unsigned int)time(NULL));
	//���岢ʵ����һ�������PointCloud�ṹ����ʹ��������������
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
	* 1������һ���˲���ʵ����ʹ�ð˲����ֱ��ʽ��г�ʼ��������˲���������Ҷ�ڵ���*
	*������һ���������������ֱ��ʲ���������Ͱ˲�����������С���صĳ��ȡ���ˣ�  *
	*�˲���������Ƿֱ��ʵĺ�����Ҳ�ǵ��ƵĿռ�ά���ĺ��������֪�����Ƶı߽��*
	*��Ӧ��ʹ��finebeliingBox�������������˲�����                              *
	*Ȼ��ΪPointCloud����һ��ָ�룬�������еĵ���ӵ��˲����С�                *
	-----------------------------------------------------------------------------*/
	float resolution = 128.0f;                                            // �˲����ֱ���
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);// ʹ�÷ֱ��ʳ�ʼ���˲���
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	pcl::PointXYZ searchPoint;
	searchPoint.x = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.y = 1024.0f * rand() / (RAND_MAX + 1.0f);
	searchPoint.z = 1024.0f * rand() / (RAND_MAX + 1.0f);

	/*------------------------------------------------------------------------------
	* 2��һ��PointCloud��˲�����������Ϳ���ִ����������������ʹ�õĵ�һ�������� *
	*�ǡ�Voxel�����е��ھӡ�������������������Ӧ��Ҷ�ڵ����أ������ص�����������*
	*��Щָ��������ͬһ���ط�Χ�ڵĵ��йء�                                       *
	*��ˣ����������������֮��ľ���ȡ���ڰ˲����ķֱ��ʲ�����                   *
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
	* 3����Σ�֤����K���������������������У�K������Ϊ10��                           *
	*��K������������������������д�����������������С�                            *
	*��һ����pointIdxNKNSearch�����������������(�������PointCloud���ݼ�������)��  *
	*�ڶ���������������������֮�䱣����Ӧ��ƽ�����롣                             *
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
	* 4�����뾶�����е��ھӡ��Ĺ���ԭ��ǳ������ڡ�K�������������              *
	*�������������д�������ֱ�������������ƽ�����������������С�         *
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

#include <pcl/compression/octree_pointcloud_compression.h> //����ѹ��
#include <pcl/io/ply_io.h>
//PCL �˲�����Ӧ�á�������ѹ��
void pointCloudCompression(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& sourceCloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& cloudOut)
{
	/*
		�����ļ�Ԥ��Ϊ����ѹ�������˲�������ѹ����û��Ҫ������Щ������
		compressionProfile = MED_RES_ONLINE_COMPRESSION_WITH_COLOR, �����ļ�
		showStatistics = false, �Ƿ�ѹ����ص�ͳ����Ϣ��ӡ����׼�����
		pointResolution = 0.001, ���������ı��뾫�ȣ�Ӧ����ΪС�ڴ��������ȵ�ֵ����ķֱ��ʾ��������ڱ���ʱ���Ծ�ȷ�ĳ̶ȣ�����ϸ�ڱ���ʱ��Ч
		octreeResolution = 0.01, �˲����ֱ��ʣ����ְ˲���ʱ��С�飬��voxel�ı߳�
		doVoxelGridDownDownSampling = true, ���������²�����ÿ��������ֻ������������һ���㣻false�����ϸ�ڱ���
		iFrameRate = 100, ����������е�֡���ʵ�����������򲻽��б���ѹ����ÿ��һ��֡������I���룬�м�֡����P����
		doColorEncoding_arg = true, �Ƿ�Բ�ɫ����ѹ��
		colorBitResolution_arg = 6 ����ÿһ����ɫ�ɷֱ������ռ��λ��

		��������ͼ��
		�˲�������ṹ->�Ƿ����ϸ��->ϸ�ڱ���->�Ƿ������ɫ->��ɫ����->�ر���
		1���ڸ�����ǰ�����ж���ǰ֡��I֡����P֡��I֡���������룻P֡���ڱ���˲���ռλ��ʱ��������Ǻ�֮ǰ֡�����֮��Ĳ�ֵ
		2���˲����Ľṹ��������ڰ˲�������֮�󣬱������ص�ռλ�룬���doVoxelGridDownDownSampling�����²�����������ϸ�ڱ��벿�֡�
		������ڽ���ʱֻҪ���ݰ˲����Ľṹ��Ϣ�ָ����˲����ṹ�����������ĵ�Ϊ������
		3��ϸ�ڱ���:���������ڵ�ϸ�ڡ����˲�������ʱ�����صĴ�С�ϴ󣨵�ĵ�λ�Ǻ��׼��ģ���ÿ�����Ӧһ���߳�1mm��������飩��
		�����������ð˲����ֱ���ʱ������Ϊ5mm����˲������ֵ��߳�5mm���������ֹͣ����ʱһ�������ڻ��ж���㡣
		��������ϸ�ڱ��룬doVoxelGridDownDownSamplingΪtrue���ڽ���ʱһ��������ֻ����������λ�ûָ�һ���㣨��������ľ�ֵ�����ᵼ�µ����ʧ
		��ʱ����ϸ�ڱ��룬�����һ�������ڵĵ��ϸ����Ϣ������λ�������ԣ�
	*/

	// 1�����ò���
	pcl::io::compression_Profiles_e compressionProfile = pcl::io::MANUAL_CONFIGURATION; // ����ѹ��ѡ��,���ø߼�����������
	bool showStatistics = true;              // �����Ƿ������ӡѹ����Ϣ
	const float pointResolution = 0.001;     // ���������ı��뾫�ȣ��ò���Ӧ��ΪС�ڴ��������ȵ�һ��ֵ
	const float octreeResolution = 0.01;     // �˲����ֱ���
	bool doVoxelGridDownDownSampling = true; // �Ƿ���������²�����ÿ��������ֻ������������һ���㣩
	const unsigned int iFrameRate = 100;     // ��ֱ���ѹ������
	bool doColorEncoding = false;             // �Ƿ�Բ�ɫ����ѹ��
	const unsigned char colorBitResolution = 8;// ����ÿһ����ɫ�ɷֱ������ռ��λ��

	// 2����ʼ��ѹ������,�������
	pcl::io::OctreePointCloudCompression<pcl::PointXYZ>* PointCloudEncoder;
	PointCloudEncoder = new pcl::io::OctreePointCloudCompression<pcl::PointXYZ>(compressionProfile,
		showStatistics,
		pointResolution,
		octreeResolution,
		doVoxelGridDownDownSampling,
		iFrameRate,
		doColorEncoding,
		colorBitResolution);

	// �洢ѹ�����Ƶ��ֽ�������
	std::stringstream compressedData; 
	// 3��ѹ������
	PointCloudEncoder->encodePointCloud(sourceCloud->makeShared(), compressedData); // ѹ������
	compressedData.write("compressed.bin", sizeof(compressedData));

	// 4����ѹ������
	PointCloudEncoder->decodePointCloud(compressedData, cloudOut); // ��ѹ������
}

#include <pcl/octree/octree.h>
//PCL �˲�����Ӧ�á����ռ�仯���
void spatialChangeDetection()
{
	/*
		�˲���������ʵ�ֶ���������֮��ı仯��⣬��Щ������ڳߴ硢�ֱ��ʡ��ܶȡ���˳��ȷ�����������
		ͨ���ݹ�رȽϰ˲��������ṹ�����Լ������˲����������������֮�������������Ŀռ�仯
	*/

	//�˲����ֱ��� �����صĴ�С
	float resolution = 0.1f;
	//��ʼ���ռ���Ʊ仯������
	pcl::octree::OctreePointCloudChangeDetector<pcl::PointXYZ> octree(resolution);//����ԽС��Լ��Խ����
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudA(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloudA);
	//��ӵ��Ƶ��˲����������˲���
	octree.setInputCloud(cloudA);
	octree.addPointsFromInputCloud();
	//�����˲������棬����cloudA��Ӧ�İ˲��������ڴ���
	octree.switchBuffers();//pcl�˲���˫���弼��������ʱ������Ƹ�Ч�Ĵ���������
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloudB(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::io::loadPCDFile<pcl::PointXYZ>("../bunny.pcd", *cloudB);
	//���cloudB���˲���
	octree.setInputCloud(cloudB);
	octree.addPointsFromInputCloud();
	vector<int>newPointIdxVector;
	//��ȡǰһcloudA��Ӧ�İ˲�����cloudB��Ӧ�˲�����û�е�����--�����������ڰ˲����ṹ�£�����С���ص�λΪresolution�£����ƵĲ���
	octree.getPointIndicesFromNewVoxels(newPointIdxVector);//��һ����������cloudB������һ�β�����cloudA
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_change(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::copyPointCloud(*cloudB, newPointIdxVector, *cloud_change);
	//��ӡ�����
	cout << "Output from getPointIndicesFromNewVoxels:" << endl;
	for (size_t i = 0; i<newPointIdxVector.size(); ++i)
		cout << i << "# Index:" << newPointIdxVector[i]
		<< "  Point:" << cloudB->points[newPointIdxVector[i]].x << " "
		<< cloudB->points[newPointIdxVector[i]].y << " "
		<< cloudB->points[newPointIdxVector[i]].z << endl;

	// ��ʼ�����ƿ��ӻ�����
	boost::shared_ptr<pcl::visualization::PCLVisualizer>viewer(new pcl::visualization::PCLVisualizer("��ʾ����"));
	viewer->setBackgroundColor(0, 0, 0);
	// ��cloudA������ɫ���ӻ� (��ɫ).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloudA_color(cloudA, 255, 255, 255);
	viewer->addPointCloud<pcl::PointXYZ>(cloudA, cloudA_color, "cloudA");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudA");
	// ��cloudB������ɫ���ӻ� (��ɫ).
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>cloudB_color(cloudB, 0, 255, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloudB, cloudB_color, "cloudB");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloudB");
	// �Լ������ı仯������ӻ�.
	pcl::visualization::PointCloudColorHandlerCustom<pcl::PointXYZ>change_color(cloud_change, 255, 0, 0);
	viewer->addPointCloud<pcl::PointXYZ>(cloud_change, change_color, "cloud_change");
	viewer->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud_change");
	// �ȴ�ֱ�����ӻ����ڹر�
	while (!viewer->wasStopped())
	{
		viewer->spinOnce(100);
		boost::this_thread::sleep(boost::posix_time::microseconds(1000));
	}
}

//PCL ��˲�������������
using AlignedPointTVector = std::vector<pcl::PointXYZ, Eigen::aligned_allocator<pcl::PointXYZ>>;
//�����ڴ����Eigen::MatrixXf;����������㵽�����ĵ�
void computeVoxelCenter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud)
{
	//�������ظ������
	AlignedPointTVector voxel_center;
	voxel_center.clear();      //��ʼ��
	float resolution = 0.25f;   //�������ظ�ı߳�
	pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	int treeDepth_ = octree.getTreeDepth();
	cout << "the depth of treeDepth_ is : " << treeDepth_ << endl;
	int occupiedVoxelCenters = octree.getOccupiedVoxelCenters(voxel_center);//��˲�������������
	cout << "the number of occupiedVoxelCenters are : " << occupiedVoxelCenters << endl;//������ظ����ĸ���
	cout << "the number of voxel are : " << voxel_center.size() << endl;   //����������ĵĸ���
	cout << "voxel0 is : " << voxel_center[0].x << voxel_center[0].y << endl;
}

//PCL ���ڰ˲����������˲�--ʹ�ð˲������������ĵ�������ԭʼ���ƣ������ĵ���ܲ��ǵ����еĵ㣩
using AlignedPointT = Eigen::aligned_allocator<pcl::PointXYZ>;
void octreeVoxelFilter(const pcl::PointCloud<pcl::PointXYZ>::ConstPtr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octree_filter_cloud)
{
	/*
		pclʵ�ֻ������ص��˲���ʽ�Ե��ƽ����²���
		�˲���ͬ��Ҳ�ǽ������أ����ͬ��Ҳ���ԶԵ��ƽ����²���
		��򵥷������ð˲�������������������ÿһ�������ڵĵ㣬ʵ�ֵ����²���
		�˷���ApproximateVoxelGrid������ͬ�����������ĵ���������ڵĵ�
		Ψһ����ApproximateVoxelGrid���������������صĳ���ߣ����˲���ֻ���ǹ��������������
		�Ľ����þ���������������ĵ��������������ĵ㣬�Ӷ�ʹ���²���֮��ĵ㻹��ԭʼ�����е�����
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
	std::cout << "�������ĵ��˲�����Ƹ���Ϊ��" << voxel_centers.size() << std::endl;
}

#include <pcl/kdtree/kdtree_flann.h>
//PCL ���ڰ˲����������˲�--ʹ�ð˲������������ĵ������ڵ�������ԭʼ���ƣ����ջ�ȡ�ĵ��ƶ�����ԭʼ���������еĵ㣩
void octreeVoxelFilterCenterKNN(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr& octknn_filter_cloud)
{
	float m_resolution = 0.002;
	pcl::octree::OctreePointCloud<pcl::PointXYZ> octree(m_resolution);
	octree.setInputCloud(cloud);
	octree.addPointsFromInputCloud();
	std::vector<pcl::PointXYZ, AlignedPointT> voxel_centers;
	octree.getOccupiedVoxelCenters(voxel_centers);
	//-----------K���������------------
	//�����²����Ľ����ѡ��������������ĵ���Ϊ���յ��²�����
	pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
	kdtree.setInputCloud(cloud);
	pcl::PointIndicesPtr inds = boost::shared_ptr<pcl::PointIndices>(new pcl::PointIndices());//������������ڽ�����ȡ���������±�����
	for (size_t i = 0; i < voxel_centers.size(); ++i) 
	{
		pcl::PointXYZ searchPoint;
		searchPoint.x = voxel_centers[i].x;
		searchPoint.y = voxel_centers[i].y;
		searchPoint.z = voxel_centers[i].z;
		int K = 1;//���������
		std::vector<int> pointIdxNKNSearch(K);
		std::vector<float> pointNKNSquaredDistance(K);
		if (kdtree.nearestKSearch(searchPoint, K, pointIdxNKNSearch, pointNKNSquaredDistance) > 0) 
		{
			inds->indices.push_back(pointIdxNKNSearch[0]);
		}
	}
	pcl::copyPointCloud(*cloud, inds->indices, *octknn_filter_cloud);
	std::cout << "������������ڵ��˲�����Ƹ���Ϊ��" << octknn_filter_cloud->points.size() << std::endl;
};

//------------------------------------------------------------------------------------------------//

//��ʾ����--˫����
void visualizeOCCloud(pcl::PointCloud<pcl::PointXYZ>::Ptr& cloud,
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