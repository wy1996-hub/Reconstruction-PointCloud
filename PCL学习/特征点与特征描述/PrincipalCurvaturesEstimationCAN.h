#pragma once
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/search/search.h>

class PrincipalCurvaturesEstimationCAN 
{
public:
	using PointInT = pcl::PointXYZ;
	using PointNT = pcl::Normal;
	using PointOutT = pcl::PrincipalCurvatures;
	using PointCloud = pcl::PointCloud<PointInT>;
	using PointCloudPtr = typename PointCloud::Ptr;
	using PointCloudConstPtr = typename PointCloud::ConstPtr;
	using PointCloudN = pcl::PointCloud<PointNT>;
	using PointCloudNPtr = typename PointCloudN::Ptr;
	using PointCloudNConstPtr = typename PointCloudN::ConstPtr;
	using PointCloudOut = pcl::PointCloud<PointOutT>;
	using KdTree = pcl::search::Search<PointInT>;
	using KdTreePtr = typename KdTree::Ptr;

	void setInputCloud(const PointCloudConstPtr& cloud) 
	{
		this->input_ = cloud;
	}
	void setInputNormals(const PointCloudNConstPtr& normals) 
	{
		this->normals_ = normals;
	}
	void setSearchMethod(const KdTreePtr& tree) 
	{
		this->tree_ = tree;
	}
	void setRadiusSearch(double radius) 
	{
		this->search_radius_ = radius;
	}
	void compute(PointCloudOut& output);

protected:
	PointCloudConstPtr input_;
	PointCloudNConstPtr normals_;
	KdTreePtr tree_;
	double search_radius_;
};



