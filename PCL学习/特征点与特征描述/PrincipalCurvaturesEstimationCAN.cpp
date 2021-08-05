#include "PrincipalCurvaturesEstimationCAN.h"
#include <Eigen/Dense>

void PrincipalCurvaturesEstimationCAN::compute(PointCloudOut& output) 
{
	output.resize(this->input_->size());
	std::vector<int> neighbor_ids;
	std::vector<float> sq_distances;
	Eigen::Vector3f x = (Eigen::Vector3f() << 1, 0, 0).finished();
	Eigen::Vector3f z = (Eigen::Vector3f() << 0, 0, 1).finished();
	Eigen::Vector3f N, p, q, Q, n, mu, max_eigenvector;
	Eigen::Matrix3f R;
	Eigen::MatrixXf M;
	Eigen::VectorXf r;
	Eigen::Matrix2f W;
	Eigen::EigenSolver<Eigen::Matrix2f> eigen_solver;
	Eigen::Vector2f::Index max_ind;
	Eigen::Vector2f eigenvalues;
	float nxy, k, theta;
	int jn, j, row_ind;
	for (int i = 0; i < this->input_->size(); i++) 
	{
		p = Eigen::Vector3f::Map(this->input_.get()->points[i].data);
		N = Eigen::Vector3f::Map(this->normals_.get()->points[i].data_n);
		R = (2 * ((N + z) * (N + z).transpose()) / ((N + z).transpose() * (N + z))) - Eigen::Matrix3f::Identity(); 
		if (this->tree_->radiusSearch(this->input_.get()->points[i], this->search_radius_, neighbor_ids, sq_distances) > 2) 
		{
			M.resize(neighbor_ids.size() - 1, 3);
			r.resize(neighbor_ids.size() - 1); //减去1是因为我们跳过了搜索点
			row_ind = 0;
			for (j = 1; j < neighbor_ids.size(); j++) {
				jn = neighbor_ids[j];
				q = R * (Eigen::Vector3f::Map(this->input_.get()->points[jn].data) - p);
				Q << q.head<2>(), 0;
				n = R * Eigen::Vector3f::Map(this->normals_.get()->points[jn].data_n);
				nxy = (q(0) * n(0) + q(1) * n(1)) / sqrt(pow(q(0), 2) + pow(q(1), 2));
				k = -nxy / sqrt((pow(nxy, 2) + pow(n(2), 2)) * (pow(q(0), 2) + pow(q(1), 2)));
				theta = acos(x.dot(Q) / (x.norm() * Q.norm()));
				if (!isnan(k)) {
					M(row_ind, 0) = pow(cos(theta), 2);
					M(row_ind, 1) = 2 * cos(theta) * sin(theta);
					M(row_ind, 2) = pow(sin(theta), 2);
					r(row_ind) = k;
					++row_ind;
				}
				else {
					M.resize(M.rows() - 1, 3);
					r.resize(r.rows() - 1);
				}
			}
			mu = M.colPivHouseholderQr().solve(r);
			if (!isnan(mu(0))) {
				W << mu(0), mu(1), mu(1), mu(2);
				eigen_solver.compute(W, true);
				eigenvalues = eigen_solver.eigenvalues().real();
				output[i].pc1 = eigenvalues.maxCoeff(&max_ind);
				output[i].pc2 = eigenvalues.minCoeff();
				max_eigenvector = R.transpose() * (Eigen::Vector3f() << eigen_solver.eigenvectors().col(max_ind).real(), 0).finished();
				output[i].principal_curvature_x = max_eigenvector(0);
				output[i].principal_curvature_y = max_eigenvector(1);
				output[i].principal_curvature_z = max_eigenvector(2);
			}
		}
	}
}


