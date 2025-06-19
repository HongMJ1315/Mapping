// sammon_eigen.cpp
#include "sammon_eigen.h"
#include <Eigen/Dense>
#include <algorithm>

// Constructor: compute high-d distances dp and initialize low-d matrix
SammonEigen::SammonEigen(DatasetsEigen &data,
    int new_dim, int max_iter,
    double lr, double alpha, double esp){
    // build dp matrix
    this->dataset_size = data.get_size();
    this->feature_size = data.get_feature_size();
    this->new_dim = new_dim;
    this->max_iter = max_iter;
    this->lr = lr;
    this->alpha = alpha;
    this->esp = esp;
    this->cnt = 0;
    this->dp = Eigen::MatrixXd(dataset_size, dataset_size);
    Eigen::MatrixXd X = data.get_feature(); // (N×orig_dim)
    for(int i = 0; i < dataset_size; i++){
        for(int j = 0; j < dataset_size; j++){
            if(i < j){
                double d = sqrt((X.row(i) - X.row(j)).dot((X.row(i) - X.row(j))));
                this->dp(i, j) = this->dp(j, i) = d;
            }
            else{
                this->dp(i, i) = 0;
            }
        }
    }
    // std::cout << dp << std::endl;
    // normalization constant
    // normalization constant: since dp.diagonal()==0, 
    // sum(dp) = 2 * sum_{i<j} dp(i,j)
    double sum_all = dp.sum();            // sum of all entries
    double sum_upper = sum_all * 0.5;     // half of that is the sum over i<j
    c = 1.0 / sum_upper;

    // initialize low-d coordinates randomly
    ori_mtx = Eigen::MatrixXd::Random(dataset_size, data.get_feature_size());
    // ori_mtx = X;
}

// Perform one iteration of Sammon mapping
void SammonEigen::train(){
    if(this->cnt >= this->max_iter) return;
    const int N = dataset_size;
    Eigen::MatrixXd &Y = ori_mtx; // (N×k)

    // 1) Compute low-d distances
    Eigen::VectorXd r = Y.rowwise().squaredNorm();               // (N×1)
    Eigen::ArrayXXd D2 = (r.replicate(1, N) + r.transpose().replicate(N, 1)
        - 2.0 * (Y * Y.transpose())).array();
    D2 = D2.max(0.0);
    Eigen::ArrayXXd D = D2.sqrt().max(esp);

    // 2) Mask high-d distances
    Eigen::ArrayXXd dpArr = dp.array();                         // (N×N)
    Eigen::ArrayXXd valid = (dpArr >= esp).cast<double>();      // 0/1 mask

    // 3) Compute weighted differences
    Eigen::ArrayXXd numer = dpArr - D;
    Eigen::ArrayXXd denom = dpArr * D;
    Eigen::ArrayXXd Farr = valid.select(numer / denom, 0.0);

    // 4) Convert to Matrix and zero diagonal
    Eigen::MatrixXd F = Farr.matrix();  F.diagonal().setZero();

    // 5) Row-sums
    Eigen::VectorXd s = F.rowwise().sum();                     // (N×1)

    // 6) Gradient
    Eigen::MatrixXd G = s.asDiagonal() * Y - F * Y;            // (N×k)

    // 7) Update
    Y.noalias() += (2.0 * c * lr) * G;

    double stress = 0.0;
    for(int i = 0; i < N - 1; ++i){
        for(int j = i + 1; j < N; ++j){
            double orig = dp(i, j);
            if(orig < esp) continue;
            double low = (Y.row(i) - Y.row(j)).norm();
            stress += (orig - low) * (orig - low) / orig;
        }
    }

    std::cout << "Iter = " << cnt << " LR = " << lr << " ERR = " << stress << std::endl;

    // std::cout << Y.row(5) << std::endl;
    // 8) Decay and count
    cnt++;
    if(cnt % 100 == 0) lr *= alpha;
}

// Write low-d results back into a DatasetsEigen
void SammonEigen::get_new_data(DatasetsEigen &out){
    out.set_feature(ori_mtx);
}
