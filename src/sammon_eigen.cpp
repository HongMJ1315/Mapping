#include "sammon_eigen.h"
#include <Eigen/Dense>
#include <algorithm>

SammonEigen::SammonEigen(DatasetsEigen &data,
    int new_dim, int max_iter,
    double lr, double alpha, double esp){
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

    double sum_all = dp.sum();            
    double sum_upper = sum_all * 0.5;    
    c = 1.0 / sum_upper;

    ori_mtx = Eigen::MatrixXd::Random(dataset_size, data.get_feature_size());
    // ori_mtx = X;
}

void SammonEigen::train(){
    if(this->cnt >= this->max_iter) return;
    const int N = dataset_size;
    Eigen::MatrixXd &Y = ori_mtx; // (N×k)

    Eigen::VectorXd r = Y.rowwise().squaredNorm();               // (N×1)
    Eigen::ArrayXXd D2 = (r.replicate(1, N) + r.transpose().replicate(N, 1)
        - 2.0 * (Y * Y.transpose())).array();
    D2 = D2.max(0.0);
    Eigen::ArrayXXd D = D2.sqrt().max(esp);

    Eigen::ArrayXXd dpArr = dp.array();                         // (N×N)
    Eigen::ArrayXXd valid = (dpArr >= esp).cast<double>();      // 0/1 mask

    Eigen::ArrayXXd numer = dpArr - D;
    Eigen::ArrayXXd denom = dpArr * D;
    Eigen::ArrayXXd Farr = valid.select(numer / denom, 0.0);

    Eigen::MatrixXd F = Farr.matrix();  F.diagonal().setZero();

    Eigen::VectorXd s = F.rowwise().sum();                     // (N×1)

    Eigen::MatrixXd G = s.asDiagonal() * Y - F * Y;            // (N×k)

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

    cnt++;
    if(cnt % 100 == 0) lr *= alpha;
}

void SammonEigen::get_new_data(DatasetsEigen &out){
    out.set_feature(ori_mtx);
}
