#include "sammon.h"

double Sammon::get_dis(std::vector<double> vi, std::vector<double> vj){
    if(vi.size() != vj.size()){
        std::cout << "DIM ERROR" << std::endl;
    }
    double len = 0;
    for(int i = 0; i < vi.size(); i++){
        len += (vi[i] - vj[i]) * (vi[i] - vj[i]);
    }
    return sqrt(len);
}

double Sammon::get_dis(Eigen::VectorXd vi, Eigen::VectorXd vj){
    Eigen::VectorXd diff = vi - vj;
    double len = diff.dot(diff);
    return sqrt(len);
}

Sammon::Sammon(Datasets &data, int new_dim, int max_iter, double lr, double alpha, double esp){
    this->dataset_size = data.get_size();
    this->feature_size = data.get_feature_size();
    this->new_dim = new_dim;
    this->max_iter = max_iter;
    this->lr = lr;
    this->alpha = alpha;
    this->esp = esp;
    this->cnt = 0;
    int dataset_size = data.get_size();
    dp = Eigen::MatrixXd(dataset_size, dataset_size);
    for(int i = 0; i < dataset_size; i++){
        for(int j = 0; j < dataset_size; j++){
            if(i < j){
                Datasets::Data di = data.get_data(i);
                Datasets::Data dj = data.get_data(j);

                std::vector<double> vi = di.feature;
                std::vector<double> vj = dj.feature;


                this->dp(j, i) = this->dp(i, j) = get_dis(vi, vj);
            }
            if(i == j) this->dp(i, i) = 0;
        }
    }

    double sum_dp = 0;
    for(int i = 0; i < dataset_size - 1; i++){
        for(int j = i + 1; j < dataset_size; j++){
            sum_dp += this->dp(i, j);
        }
    }
    this->c = 1.f / sum_dp;

    /*
    this->ori_mtx = Eigen::MatrixXd(dataset_size, data.get_feature_size());
    for(int i = 0; i < dataset_size; i++){
        Datasets::Data d = data.get_data(i);
        for(int j = 0; j < data.get_feature_size(); j++){
            ori_mtx(i, j) = d.feature[j];
        }
    }
    // */

    this->ori_mtx = Eigen::MatrixXd::Random(dataset_size, data.get_feature_size());

}


/*
void Sammon::train(){
    if(cnt >= max_iter) return;
    std::vector<Eigen::VectorXd> arr_grad(dataset_size, Eigen::VectorXd::Zero(feature_size));

    for(int i = 0; i < dataset_size - 1; i++){
        for(int j = i + 1; j < dataset_size; j++){
            double orig = dp(i, j);
            if(orig < esp)
                continue;

            auto yi = ori_mtx.row(i).transpose().eval();
            auto yj = ori_mtx.row(j).transpose().eval();

            double dis = get_dis(yi, yj);
            if(dis < esp){
                dis = esp;
            }

            double f = (this->dp(i, j) - dis) / (this->dp(i, j) * dis);
            auto diff = (yi - yj).eval();

            arr_grad[i] += f * diff;
            arr_grad[j] -= f * diff;
        }
    }

    for(int i = 0; i < dataset_size; i++){
        ori_mtx.row(i) += 2 * c * lr * (arr_grad[i].transpose());
    }

    lr *= alpha;
    cnt++;
    std::cout << cnt << std::endl;
}
// */

void Sammon::train(){
    if(cnt >= max_iter) return;

    Eigen::MatrixXd &Y = ori_mtx;        // N×k
    const int N = dataset_size;
    const double eps = esp;

    Eigen::VectorXd r = Y.rowwise().squaredNorm();          // (N×1)
    Eigen::ArrayXXd D2 = (r.replicate(1, N)
        + r.transpose().replicate(N, 1)
        - 2.0 * (Y * Y.transpose())).array();
    D2 = D2.max(0.0);                                        
    Eigen::ArrayXXd Darr = D2.sqrt().max(eps);               

    Eigen::ArrayXXd dpArr = dp.array();                     // (N×N)
    Eigen::ArrayXXd valid = (dpArr >= eps).cast<double>();  

    Eigen::ArrayXXd numer = dpArr - Darr;                 
    Eigen::ArrayXXd denom = dpArr * Darr;            
    Eigen::ArrayXXd Farr = valid.select(numer / denom, 0.0);

    Eigen::MatrixXd F = Farr.matrix();
    F.diagonal().setZero();

    Eigen::VectorXd s = F.rowwise().sum();                   // (N×1)

    Eigen::MatrixXd G = s.asDiagonal() * Y - F * Y;          // (N×k)

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

    ++cnt;
    if(cnt % 50 == 0)
        lr *= alpha;
}

void Sammon::get_new_data(Datasets &old){
    for(int i = 0; i < dataset_size; i++){
        std::vector<double> feature(feature_size);
        Datasets::Data d = old.get_data(i);
        for(int j = 0; j < feature_size; j++){
            feature[j] = ori_mtx(i, j);
            // std::cout << feature[j] << " ";
        }
        // std::cout << std::endl;
        old.set_dataset_feature(i, feature);
    }
}