#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "pca_eigen.h"

class SammonEigen{
private:
    int dataset_size;
    int feature_size;
    int new_dim;
    int max_iter;
    int cnt;
    double lr;
    double alpha;
    double esp;
    double c;
    Eigen::MatrixXd dp;
    Eigen::MatrixXd ori_mtx;

    double get_dis(std::vector<double>, std::vector<double>);
    double get_dis(Eigen::VectorXd, Eigen::VectorXd);

public:
    SammonEigen(DatasetsEigen &, int, int, double, double, double);
    void train();
    void get_new_data(DatasetsEigen &);
};