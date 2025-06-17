#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "pca.h"

#include "Eigen/Eigen"
#include "Eigen/Core"
#include "Eigen/Dense"

class Sammon{
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
    Sammon(Datasets &, int, int, double, double, double);
    void train();
    Datasets get_new_data(Datasets&);
};