#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>

#include "Eigen/Eigen"
#include "Eigen/Core"
#include "Eigen/Dense"

class DatasetsEigen{
public:
    DatasetsEigen();
    DatasetsEigen(char *);
    DatasetsEigen(int, int);
    int get_size();
    int get_feature_size();
    Eigen::MatrixXd get_feature();
    Eigen::VectorXd get_target();
    void set_feature(Eigen::MatrixXd);
    void set_target(Eigen::VectorXd);
private:
    Eigen::MatrixXd feature_mtx;
    Eigen::VectorXd target_vect;
    int feature_size;
    int dataset_size;
};