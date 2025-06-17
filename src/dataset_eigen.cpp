#include "dataset_eigen.h"

DatasetsEigen::DatasetsEigen(){}

DatasetsEigen::DatasetsEigen(int data_size, int feature_size){
    this->feature_size = feature_size;
    feature_mtx = Eigen::MatrixXd(data_size, feature_size);
    target_vect = Eigen::VectorXd(data_size);
}

DatasetsEigen::DatasetsEigen(char *filename){

    std::ifstream file(filename);
    if(file.fail()){
        std::cout << "ERROR" << std::endl;
        return;
    }
    this->dataset_size = 0;
    char garbage;
    file >> dataset_size; file >> garbage;

    file >> this->feature_size; file >> garbage;
    this->feature_size--;
    std::cout << dataset_size << " " << this->feature_size << std::endl;

    feature_mtx = Eigen::MatrixXd(dataset_size, feature_size);
    target_vect = Eigen::VectorXd(dataset_size);


    for(int i = 0; i < dataset_size; i++){
        for(int j = 0; j < feature_size; j++){
            if(j == 0){
                file >> feature_mtx(i, j);
            }
            else{
                file >> garbage;
                file >> feature_mtx(i, j);
            }
        }
        file >> garbage;
        file >> target_vect(i);
    }
}

int DatasetsEigen::get_size(){
    return dataset_size;
}

int DatasetsEigen::get_feature_size(){
    return feature_size;
}

Eigen::MatrixXd DatasetsEigen::get_feature(){
    return this->feature_mtx;
}

Eigen::VectorXd DatasetsEigen::get_target(){
    return this->target_vect;
}

void DatasetsEigen::set_feature(Eigen::MatrixXd feature){
    if(feature.rows() == this->feature_mtx.rows() &&
        feature.cols() == this->feature_mtx.cols())
        this->feature_mtx = feature;
    std::cout << "Set feature Error" << std::endl;
}

void DatasetsEigen::set_target(Eigen::VectorXd target){
    if(target.size() == this->target_vect.size()){
        this->target_vect = target;
    }
    std::cout << "Set target Error" << std::endl;
}