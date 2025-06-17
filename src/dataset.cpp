#include "dataset.h"

Datasets::Datasets(){}

Datasets::Datasets(int data_size, int feature_size){
    this->feature_size = feature_size;
    this->datasets.resize(data_size);
    for(auto &i : this->datasets){
        i.feature.resize(feature_size);
    }
}

Datasets::Datasets(char *filename){

    std::ifstream file(filename);
    if(file.fail()){
        std::cout << "ERROR" << std::endl;
        return;
    }
    int dataset_size = 0;
    char garbage;
    file >> dataset_size; file >> garbage;

    file >> this->feature_size; file >> garbage;
    this->feature_size--;
    std::cout << dataset_size << " " << this->feature_size << std::endl;
    for(int i = 0; i < dataset_size; i++){
        Data data;
        data.feature.resize(feature_size);
        for(int j = 0; j < feature_size; j++){
            if(j == 0){
                file >> data.feature[j];
            }
            else{
                file >> garbage;
                file >> data.feature[j];
            }
        }
        file >> garbage;
        file >> data.target;

        this->datasets.push_back(data);

    }
}

int Datasets::get_size(){
    return (int) this->datasets.size();
}

int Datasets::get_feature_size(){
    return this->feature_size;
}

Datasets::Data Datasets::get_data(int idx){
    return this->datasets[idx];
}

void Datasets::set_dataset_feature(int idx, std::vector<double> feature){
    // std::cout << feature.size() << " " << (this->datasets[idx]).feature.size() << std::endl;
    if(feature.size() == (this->datasets[idx]).feature.size())
        (this->datasets[idx]).feature = feature;
    else{
        std::cout << "dim error at set dataset feature" << std::endl;
    }
}

void Datasets::set_dataset_target(int idx, int target){
    this->datasets[idx].target = target;
}