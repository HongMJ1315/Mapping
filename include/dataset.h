#include <string>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <vector>


class Datasets{
public:
    struct Data{
        std::vector<double> feature;
        int target;
    };
    Datasets();
    Datasets(char *);
    Datasets(int, int);
    int get_size();
    int get_feature_size();
    Data get_data(int);
    void set_dataset_feature(int, std::vector<double>);
    void set_dataset_target(int, int);
private:
    std::vector<Data> datasets;
    int feature_size;
};