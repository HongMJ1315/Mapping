#include "pca.h"

void pca(Datasets &hd, Datasets &td){

    auto datasets_size = hd.get_size();
    auto original_dim = hd.get_feature_size();
    std::vector<double> mu = std::vector<double>(original_dim, 0);
    for(int i = 0; i < datasets_size; i++){
        for(int j = 0; j < original_dim; j++){
            Datasets::Data data = hd.get_data(i);
            mu[j] += data.feature[j];
        }
    }

    for(int i = 0; i < original_dim; i++){
        mu[i] /= double(datasets_size);
    }

    Eigen::MatrixXd xc(datasets_size, original_dim);
    for(int i = 0; i < datasets_size; i++){
        std::vector<double> feature = hd.get_data(i).feature;
        for(int j = 0; j < original_dim; j++){
            feature[j] -= mu[j];
        }
        for(int j = 0; j < original_dim; j++)
            xc(i, j) = feature[j];
    }

    // std::cout << xc << std::endl;

    Eigen::MatrixXd c = (xc.transpose() * xc) / double(datasets_size - 1);
    // std::cout << c << std::endl;

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(c);
    if(solver.info() != Eigen::Success){
        std::cerr << "Eigen decomposition failed\n";
        return;
    }

    // 特征向量按特征值升序排列在 solver.eigenvectors() 的列中
    Eigen::MatrixXd eigvecs = solver.eigenvectors();

    // 假设 original_dim ≥ 3
    int k = td.get_feature_size();
    std::cout << k << std::endl;
    // 最后三列是最大特征值对应的向量
    Eigen::MatrixXd V_k = eigvecs.rightCols(k);      // original_dim × k
    Eigen::MatrixXd Y = xc * V_k;                    // datasets_size × k

    // 把投影结果写入 td
    for(int i = 0; i < datasets_size; i++){
        std::vector<double> proj(k);
        for(int j = 0; j < k; j++){
            proj[j] = Y(i, j);
        }
        td.set_dataset_feature(i, proj);
        Datasets::Data data = hd.get_data(i);
        td.set_dataset_target(i, data.target);
    }

    for(int i = 0; i < datasets_size; i++){
        Datasets::Data data = td.get_data(i);
        for(auto i : data.feature) std::cout << i << " ";
        std::cout << std::endl;;
    }
}