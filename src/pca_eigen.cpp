#include "pca_eigen.h"


void pca_eigen(DatasetsEigen &hd, DatasetsEigen &td){
    int n = hd.get_size();
    int orig_dim = hd.get_feature_size();
    int k = td.get_feature_size();

    Eigen::MatrixXd X = hd.get_feature();         // n×orig_dim
    Eigen::VectorXd T = hd.get_target();         // n

    Eigen::RowVectorXd mu = X.colwise().mean(); // 1×orig_dim

    Eigen::MatrixXd Xc = X.rowwise() - mu;

    Eigen::MatrixXd C = (Xc.transpose() * Xc) / double(n - 1);

    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);
    if(solver.info() != Eigen::Success){
        std::cerr << "PCA Eigen decomposition failed\n";
        return;
    }
    Eigen::MatrixXd eigvecs = solver.eigenvectors();
    
    Eigen::MatrixXd V_k = eigvecs.rightCols(k);

    Eigen::MatrixXd Y = Xc * V_k;

    td = DatasetsEigen(n, k);
    std::cout << n << " " << Y.rows() << std::endl;
    std::cout << k << " " << Y.cols() << std::endl;
    td.set_feature(Y);
    td.set_target(T);
}
