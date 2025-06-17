#include "pca_eigen.h"


void pca_eigen(DatasetsEigen &hd, DatasetsEigen &td){
    // 1. 读取原始维度和样本数
    int n = hd.get_size();
    int orig_dim = hd.get_feature_size();
    // PCA 后要降到的维度，从 td 的构造时 feature_size 获取
    int k = td.get_feature_size();

    // 2. 原始特征矩阵 (n × orig_dim) 和标签向量 (n)
    //    这里假设 feature_mtx, target_vect 是 public 或者有 getter
    Eigen::MatrixXd X = hd.get_feature();         // n×orig_dim
    Eigen::VectorXd T = hd.get_target();         // n

    // 3. 去中心化：先计算每列均值 μ (1 × orig_dim)
    Eigen::RowVectorXd mu = X.colwise().mean(); // 1×orig_dim

    //    然后 Xc = X − 1·μ （每行都减去 mu）
    Eigen::MatrixXd Xc = X.rowwise() - mu;

    // 4. 协方差矩阵 C = (Xcᵀ Xc) / (n−1)  （orig_dim × orig_dim）
    Eigen::MatrixXd C = (Xc.transpose() * Xc) / double(n - 1);

    // 5. 特征分解：C v = λ v
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(C);
    if(solver.info() != Eigen::Success){
        std::cerr << "PCA Eigen decomposition failed\n";
        return;
    }
    Eigen::MatrixXd eigvecs = solver.eigenvectors();  // 每列是一个特征向量

    // 6. 取出最大的 k 个特征向量 (orig_dim × k)
    Eigen::MatrixXd V_k = eigvecs.rightCols(k);

    // 7. 投影：Y = Xc * V_k  →  (n × k)
    Eigen::MatrixXd Y = Xc * V_k;

    // 8. 把结果写回到 td：先重新构造 td，再赋值
    td = DatasetsEigen(n, k);
    td.set_feature(std::move(Y));
    td.set_target(std::move(T));
}
