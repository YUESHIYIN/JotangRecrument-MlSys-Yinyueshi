#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <Eigen/Dense>

// 使用 Eigen 命名空间
using namespace Eigen;

// --- 1. 读取 CSV 数据的函数 ---
// 从文件中读取数据并加载到 Eigen 矩阵中
void load_csv(const std::string& path, MatrixXd& X, MatrixXd& y) {
    std::ifstream file(path);
    std::string line;
    std::vector<double> values;
    int rows = 0;
    while (getline(file, line)) {
        std::stringstream ss(line);
        std::string value_str;
        while (getline(ss, value_str, ',')) {
            values.push_back(stod(value_str));
        }
        rows++;
    }
    // 将 vector 转换为 Eigen 矩阵
    MatrixXd data = Map<Matrix<double, Dynamic, Dynamic, RowMajor>>(values.data(), rows, 3);
    X = data.leftCols(2);
    y = data.rightCols(1);
}


// --- 2. 定义激活函数及其导数 ---
MatrixXd relu(const MatrixXd& x) {
    return x.cwiseMax(0);
}

MatrixXd relu_derivative(const MatrixXd& x) {
    return (x.array() > 0).cast<double>();
}

MatrixXd sigmoid(const MatrixXd& x) {
    return 1.0 / (1.0 + (-x).array().exp());
}


// --- 3. 主函数 ---
int main() {
    // 加载数据
    MatrixXd X_train, y_train;
    load_csv("train_data.csv", X_train, y_train);

    // 初始化参数
    int input_neurons = 2;
    int hidden_neurons1 = 16;
    int hidden_neurons2 = 16;
    int output_neurons = 1;

    MatrixXd w1 = MatrixXd::Random(input_neurons, hidden_neurons1) * 0.01;
    MatrixXd b1 = MatrixXd::Zero(1, hidden_neurons1);
    MatrixXd w2 = MatrixXd::Random(hidden_neurons1, hidden_neurons2) * 0.01;
    MatrixXd b2 = MatrixXd::Zero(1, hidden_neurons2);
    MatrixXd w3 = MatrixXd::Random(hidden_neurons2, output_neurons) * 0.01;
    MatrixXd b3 = MatrixXd::Zero(1, output_neurons);

    // 训练
    int epochs = 100;
    double learning_rate = 0.01;
    long m = X_train.rows();

    std::cout << "开始 C++ (Eigen) 训练..." << std::endl;

    for (int epoch = 0; epoch < epochs; ++epoch) {
        // 前向传播
        MatrixXd z1 = X_train * w1 + b1.replicate(m, 1);
        MatrixXd a1 = relu(z1);
        MatrixXd z2 = a1 * w2 + b2.replicate(m, 1);
        MatrixXd a2 = relu(z2);
        MatrixXd z3 = a2 * w3 + b3.replicate(m, 1);
        MatrixXd a3 = sigmoid(z3);

        // 计算损失
        MatrixXd logprobs = (y_train.array() * a3.array().log() + (1 - y_train.array()) * (1 - a3.array()).log());
        double loss = -logprobs.sum() / m;

        // 反向传播 - 修正了导数计算部分
        MatrixXd dz3 = a3 - y_train;
        MatrixXd dw3 = (1.0/m) * (a2.transpose() * dz3);
        MatrixXd db3 = (1.0/m) * dz3.colwise().sum();

        // 修正：使用 z2 计算 relu 导数，而不是 a1
        MatrixXd dz2 = (dz3 * w3.transpose()).array() * relu_derivative(z2).array();
        MatrixXd dw2 = (1.0/m) * (a1.transpose() * dz2);
        MatrixXd db2 = (1.0/m) * dz2.colwise().sum();

        // 修正：使用 z1 计算 relu 导数，而不是 X_train
        MatrixXd dz1 = (dz2 * w2.transpose()).array() * relu_derivative(z1).array();
        MatrixXd dw1 = (1.0/m) * (X_train.transpose() * dz1);
        MatrixXd db1 = (1.0/m) * dz1.colwise().sum();
        
        // 参数更新
        w1 -= learning_rate * dw1;
        b1 -= learning_rate * db1;
        w2 -= learning_rate * dw2;
        b2 -= learning_rate * db2;
        w3 -= learning_rate * dw3;
        b3 -= learning_rate * db3;

        if ((epoch + 1) % 10 == 0) {
            // 计算准确率
            MatrixXd predicted = (a3.array() >= 0.5).cast<double>();
            double accuracy = ((predicted.array() == y_train.array()).cast<double>().sum()) / m;
            std::cout << "Epoch [" << epoch + 1 << "/" << epochs << "], Loss: " << loss << ", Accuracy: " << accuracy << std::endl;
        }
    }

    std::cout << "训练完成！" << std::endl;

    return 0;
}

//输入
//cd cpp_impl
//g++ -I /usr/include/eigen3 main.cpp -o main
//./main
//运行