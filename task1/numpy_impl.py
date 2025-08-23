import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. 准备数据集 (与 PyTorch 版本几乎一样) ---
X, y = make_moons(n_samples=1000, noise=0.2, random_state=42)
y = y.reshape(-1, 1) # 转换为列向量
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# --- 2. 定义激活函数及其导数 ---
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    return (x > 0).astype(float)


# --- 3. 初始化模型参数 ---
input_neurons = 2
hidden_neurons1 = 16
hidden_neurons2 = 16
output_neurons = 1

# 初始化权重和偏置
# np.random.randn 生成标准正态分布的随机数
w1 = np.random.randn(input_neurons, hidden_neurons1) * 0.01
b1 = np.zeros((1, hidden_neurons1))
w2 = np.random.randn(hidden_neurons1, hidden_neurons2) * 0.01
b2 = np.zeros((1, hidden_neurons2))
w3 = np.random.randn(hidden_neurons2, output_neurons) * 0.01
b3 = np.zeros((1, output_neurons))


# --- 4. 训练模型 ---
epochs = 100
learning_rate = 0.01
losses = []
accuracies = []

for epoch in range(epochs):
    # --- 前向传播 (Forward Propagation) ---
    # Layer 1
    z1 = np.dot(X_train, w1) + b1
    a1 = relu(z1)
    # Layer 2
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    # Layer 3 (Output)
    z3 = np.dot(a2, w3) + b3
    a3 = sigmoid(z3) # a3 就是我们的预测值 y_pred

    # --- 计算损失 (Binary Cross-Entropy Loss) ---
    m = y_train.shape[0]
    loss = -1/m * np.sum(y_train * np.log(a3) + (1 - y_train) * np.log(1 - a3))
    losses.append(loss)

    # --- 反向传播 (Backward Propagation) - 最核心的部分! ---
    # 从输出层开始，反向计算梯度 (链式法则)
    # Layer 3
    dz3 = a3 - y_train
    dw3 = 1/m * np.dot(a2.T, dz3)
    db3 = 1/m * np.sum(dz3, axis=0, keepdims=True)
    # Layer 2
    dz2 = np.dot(dz3, w3.T) * relu_derivative(a1) # 注意这里是a1, 因为ReLU的导数依赖于它的输入
    dw2 = 1/m * np.dot(a1.T, dz2)
    db2 = 1/m * np.sum(dz2, axis=0, keepdims=True)
    # Layer 1
    dz1 = np.dot(dz2, w2.T) * relu_derivative(X_train) # 同上，是X_train
    dw1 = 1/m * np.dot(X_train.T, dz1)
    db1 = 1/m * np.sum(dz1, axis=0, keepdims=True)

    # --- 参数更新 ---
    w1 -= learning_rate * dw1
    b1 -= learning_rate * db1
    w2 -= learning_rate * dw2
    b2 -= learning_rate * db2
    w3 -= learning_rate * dw3
    b3 -= learning_rate * db3

    # 计算并记录准确率
    predicted = (a3 >= 0.5).astype(int)
    accuracy = accuracy_score(y_train, predicted)
    accuracies.append(accuracy)

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss:.4f}, Accuracy: {accuracy:.4f}')

# --- 5. 绘制决策边界热力图 (与 PyTorch 版本逻辑类似) ---
def plot_decision_boundary_numpy(X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # 手动进行前向传播
    z1 = np.dot(grid_points, w1) + b1
    a1 = relu(z1)
    z2 = np.dot(a1, w2) + b2
    a2 = relu(z2)
    z3 = np.dot(a2, w3) + b3
    Z = sigmoid(z3)
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdYlBu, edgecolors='k', alpha=0.7)
    plt.title("Decision Boundary Heatmap (NumPy)")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("decision_boundary_numpy.png")
    print("NumPy决策边界热力图已保存为 decision_boundary_numpy.png")

plot_decision_boundary_numpy(X, y)