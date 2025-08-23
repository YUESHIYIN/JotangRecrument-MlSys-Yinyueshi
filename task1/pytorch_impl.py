import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# --- 1. 准备数据集 ---

# 生成数据集
# X 是特征 (坐标), y 是标签 (0 或 1)
# noise 参数控制了数据的混乱程度，可以调整它看看效果
X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# 将 numpy 数组转换为 PyTorch Tensors
X = torch.from_numpy(X.astype(np.float32))
y = torch.from_numpy(y.astype(np.float32)).view(-1, 1) # 转换为列向量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 可视化原始数据集 ---
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdYlBu, alpha=0.7)
plt.title("Original Dataset (make_moons)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.savefig("dataset_visualization.png") # 保存图像
print("数据集可视化图像已保存为 dataset_visualization.png")


# --- 2. 定义神经网络模型 ---
# 这是一个简单的多层感知机 (MLP)
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        # 定义网络层
        self.layers = nn.Sequential(
            nn.Linear(2, 16),      # 输入层 (2个特征) -> 第一个隐藏层 (16个神经元)
            nn.ReLU(),             # ReLU 激活函数
            nn.Linear(16, 16),     # 第二个隐藏层
            nn.ReLU(),
            nn.Linear(16, 1),      # 输出层 (1个神经元，用于二分类)
            nn.Sigmoid()           # Sigmoid 激活函数，将输出压缩到 0-1 之间，代表概率
        )

    def forward(self, x):
        return self.layers(x)

model = MLP()


# --- 3. 定义损失函数和优化器 ---
loss_function = nn.BCELoss() # 二元交叉熵损失，适用于二分类
optimizer = optim.Adam(model.parameters(), lr=0.01)


# --- 4. 训练模型 ---
epochs = 100
train_losses = []
train_accuracies = []

for epoch in range(epochs):
    model.train() # 设置为训练模式

    # 前向传播
    y_pred = model(X_train)
    loss = loss_function(y_pred, y_train)

    # 反向传播和优化
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 计算并记录训练集上的准确率
    with torch.no_grad():
        predicted = (y_pred >= 0.5).float()
        accuracy = accuracy_score(y_train, predicted)
        train_accuracies.append(accuracy)
        train_losses.append(loss.item())

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}')

# --- 5. 绘制损失和准确率曲线 ---
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.title("Training Loss Curve")
plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.subplot(1, 2, 2)
plt.plot(train_accuracies)
plt.title("Training Accuracy Curve")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("training_curves.png")
print("训练曲线图像已保存为 training_curves.png")


# --- 6. 绘制决策边界热力图 ---

def plot_decision_boundary(model, X, y):
    # 设置绘图范围，比数据集的边界稍大一些
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # 生成一个网格，覆盖整个绘图区域
    # h 是网格的精细度
    h = 0.01
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # 将网格点转换为 PyTorch Tensor，并用模型进行预测
    # ravel() 将网格 xx 和 yy 展平为一维数组
    # c_() 将它们合并成一个两列的矩阵，代表了空间中每个点的坐标
    grid_points = torch.from_numpy(np.c_[xx.ravel(), yy.ravel()]).float()
    
    model.eval() # 设置为评估模式
    with torch.no_grad():
        # Z 是模型对空间中每个点的预测概率
        Z = model(grid_points)
    
    # 将预测结果 Z 的形状变回网格的形状
    Z = Z.reshape(xx.shape)
    
    # 绘制热力图
    plt.figure(figsize=(8, 6))
    # contourf 会根据 Z 的值填充颜色，形成热力图
    plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu, alpha=0.8)
    
    # 将原始数据点也画上去
    plt.scatter(X[:, 0], X[:, 1], c=y.squeeze(), cmap=plt.cm.RdYlBu, edgecolors='k', alpha=0.7)
    
    plt.title("Decision Boundary Heatmap")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig("decision_boundary.png")
    print("决策边界热力图已保存为 decision_boundary.png")

# 调用函数，绘制最终的热力图
plot_decision_boundary(model, X, y)

# 在 python 脚本末尾添加
np.savetxt("cpp_impl/train_data.csv", np.hstack((X_train, y_train)), delimiter=",")
np.savetxt("cpp_impl/test_data.csv", np.hstack((X_test, y_test)), delimiter=",")
print("已为 C++ 实现保存数据文件。")