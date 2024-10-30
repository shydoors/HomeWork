import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error


# get data from boston.csv and Create DataFrame from the data
# the path can't be absolute path if you don't want to do anything and
# download this just for FINISHING HOMEWORK
data = []
with open('boston.csv', 'r') as f:
    for idx, line in enumerate(f):
        if idx == 0:
            continue  # Skip the header line
        values = line.strip().split(',')
        if len(values) == 14:  # Only process valid lines
            data.append([float(x) for x in values])
#the header of the database；
columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df = pd.DataFrame(data, columns=columns)
#DataFrame created




# 计算相关性并绘制柱状图
plt.figure(figsize=(20, 20))#give the size of the plot saved as png file
correlation = df.corr()['MEDV'].sort_values(ascending=False)
plt.bar(range(len(correlation)), correlation)
plt.xticks(range(len(correlation)), correlation.index, rotation=45)
plt.title('Correlation with House Price (MEDV)')#the tatil will be put on the immediately of the output image
plt.tight_layout()
plt.savefig('correlation.png')#the name of pic when you saved as document.png
plt.close()
#计算与房价（MEDV）的相关性，并绘制相关性柱状图。相关性值被排序并保存为 correlation.png






# Create scatter plots for top correlated features
# 画散点图
# 创建一个 2x2 的子图，图像大小为 30x15
fig, axes = plt.subplots(2, 2, figsize=(30, 15))# 定义要绘制的特征列表
features = ['RM', 'LSTAT', 'PTRATIO', 'NOX']
# 遍历特征列表，使用 enumerate 获取索引和特征名
for idx, feature in enumerate(features):
    # 计算当前特征所在的行和列
    row = idx // 2  # 行索引
    col = idx % 2  # 列索引
    # 从 DataFrame 中提取特征数据，并将其重塑为二维数组
    X = df[feature].values.reshape(-1, 1)
    # 提取目标变量（房价）
    y = df['MEDV'].values
    # 创建线性回归模型
    model = LinearRegression()
    # 拟合线性回归模型
    model.fit(X, y)
    # 使用模型进行预测
    y_pred = model.predict(X)
    # 计算 R²（决定系数）值
    r2 = r2_score(y, y_pred)
    # 在对应的子图上绘制散点图
    axes[row, col].scatter(X, y, alpha=0.5)
    # 在散点图上绘制线性回归的预测线
    axes[row, col].plot(X, y_pred, color='red', linewidth=2)
    # 设置 x 轴标签为当前特征名
    axes[row, col].set_xlabel(feature)
    # 设置 y 轴标签为 'Price (MEDV)'
    axes[row, col].set_ylabel('Price (MEDV)')
    # 设置子图标题，包含特征名和 R² 值
    axes[row, col].set_title(f'{feature} vs Price (R² = {r2:.3f})')
# 调整子图布局，避免重叠
plt.tight_layout()
# 将绘制的图保存为 'scatter_plots.png'
plt.savefig('scatter_plots.png')
# 关闭图形窗口
plt.close()










# EN:Print correlation values
# CN:打印相关性值
print("\nCorrelation with house prices (MEDV):")
print(correlation)








# Multiple linear regression
X = df[features]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("\nMultiple Linear Regression Results:")
print(f"R^2 Score: {r2_score(y_test, y_pred):.3f}")
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")