import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 读取数据
df = pd.read_excel('data1.xlsx', header=None)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values

# 构建2阶多项式的特征矩阵
X = np.column_stack([x**2, x, np.ones_like(x)])  # 只保留x², x和常数项

# 线性回归模型
model = LinearRegression(fit_intercept=False)  # 截距由常数项处理
model.fit(X, y)

# 获取系数
coeffs = model.coef_

# 生成平滑的x值用于绘图
x_smooth = np.linspace(min(x), max(x), 200)
X_smooth = np.column_stack([x_smooth**2, x_smooth, np.ones_like(x_smooth)])  # 2阶多项式特征

# 预测y值
y_smooth = model.predict(X_smooth)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_smooth, y_smooth, 'r-',
         label=f'Fit: {coeffs[0]:.2e}x² + {coeffs[1]:.2e}x + {coeffs[2]:.2e}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2nd Degree Polynomial Fit using Linear Regression')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_fit_2nd.png')
plt.close()

# 打印系数
print("Polynomial coefficients (using linear regression):")
print(f"a (x²): {coeffs[0]:.6e}")
print(f"b (x): {coeffs[1]:.6e}")
print(f"d (constant): {coeffs[2]:.6e}")

# 计算R²值
y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"\nR-squared: {r2:.6f}")

# 显示图形
plt.show()