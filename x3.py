import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
df = pd.read_excel('data1.xlsx',header=None)
x = df.iloc[:, 0].values
y = df.iloc[:, 1].values
X = np.column_stack([x**3, x**2, x, np.ones_like(x)])
model = LinearRegression(fit_intercept=False)  # intercept is handled by ones column
model.fit(X, y)
coeffs = model.coef_
x_smooth = np.linspace(min(x), max(x), 200)
X_smooth = np.column_stack([x_smooth**3, x_smooth**2, x_smooth, np.ones_like(x_smooth)])
y_smooth = model.predict(X_smooth)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x_smooth, y_smooth, 'r-',
         label=f'Fit: {coeffs[0]:.2e}x³ + {coeffs[1]:.2e}x² + {coeffs[2]:.2e}x + {coeffs[3]:.2e}')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('3rd Degree Polynomial Fit using Linear Regression')
plt.legend()
plt.grid(True)
plt.savefig('polynomial_fit_3rd.png')
plt.close()
# Print coefficients
print("Polynomial coefficients (using linear regression):")
print(f"a (x³): {coeffs[0]:.6e}")
print(f"b (x²): {coeffs[1]:.6e}")
print(f"c (x): {coeffs[2]:.6e}")
print(f"d (constant): {coeffs[3]:.6e}")
y_pred = model.predict(X)
r2 = model.score(X, y)
print(f"\nR-squared: {r2:.6f}")
plt.show(x,y_pred)