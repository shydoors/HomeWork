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


plt.figure(figsize=(12, 8))#give the size of the plot saved as png file
correlation = df.corr()['MEDV'].sort_values(ascending=False)
plt.bar(range(len(correlation)), correlation)
plt.xticks(range(len(correlation)), correlation.index, rotation=45)
plt.title('Correlation with House Price (MEDV)')#the tatil will be put on the immediately of the output image
plt.tight_layout()
plt.savefig('correlation.png')#the name of pic when you saved as document.png
plt.close()
#计算与房价（MEDV）的相关性，并绘制相关性柱状图。相关性值被排序并保存为 correlation.png




# Create scatter plots for top correlated features
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
features = ['RM', 'LSTAT', 'PTRATIO', 'NOX']

for idx, feature in enumerate(features):
    row = idx // 2
    col = idx % 2

    X = df[feature].values.reshape(-1, 1)
    y = df['MEDV'].values

    # Fit linear regression
    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    r2 = r2_score(y, y_pred)

    axes[row, col].scatter(X, y, alpha=0.5)
    axes[row, col].plot(X, y_pred, color='red', linewidth=2)
    axes[row, col].set_xlabel(feature)
    axes[row, col].set_ylabel('Price (MEDV)')
    axes[row, col].set_title(f'{feature} vs Price (R² = {r2:.3f})')

plt.tight_layout()
plt.savefig('scatter_plots.png')
plt.close()

# Print correlation values
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
print(f"R² Score: {r2_score(y_test, y_pred):.3f}")
print("\nFeature Coefficients:")
for feature, coef in zip(features, model.coef_):
    print(f"{feature}: {coef:.3f}")