import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

dataset = pd.read_csv('data/housing.csv')
X = dataset.iloc[:, 0:3].values
y = dataset.iloc[:, -1].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
# poly_reg = PolynomialFeatures(degree=4)
# X_poly = poly_reg.fit_transform(X_train)
regressor = LinearRegression()
regressor.fit(X_train, y_train)
#
# regressor = LinearRegression()
# regressor.fit(X, y)
y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score
from sklearn.metrics import accuracy_score
accuracy_score = regressor.score(X, y)
print(accuracy_score)

plt.scatter(y_test, y_pred)
plt.xlabel("Prices")
plt.ylabel("Predicted prices")
plt.title("Prices vs Predicted prices")
plt.show()
