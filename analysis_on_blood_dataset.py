import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#1. Get the Data

dataset = pd.read_excel('C:\\Users\\vasu\\Desktop\\projects\\blood dataset\\blood.xlsx')
X = dataset.iloc[2:, 1].values
y = dataset.iloc[2:, 2].values
X = X.reshape(-1, 1)

#2. Discovery and Visualization

plt.scatter(X, y)
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Analysis on the Blood Dataset')
plt.show()

#3. Data Preprocessing

#No need

#4. Select and Train an ML Algo

from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

lin_reg.predict([[20]])
lin_reg.predict([[30]])

c = lin_reg.intercept_

m = lin_reg.coef_

m, c

ayush = m * 20 + c
ayush

rahul = m * 30 + c
rahul

y_pred = lin_reg.predict(X)

plt.scatter(X, y)
plt.plot(X, y_pred, c = "red")
plt.xlabel('Age')
plt.ylabel('Systolic Blood Pressure')
plt.title('Analysis on the Blood Dataset')
plt.show()

lin_reg.score(X, y)































































