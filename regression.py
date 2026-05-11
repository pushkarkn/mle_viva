import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import sklearn as sk
from sklearn import linear_model
from sklearn import metrics
import matplotlib as plt 

df = pd.read_csv('Housing.csv')
df.info()

df_encoded = pd.get_dummies(df, drop_first=True)

X = df_encoded.drop('price', axis=1)
y = df_encoded['price']


model = LinearRegression()
model.fit(X, y)

print(f"R^2 Score: {model.score(X, y):.4f}")
print(f"Intercept: {model.intercept_}")

df['predicted_price'] = model.predict(X)
print(df[['price', 'predicted_price']].head())

from sklearn import metrics

predictions = model.predict(X)

mae = metrics.mean_absolute_error(y, predictions)
mse = metrics.mean_squared_error(y, predictions)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(y, predictions)

print(f"Mean Absolute Error (MAE):     {mae:,.2f}")
print(f"Mean Squared Error (MSE):      {mse:,.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
print(f"R² Score:                      {r2:.4f}")
