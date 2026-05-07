import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# LOAD DATA
data = pd.read_csv("data/processed/training_data_2.csv")
data.columns = data.columns.str.strip().str.lower()

# FEATURES / TARGET
X = data.drop(columns=["points"])
y = data["points"]

# TRAIN / TEST SPLIT (time-based)
train_data = data[data["year"] <= 2018]
test_data = data[data["year"] > 2018]

X_train = train_data.drop(columns=["points"])
y_train = train_data["points"]

X_test = test_data.drop(columns=["points"])
y_test = test_data["points"]

# SCALE
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# MODEL 
model = SGDRegressor(max_iter=1, warm_start=True, learning_rate="constant", eta0=0.001)

train_losses = []
test_losses = []

epochs = 50

for i in range(epochs):
    model.fit(X_train, y_train)

    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_loss = mean_squared_error(y_train, train_pred)
    test_loss = mean_squared_error(y_test, test_pred)

    train_losses.append(train_loss)
    test_losses.append(test_loss)

# PLOT
plt.figure(figsize=(10,5))
plt.plot(train_losses, label="Train Loss")
plt.plot(test_losses, label="Test Loss")

plt.title("Model Learning Curve")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid()
plt.show()