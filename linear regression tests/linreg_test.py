import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# LOAD DATA


data = pd.read_csv("data/processed/training_data_1.csv")

# FEATURES / TARGET

# Remove target columns from features
X = data.drop(columns=["points", "rank"])

# Target we want to predict
y = data["points"]

# TIME-BASED TRAIN / TEST SPLIT

# Train on older years
train_data = data[data["year"] <= 2018]

# Test on newer years
test_data = data[data["year"] > 2018]

X_train = train_data.drop(columns=["points", "rank"])
y_train = train_data["points"]

X_test = test_data.drop(columns=["points", "rank"])
y_test = test_data["points"]

# SCALE FEATURES

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# MODEL

model = LinearRegression()

# Train model
model.fit(X_train_scaled, y_train)

# PREDICTIONS

predictions = model.predict(X_test_scaled)

# EVALUATION

mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== MODEL PERFORMANCE =====")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# SHOW SAMPLE PREDICTIONS

results = pd.DataFrame({
    "Actual Points": y_test.values,
    "Predicted Points": predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(20))