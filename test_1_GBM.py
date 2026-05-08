import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# LOAD DATA
data = pd.read_csv("data/processed/Advanced_Training_Data_1.csv")
data.columns = data.columns.str.strip().str.lower()

# DROP NON-NUMERIC / UNUSED COLUMNS (IMPORTANT)
data = data.drop(columns=["school_id"], errors="ignore")
data = data.drop(columns=["school"], errors="ignore")

# TIME SPLIT
train_data = data[data["year"] <= 2018]
test_data = data[data["year"] > 2018]

X_train = train_data.drop(columns=["points"])
y_train = train_data["points"]

X_test = test_data.drop(columns=["points"])
y_test = test_data["points"]

#MODEL
model = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    num_leaves=20,
    min_child_samples=2,      # Very aggressive for small datasets
    min_split_gain=0.0,       # Allow even tiny improvements
    path_smooth=0.1,          # Helps with "noisy" predictions
    reg_alpha=0.5,
    reg_lambda=0.5,
    importance_type='gain'
)

# TRAIN 
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l1",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

# PREDICT
predictions = model.predict(X_test)

# EVALUATE
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("\n===== LIGHTGBM PERFORMANCE =====")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")

# RESULTS
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
})

print("\n===== SAMPLE PREDICTIONS =====")
print(results.head(20))

# FEATURE IMPORTANCE
#lgb.plot_importance(model, max_num_features=15)
#plt.show()

# SORT so the line makes sense visually
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": predictions
}).reset_index(drop=True)

plt.figure()

results = results.sort_values(by="Actual").reset_index(drop=True)

plt.figure(figsize=(10, 6))
plt.plot(results["Actual"], label="Actual", color='blue', alpha=0.6)
plt.plot(results["Predicted"], label="Predicted", color='orange', alpha=0.8)
plt.title("Actual vs Predicted (Sorted by Magnitude)")

# actual vs predicted lines
plt.plot(results["Actual"], label="Actual", linewidth=2)
plt.plot(results["Predicted"], label="Predicted", linewidth=2)

plt.title("Actual vs Predicted (LightGBM)")
plt.xlabel("Sample Index")
plt.ylabel("Points")
plt.legend()

plt.show()