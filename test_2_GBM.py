import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ============================================
# 1. LOAD DATA
# ============================================
data = pd.read_csv("data/processed/Advanced_Training_Data_1.csv")

# Clean column names
data.columns = data.columns.str.strip().str.lower()

# ============================================
# 2. REMOVE LEAKAGE COLUMNS
# ============================================

# These columns leak the answer directly or indirectly
leakage_cols = [
    "total_calculated_points",
    "points_per_event",
    "win_percentage"
]

# Drop ONLY if they exist
data = data.drop(columns=leakage_cols, errors="ignore")

# ============================================
# 3. TRAIN / TEST SPLIT
# ============================================
train = data[data["year"] <= 2018]
test = data[data["year"] > 2018]

# Columns we NEVER train on
drop_cols = [
    "points",
    "year",
    "school_id",
    "meet_id"
]

# Features / target
X_train = train.drop(columns=drop_cols, errors="ignore")
y_train = train["points"]

X_test = test.drop(columns=drop_cols, errors="ignore")
y_test = test["points"]

# ============================================
# 4. LIGHTGBM MODEL
# ============================================
model = lgb.LGBMRegressor(
    n_estimators=500,
    learning_rate=0.03,
    num_leaves=20,
    min_child_samples=10,
    reg_alpha=0.5,
    reg_lambda=1.0,
    random_state=42,
    importance_type="gain"
)

# ============================================
# 5. TRAIN
# ============================================
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="l2",
    callbacks=[
        lgb.early_stopping(stopping_rounds=50),
        lgb.log_evaluation(50)
    ]
)

# ============================================
# 6. PREDICT
# ============================================
preds = model.predict(X_test)

# ============================================
# 7. EVALUATE
# ============================================
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n===== LIGHTGBM PERFORMANCE =====")
print(f"MAE: {mae:.2f}")
print(f"R² Score: {r2:.4f}")

# ============================================
# 8. FEATURE IMPORTANCE
# ============================================
lgb.plot_importance(model, max_num_features=15)
plt.title("Feature Importance")
plt.show()

# ============================================
# 9. VISUALIZE PREDICTIONS
# ============================================
results = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": preds
})

# Sort for smoother graph
results = results.sort_values("Actual").reset_index(drop=True)

plt.figure(figsize=(12, 6))

plt.plot(
    results["Actual"],
    label="Actual",
    linewidth=3
)

plt.plot(
    results["Predicted"],
    label="Predicted",
    linewidth=2
)

plt.fill_between(
    range(len(results)),
    results["Actual"],
    results["Predicted"],
    alpha=0.15
)

plt.title("Actual vs Predicted (LightGBM)")
plt.xlabel("Sorted Samples")
plt.ylabel("Points")
plt.legend()

plt.show()