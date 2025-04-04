# === Random Forest Regression Model ===
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error

# === Load the cleaned dataset ===
df = pd.read_csv("spy_longdated_calls_cleaned.csv")

# === Feature set ===
features = [
    'C_IV',
    'C_DELTA',
    'C_VEGA',
    'C_GAMMA',
    'C_THETA',
    'T',
    'moneyness',
    'log_moneyness',
    'C_VOLUME',
    'BID_SIZE',
    'ASK_SIZE',
    'STRIKE_DISTANCE',
    'spread'
]

# === Drop rows with missing values ===
df = df.dropna(subset=features + ['mid_price'])

# === Train/Test Split ===
X = df[features]
y = df['mid_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Train the Random Forest ===
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# === Predict ===
y_pred = model.predict(X_test)

# === Evaluate ===
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Random Forest Results")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

# Optional: Add predictions to a new DataFrame
df_test = X_test.copy()
df_test['true_mid_price'] = y_test
df_test['predicted_mid_price'] = y_pred



#feature importance in regression model
import matplotlib.pyplot as plt

importances = model.feature_importances_
sorted_idx = np.argsort(importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(range(len(features)), importances[sorted_idx], align='center')
plt.yticks(range(len(features)), [features[i] for i in sorted_idx])
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.gca().invert_yaxis()
plt.show()

mae_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_mean_absolute_error')
print("CV MAE:", round(np.mean(mae_scores), 4))