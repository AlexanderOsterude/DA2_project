#Neural Network

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score



df = pd.read_csv("spy_longdated_calls_cleaned.csv")

features = [
    'C_IV', 'C_DELTA', 'C_VEGA', 'C_GAMMA', 'C_THETA',
    'T', 'moneyness', 'log_moneyness',
    'C_VOLUME', 'BID_SIZE', 'ASK_SIZE',
    'STRIKE_DISTANCE', 'spread'
]

df = df.dropna(subset=features + ['mid_price'])

X = df[features].values
y = df['mid_price'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1)  # Output layer for regression
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

history = model.fit(
    X_train_scaled, y_train,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

y_pred = model.predict(X_test_scaled).flatten()

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n Neural Network Evaluation:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")


plt.figure(figsize=(6, 5))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Actual Mid Price")
plt.ylabel("Predicted Mid Price")
plt.title("NN: Predicted vs Actual Mid Price")
plt.grid(True)
plt.tight_layout()
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(6, 5))
sns.histplot(residuals, bins=50, kde=True)
plt.axvline(0, color='red', linestyle='--')
plt.title("NN: Residual Distribution (Actual - Predicted)")
plt.xlabel("Residual")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
plt.plot(history.history['loss'], label='Training Loss (MSE)')
plt.plot(history.history['val_loss'], label='Validation Loss (MSE)')
plt.title("NN: Training vs Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

r2 = r2_score(y_test, y_pred)
print("R² Score:", round(r2, 4))
