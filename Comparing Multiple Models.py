#Implementing and Comparing AdaBoost, Bagging, LinearRegression and other features 
import random as rd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.linear_model import LinearRegression, Lasso, Ridge, LassoCV, RidgeCV, LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import mean_squared_error,mean_absolute_error, accuracy_score, r2_score
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn.ensemble import (
    RandomForestRegressor,
    RandomForestClassifier,
    AdaBoostRegressor,
    AdaBoostClassifier,
    BaggingRegressor,
    BaggingClassifier,
)


# seed for reproducibility
rd.seed(42)

# Load the cleaned dataset
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

# Train/Test Split
X = df[features]
y = df['mid_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    return model, mae, rmse, r2

models = {
    'Linear Regression': LinearRegression(),
    'Decision Tree': DecisionTreeRegressor(),
    'Bagging': BaggingRegressor(),
    'Random Forest': RandomForestRegressor(),
    'AdaBoost': AdaBoostRegressor()
}

results = {}


for name, model in models.items():
    m, mae, rmse, r2 = evaluate_model(model, X_train, y_train, X_test, y_test)
    results[name] = {'MAE': round(mae, 4), 'RMSE': round(rmse, 4), 'R^2': round(r2, 4)}
    print(f"{name}: MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")


# === Loop through models and plot predicted vs actual and residuals ===
for name, model in models.items():
    # Fit & predict
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    residuals = y_test - y_pred

    # === Plot Predicted vs Actual ===
    plt.figure(figsize=(6, 5))
    sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')  # 45-degree line
    plt.xlabel("Actual Mid Price")
    plt.ylabel("Predicted Mid Price")
    plt.title(f"{name} – Predicted vs Actual")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Plot Residuals ===
    plt.figure(figsize=(6, 5))
    sns.histplot(residuals, bins=50, kde=True)
    plt.axvline(0, color='red', linestyle='--')
    plt.xlabel("Residual (Actual - Predicted)")
    plt.title(f"{name} – Residual Distribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

