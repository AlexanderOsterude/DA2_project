#Random Forest Regression model 
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

rd.seed(42)

df = pd.read_csv("spy_longdated_calls_cleaned.csv")

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

df = df.dropna(subset=features + ['mid_price'])

X = df[features]
y = df['mid_price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Random Forest Results")
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))

df_test = X_test.copy()
df_test['true_mid_price'] = y_test
df_test['predicted_mid_price'] = y_pred



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

rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
print("CV RMSE:", round(np.mean(rmse_scores), 4))


#Hyperparameter tuning with gridsearch 
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=3)
grid_search.fit(X_train, y_train)
print("Best Params:", grid_search.best_params_)

#Grid search performance
best_model = grid_search.best_estimator_

y_pred = best_model.predict(X_test)

# Evaluate performance
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("Best Model Evaluation")
print("Best Params:", grid_search.best_params_)
print("MAE:", round(mae, 4))
print("RMSE:", round(rmse, 4))


results_df = pd.DataFrame(grid_search.cv_results_)
results_df = results_df.sort_values("mean_test_score", ascending=False)
print(results_df[['params', 'mean_test_score', 'rank_test_score']].head())

results_df




# Scatter Plot: Actual vs. Predicted 
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.3, color='mediumseagreen', edgecolor='k')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label="Perfect Prediction")

plt.title("Predicted vs. Actual Mid Price")
plt.xlabel("Actual Mid Price")
plt.ylabel("Predicted Mid Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Plot Residual Distribution
residuals = y_test - y_pred
plt.figure(figsize=(10, 6))
plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
plt.axvline(x=0, color='red', linestyle='--', label='Zero Error')
plt.title("Residual Distribution: Mid Price - Predicted Price")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.legend()
plt.grid(True)
plt.show()