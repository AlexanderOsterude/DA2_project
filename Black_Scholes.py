import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('spy_longdated_calls_cleaned.csv')

def black_scholes_call(S, K, sigma, r, t):
    d1 = (np.log(S / K) + (r + (sigma**2)/2) * t) / (sigma * np.sqrt(t))
    d2 = d1 - sigma * np.sqrt(t)
    C = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return C

r = 0.01

df['bs_price'] = black_scholes_call(
    S=df['UNDERLYING_LAST'],
    K=df['STRIKE'],
    sigma=df['C_IV'],
    r=r,
    t=df['T']
)

# Calculate error metrics
mae = mean_absolute_error(df['mid_price'], df['bs_price'])
rmse = np.sqrt(mean_squared_error(df['mid_price'], df['bs_price']))

print("BS MAE:", round(mae, 4))
print("BS RMSE:", round(rmse, 4))

#output_file = "spy_longdated_calls_cleaned_bs.csv"
#df.to_csv(output_file, index=False)
#df

#print(df[['QUOTE_DATE', 'STRIKE', 'C_IV', 'mid_price', 'bs_price']].head())

# Load your data
df = pd.read_csv('spy_longdated_calls_cleaned_bs.csv')

# Drop missing rows
df = df.dropna(subset=['mid_price', 'bs_price'])

# Scatter plot
plt.figure(figsize=(8, 6))
sns.scatterplot(x='mid_price', y='bs_price', data=df, alpha=0.5)
plt.plot([df['mid_price'].min(), df['mid_price'].max()],
         [df['mid_price'].min(), df['mid_price'].max()],
         'r--', label='Perfect Fit Line')
plt.xlabel("Actual Mid Price")
plt.ylabel("Black-Scholes Price")
plt.title("Black-Scholes vs. Actual Mid Price")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# Histogram of residuals
df['residual'] = df['mid_price'] - df['bs_price']
plt.figure(figsize=(8, 5))
sns.histplot(df['residual'], bins=50, kde=True, color='skyblue')
plt.axvline(0, color='red', linestyle='--')
plt.title("Residual Distribution: Mid Price - Black-Scholes Price")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

