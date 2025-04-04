#Heatmap

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_feature_correlation(df, target='mid_price', drop_cols=['C_ASK', 'C_BID']):
    """
    Generates a heatmap showing correlation of all numerical features with the target.
    """
    if drop_cols:
        df = df.drop(columns=drop_cols, errors='ignore')

    # Keep only numeric columns
    num_df = df.select_dtypes(include=[np.number])
    corr = num_df.corr()

    # Sort by correlation with target
    sorted_corr = corr[[target]].sort_values(by=target, ascending=False)

    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', annot=False, center=0)
    plt.title('Correlation Matrix')
    plt.show()

    print("\nFeatures most correlated with mid_price:")
    print(sorted_corr.drop(index=target).head(10))

df = pd.read_csv("spy_longdated_calls_cleaned_bs.csv")
df
plot_feature_correlation(df, target='mid_price', drop_cols=['C_ASK', 'C_BID','C_LAST', 'bs_price'])
