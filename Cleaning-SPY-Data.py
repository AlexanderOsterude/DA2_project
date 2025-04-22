import pandas as pd
import numpy as np
from scipy.stats import norm

file_path = "spy_2020_2022.csv"
df = pd.read_csv(file_path, low_memory=False)

df.columns = [col.strip().strip('[]') for col in df.columns]

#Parsing C_SIZE into BID_SIZE and ASK_SIZE 
def parse_bid_size(val):
    try:
        return int(val.split('x')[0].strip())
    except:
        return np.nan

def parse_ask_size(val):
    try:
        return int(val.split('x')[1].strip())
    except:
        return np.nan

df['BID_SIZE'] = df['C_SIZE'].apply(parse_bid_size)
df['ASK_SIZE'] = df['C_SIZE'].apply(parse_ask_size)

numeric_cols = [
    'DTE', 'UNDERLYING_LAST', 'STRIKE',
    'C_IV', 'C_LAST', 'C_BID', 'C_ASK',
    'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO',
    'C_VOLUME', 'BID_SIZE', 'ASK_SIZE'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Cleaning
df_clean = df[
    (df['DTE'] >= 60) & (df['DTE'] > 0) &
    (df['UNDERLYING_LAST'] > 0) &
    (df['STRIKE'] > 0) &
    (df['C_IV'] > 0) &
    (~df['C_LAST'].isna()) &
    (~df['C_BID'].isna()) &
    (~df['C_ASK'].isna()) &
    (~df['BID_SIZE'].isna()) &
    (~df['ASK_SIZE'].isna()) &
    (df['C_ASK'] >= df['C_BID'])
].copy()

#Computing derived features 
df_clean['mid_price'] = (df_clean['C_BID'] + df_clean['C_ASK']) / 2
df_clean['spread'] = df_clean['C_ASK'] - df_clean['C_BID']
df_clean['moneyness'] = df_clean['UNDERLYING_LAST'] / df_clean['STRIKE']
df_clean['T'] = df_clean['DTE'] / 365
df_clean['log_moneyness'] = np.log(df_clean['moneyness'])

#Filter for near-the-money options
df_near_money = df_clean[(df_clean['moneyness'] >= 0.9) & (df_clean['moneyness'] <= 1.05)]

#Select relevant columns 
columns_to_keep = [
    'QUOTE_DATE', 'EXPIRE_DATE', 'DTE', 'T',
    'UNDERLYING_LAST', 'STRIKE', 'C_IV', 'C_LAST', 'C_BID', 'C_ASK',
    'mid_price', 'spread',
    'C_DELTA', 'C_GAMMA', 'C_VEGA', 'C_THETA', 'C_RHO',
    'moneyness', 'log_moneyness', 'C_VOLUME', 'BID_SIZE', 'ASK_SIZE',
    'QUOTE_TIME_HOURS', 'STRIKE_DISTANCE'
]

df_final = df_near_money[columns_to_keep]

#Save cleaned data to CSV 
output_file = "spy_longdated_calls_cleaned.csv"
df_final.to_csv(output_file, index=False)

print(f"Cleaned data saved to '{output_file}'")
print(df_final.head())
