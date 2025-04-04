import datetime as dt
from polygon import RESTClient
import pandas as pd

# === Set API key ===
API_KEY = "N5TPWLXWaGR0IN3_ttrmMROH_q9b2yBi"


client = RESTClient(api_key=API_KEY)

# === Define parameters ===
symbol = "SPY"
min_dte = 60
today = dt.date.today()
cutoff_date = today + dt.timedelta(days=min_dte)

# === Store results ===
contracts_data = []

# === List option contracts ===
for contract in client.list_options_contracts(underlying_ticker=symbol, limit=1000):
    try:
        exp_date = dt.date.fromisoformat(contract.expiration_date)
        if contract.option_type == "call" and exp_date > cutoff_date:
            contracts_data.append({
                "contract_symbol": contract.ticker,
                "expiration_date": contract.expiration_date,
                "strike_price": contract.strike_price,
                "underlying_price": contract.underlying_price,
                "description": contract.description
            })
    except:
        continue

df_contracts = pd.DataFrame(contracts_data)
print(f"Total long-dated SPY call contracts found: {len(df_contracts)}")
print(df_contracts.head())

# === OPTIONAL: Get quote snapshot for each contract ===
# You can loop through `df_contracts['contract_symbol']` and request:
# `client.get_snapshot_option(contract_symbol)`
