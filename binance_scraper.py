import pandas as pd
import numpy as np
import pandas_ta as ta
import matplotlib.pyplot as plt
from binance.client import Client
from dotenv import load_dotenv
import os
import yaml
import argparse
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

def load_indicators_config(file_path):
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def calculate_indicators(df, interval, indicator_config):
    interval_config = indicator_config.get("indicators", {}).get(interval, {})
    
    ta_indicators = []
    for indicator, params in interval_config.items():
        # Handle special case for EMA with different lengths
        if 'kind' in params:  # For renamed indicators like ema_200
            ta_config = params.copy()
            ta_config.setdefault('kind', indicator)
        else:
            ta_config = {'kind': indicator}
            ta_config.update(params)

        ta_indicators.append(ta_config)
        
    MyStrategy = ta.Strategy(name="MyStrategy", ta=ta_indicators)
    df.ta.strategy(MyStrategy, append=True)
    return df

def fetch_data(client, coin, interval, start_time, end_time):
    
    klines_futures = client.futures_historical_klines(symbol=coin, interval=interval, start_str=start_time, end_str=end_time, limit=None) # Fetch the data using Binance's API
    
    # Convert the data to a DataFrame and process it
    df = pd.DataFrame(klines_futures, columns=['open_time','open', 'high', 'low', 'close', 'volume', 'close_time', 'qav','num_trades','taker_base_vol', 'taker_quote_vol', 'ignore'])
    df = df[['open_time','open', 'high', 'low', 'close', 'volume']].astype(np.float32) # .astype(np.float64) is a must
    
    df = df.fillna(method='ffill')
    
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    df = df.set_index("open_time")
    return df

def main():
    parser = argparse.ArgumentParser(description="Binance Data Scraper")
    parser.add_argument("--coin", type=str, required=True, help="List of coins to fetch")
    parser.add_argument("--interval", type=str, required=True, help="List of time intervals")
    parser.add_argument("--start_time", type=str, required=True, help="Start time for the data in ISO format")
    parser.add_argument("--end_time", type=str, required=True, help="End time for the data in ISO format")
    parser.add_argument("--config", type=str, required=True, help="Path to the YAML indicator config file")
    parser.add_argument("--save_folder", type=str, required=True, help="Folder to save the scraped data")
    args = parser.parse_args()
    
    # Load API key from .env file and start the client
    load_dotenv()
    client = Client(api_key=os.getenv("BINANCE_API_KEY"))
    
    # Create save folder
    save_folder = Path(args.save_folder)
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Load indicator configuration
    indicator_config = load_indicators_config(args.config)
    
    # Fetch and process data for each coin and interval
    for coin in args.coin.split(","):
        for interval in args.interval.split(","): 
            df = fetch_data(client, f"{coin}USDT", interval, args.start_time, args.end_time)
            df = calculate_indicators(df, interval, indicator_config)
            
            df.columns = [column+f"_{interval}_{coin}" for column in df.columns.tolist()] # Rename columns to include interval and coin
            
            # Save the DataFrame to a CSV file
            sanitized_start = args.start_time.replace(":", "-")
            sanitized_end = args.end_time.replace(":", "-")
            df.to_csv(save_folder / f"{coin}_{interval}_from_{sanitized_start}_to_{sanitized_end}.csv")
            
            print(f'Saved data in "{args.save_folder}/{coin}_{interval}_from_{sanitized_start}_to_{sanitized_end}.csv"')
            
if __name__ == "__main__":
    main()
