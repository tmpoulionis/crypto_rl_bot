# 📈 Crypto Trading RL Agent 💸
## 📄 **Data Preparation**

To build a trading agent we need historical data for training and backtesting. The data will consist of raw OHLCV candlesticks as well as technical indicators. Traders often use indicators from different time frames to make more informed decisions, and we will also use this strategy in our agent. The choice of time frame depends on the [type of trading](https://www.binance.com/en/square/post/15713585293665) we want to do:

1. **Day Trading:** 5-minute, 15-minute, 1-hour.
2. **Scalping:** 1-minute, 5-minute.
3. **Medium-term Trading:** 4-hour, 1-day.
4. **Position Trading:** 1-week, 1-month.

We will follow a day trading strategy, therefore we will use data from an 1-hour and a 1-day time frame and the corresponding technical indicators.

- **Candlestick data** (OHLCV) can be fetched directly from the exchange APIs, for this project we used Binance’s API.
- **Technical indicators** will be calculated using Pandas TA library.

--- 

 ## binance_scaper.py
**Arguments:**

*--coin* - "List of coins to fetch”
    
*--interval* - "List of time intervals”
    
*--start_time* - "Start time for the data in ISO format”

*--end_time* - "End time for the data in ISO format”

*--config* - "Path to the YAML indicator config file”

*--save_folder* - "Folder to save the scraped data”
    
This script automates the retrieval of historical OHLCV data from Binance for specified cryptocurrencies and time intervals. It processes the data by calculating customizable technical indicators (e.g., RSI, MACD) defined in the `config_indicators.yaml` file, structures it into pandas DataFrames with datetime indexing, and saves each dataset as a CSV file in the specified folder.
    
💡 *Note:* To run your API key you should create an .env file and put it in (”BINANCE_API_KEY=”your_api_key”). The code inserts the API as enviroment variables, you can change this in the `binance_scraper.py` file)

---

## config_indigators.yaml
    
Use this file to set configuration for  Pandas TA’s indicators, their timeframes to be calculated in and their parameters. 
    
- Each key under "indicators" represents a timeframe (e.g., "1h", "1d").
- Under each timeframe, specify the indicator names and their parameters.
- For indicators of the same type (such as multiple EMAs), include a "kind" field.
    
```yaml
    # Example config YAML file
    indicators:
      # Example for 1-hour timeframe
      "1h":
        # Indicator: Average True Range (ATR)
        atr:
          length: 14       # Period for ATR calculation
        # Indicator: Relative Strength Index (RSI)
        rsi:
          length: 14       # Period for RSI calculation
    
      # Example for 1-day timeframe
      "1d":
        # Multiple indicators of the same type require a "kind" field.
        # Exponential Moving Averages (EMA)
        ema_200:
          kind: "ema"      # Specify the type of indicator
          length: 200      # Period for the 200 EMA
        ema_50:
          kind: "ema"      # Specify the type of indicator
          length: 50       # Period for the 50 EMA
```
---

## data_preparation.ipynb
    
Use this file as the main file that you run everything from it. You will have to just run each cell in order and comments with some instructions are given. This file will run the `binance_scraper.py` and retrieve the data, load them and merge them so the dataset will be in the desired format for training.
    
This Jupyter Notebook serves as the main entry point for data preparation, execute each cell in order and follow the instruction comments.
    
1. Run the `binance_scraper.py` to retrieve historical data from Binance.
2. Load the retrieved CSV files, process the DataFrames and merge them to the final `dataset.csv` file.
3. After merging, you can use this notebook to mess with the data and visualize it to your likings.


# README will be updated :)
