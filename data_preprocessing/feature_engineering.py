# data_preprocessing/feature_engineering.py

import pandas as pd
import os
import ta  # pip install ta
import logging

def setup_logging():
    log_file = os.path.join(os.path.dirname(__file__), 'logs', 'feature_engineering.log')
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s:%(levelname)s:%(message)s'
    )

def load_data(symbol="AAPL", timeframe="day"):
    input_file = os.path.join("..\data_ingestion\data_preprocessed", f"{symbol}_{timeframe}.csv")
    if not os.path.exists(input_file):
        logging.error(f"Raw data file {input_file} not found.")
        raise FileNotFoundError(f"Raw data file {input_file} not found.")

    df = pd.read_csv(input_file, parse_dates=['date'])
    logging.info(f"Loaded data from {input_file}")
    return df

def calculate_indicators(df):
    # Moving Averages
    df['MA_Short'] = df['close'].rolling(window=20).mean()
    df['MA_Long'] = df['close'].rolling(window=50).mean()

    # RSI
    df['RSI'] = ta.momentum.RSIIndicator(close=df['close'], window=14).rsi()

    # MACD
    macd = ta.trend.MACD(close=df['close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Diff'] = macd.macd_diff()

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(close=df['close'], window=20, window_dev=2)
    df['Bollinger_High'] = bollinger.bollinger_hband()
    df['Bollinger_Low'] = bollinger.bollinger_lband()
    df['Bollinger_Width'] = bollinger.bollinger_wband()

    # OBV
    df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['close'], volume=df['volume']).on_balance_volume()

    # ROC
    df['ROC'] = ta.momentum.ROCIndicator(close=df['close'], window=14).roc()

    # Drop NaNs from rolling calculations
    df = df.dropna().reset_index(drop=True)
    logging.info("Calculated technical indicators and updated DataFrame.")
    return df

def save_features(df, symbol="AAPL", timeframe="day"):
    output_file = os.path.join("..\data_preprocessing", f"{symbol}_{timeframe}_features.csv")
    df.to_csv(output_file, index=False)
    logging.info(f"Saved enhanced data to {output_file}")
    print(f"Features saved to {output_file}")

def main():
    setup_logging()
    symbol = "AAPL"
    timeframe = "day"

    try:
        df = load_data(symbol, timeframe)
        df = calculate_indicators(df)
        save_features(df, symbol, timeframe)
        logging.info("Feature engineering completed successfully.")
    except Exception as e:
        logging.error(f"Feature engineering failed: {e}")
        print(f"Feature engineering failed: {e}")

if __name__ == "__main__":
    main()
