# data_ingestion/market_data.py

import alpaca_trade_api as tradeapi
import pandas as pd
import yaml
from datetime import datetime, timedelta
import os
import logging
import time

# Setup logging
logging.basicConfig(
    filename='logs/market_data.log',
    level=logging.INFO,
    format='%(asctime)s:%(levelname)s:%(message)s'
)

def load_config():
    """
    Loads API credentials and other configurations from config/config.yaml.
    Exits if the file is missing or contains errors.
    """
    try:
        with open('../config/config.yaml', 'r') as file:
            config = yaml.safe_load(file)
        return config
    except FileNotFoundError:
        logging.error("Configuration file not found.")
        print("Configuration file not found. Please ensure 'config/config.yaml' exists.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error parsing configuration file: {e}")
        print(f"Error parsing configuration file: {e}")
        exit(1)

# Load configuration
config = load_config()

API_KEY = config.get('alpaca', {}).get('api_key')
SECRET_KEY = config.get('alpaca', {}).get('secret_key')
BASE_URL = config.get('alpaca', {}).get('base_url')

if not all([API_KEY, SECRET_KEY, BASE_URL]):
    logging.error("API credentials are not properly set in the configuration file.")
    print("API credentials are missing. Please check 'config/config.yaml'.")
    exit(1)

# Initialize Alpaca API
try:
    api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')
except Exception as e:
    logging.error(f"Failed to initialize Alpaca API: {e}")
    print(f"Failed to initialize Alpaca API: {e}")
    exit(1)

def fetch_historical_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = 'day',
    retries: int = 3,
    delay: int = 5
) -> pd.DataFrame:
    """
    Fetches historical market data for a given symbol and date range from IEX,
    ensuring at least a 15-minute delay for free subscription compliance.

    Args:
        symbol (str): Stock symbol (e.g., 'AAPL').
        start_date (str): Start date in 'YYYY-MM-DD' format.
        end_date (str): End date in 'YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ' format.
        timeframe (str): Timeframe for data ('minute', 'hour', 'day'). Defaults to 'day'.
        retries (int): Number of retry attempts in case of transient errors. Defaults to 3.
        delay (int): Delay between retries in seconds. Defaults to 5.

    Returns:
        pd.DataFrame: A DataFrame containing historical market data from IEX
                      or an empty DataFrame if fetching fails.
    """
    try:
        # Define supported timeframes using Alpaca's TimeFrame enumeration
        timeframe_map = {
            'minute': tradeapi.TimeFrame.Minute,
            'hour': tradeapi.TimeFrame.Hour,
            'day': tradeapi.TimeFrame.Day
        }

        if timeframe not in timeframe_map:
            logging.error("Invalid timeframe specified. Must be 'minute', 'hour', or 'day'.")
            print("Invalid timeframe specified. Must be 'minute', 'hour', or 'day'.")
            return pd.DataFrame()

        now = datetime.utcnow()
        min_end_time = now - timedelta(minutes=15)  # 15-minute delay requirement

        # Parse end_date based on timeframe
        try:
            if timeframe == 'day':
                # For daily data, end_date format should be 'YYYY-MM-DD'
                requested_end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
            else:
                # Intraday data uses a stricter format, but for free plan usage,
                # let's keep it consistent if you decide to switch timeframe.
                requested_end_date = datetime.strptime(end_date, '%Y-%m-%dT%H:%M:%SZ')
        except ValueError:
            logging.error(f"end_date format is incorrect: {end_date}")
            print(f"end_date format is incorrect: {end_date}")
            return pd.DataFrame()

        # Adjust end_date to ensure it's not more recent than 15 minutes ago
        if timeframe == 'day':
            # Compare only the date portion for daily timeframe
            if requested_end_date > min_end_time.date():
                adjusted_end = min_end_time.date()
                adjusted_end_str = adjusted_end.strftime('%Y-%m-%d')
                logging.info(f"Adjusted end_date from {end_date} to {adjusted_end_str} (IEX 15-min delay).")
                print(f"Adjusted end_date from {end_date} to {adjusted_end_str} (IEX 15-min delay).")
                end_date = adjusted_end_str
            else:
                logging.info(f"Using end_date: {end_date}")
                print(f"Using end_date: {end_date}")
        else:
            # For intraday timeframes (minute/hour), enforce RFC3339 format
            if isinstance(requested_end_date, datetime):
                if requested_end_date > min_end_time:
                    adjusted_end = min_end_time
                    adjusted_end_str = adjusted_end.strftime('%Y-%m-%dT%H:%M:%SZ')
                    logging.info(f"Adjusted end_date from {end_date} to {adjusted_end_str} (IEX 15-min delay).")
                    print(f"Adjusted end_date from {end_date} to {adjusted_end_str} (IEX 15-min delay).")
                    end_date = adjusted_end_str
                else:
                    end_date = requested_end_date.strftime('%Y-%m-%dT%H:%M:%SZ')
                    logging.info(f"Using end_date: {end_date}")
                    print(f"Using end_date: {end_date}")
            else:
                logging.error("end_date must be a valid datetime for intraday timeframe.")
                print("end_date must be a valid datetime for intraday timeframe.")
                return pd.DataFrame()

        # Prepare start_date in the correct format
        try:
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            if timeframe != 'day':
                start_str = start_dt.strftime('%Y-%m-%dT00:00:00Z')
            else:
                start_str = start_date
        except ValueError:
            logging.error(f"start_date format is incorrect: {start_date}")
            print(f"start_date format is incorrect: {start_date}")
            return pd.DataFrame()

        if datetime.strptime(start_date, '%Y-%m-%d') > now:
            logging.error("Start date cannot be in the future.")
            print("Start date cannot be in the future.")
            return pd.DataFrame()

        # Attempt data fetching with retries
        for attempt in range(1, retries + 1):
            try:
                # For free subscriptions, specify feed='iex'
                bars = api.get_bars(
                    symbol,
                    timeframe_map[timeframe],
                    start=start_str,
                    end=end_date,
                    feed='iex'  # Explicitly use IEX
                ).df

                if bars.empty:
                    logging.warning("No data fetched. Check symbol and date range.")
                    print("No data fetched. Check symbol and date range.")
                    return pd.DataFrame()

                bars = bars.reset_index()
                # Rename 'timestamp' to 'date'
                if 'timestamp' in bars.columns:
                    bars.rename(columns={'timestamp': 'date'}, inplace=True)
                else:
                    logging.error("No 'timestamp' column found in data.")
                    print("No 'timestamp' column found in data.")
                    return pd.DataFrame()

                bars['date'] = pd.to_datetime(bars['date'])

                logging.info(f"Successfully fetched data for {symbol} from {start_str} to {end_date} via IEX.")
                return bars

            except Exception as exc:
                logging.error(f"Attempt {attempt}: Error fetching data from IEX: {exc}")
                print(f"Attempt {attempt}: Error fetching data from IEX: {exc}")
                if attempt < retries:
                    logging.info(f"Retrying in {delay} seconds...")
                    print(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
                else:
                    logging.error("Max retries reached. Failed to fetch IEX data.")
                    print("Max retries reached. Failed to fetch IEX data.")
                    return pd.DataFrame()

    except Exception as e:
        logging.error(f"Unhandled error in fetch_historical_data: {e}")
        print(f"Unhandled error in fetch_historical_data: {e}")
        return pd.DataFrame()

def save_data(df, symbol, timeframe):
    """
    Saves the DataFrame to a CSV file.
    """
    try:
        directory = 'data_preprocessed'
        os.makedirs(directory, exist_ok=True)
        filename = f"{symbol}_{timeframe}.csv"
        filepath = os.path.join(directory, filename)
        df.to_csv(filepath, index=False)
        logging.info(f"Data saved to {filepath}")
        print(f"Data saved to {filepath}")
    except Exception as e:
        logging.error(f"Error saving data: {e}")
        print(f"Error saving data: {e}")

def main():
    symbol = "SPY"
    timeframe = "day"  # 'minute', 'hour', or 'day'

    # 1-year date range with 15-min delay
    now_utc = datetime.utcnow()
    end_dt = now_utc - timedelta(minutes=15)
    start_dt = end_dt - timedelta(days= 5 * 365)

    start_date_str = start_dt.strftime('%Y-%m-%d')  # 'YYYY-MM-DD'
    end_date_str = end_dt.strftime('%Y-%m-%d')      # For daily timeframe: 'YYYY-MM-DD'

    print(f"Fetching {timeframe} data for {symbol} from {start_date_str} to {end_date_str} (IEX feed)...")
    logging.info(f"Fetching {timeframe} data for {symbol} from {start_date_str} to {end_date_str} (IEX feed)...")

    df = fetch_historical_data(symbol, start_date_str, end_date_str, timeframe=timeframe)

    if not df.empty:
        print(df.head())
        save_data(df, symbol, timeframe)
    else:
        print("No data fetched.")

if __name__ == "__main__":
    main()
