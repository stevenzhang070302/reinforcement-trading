# trade_execution/broker_api.py

import alpaca_trade_api as tradeapi
import yaml

# Load configuration
def load_config():
    with open('../config/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    return config

config = load_config()

API_KEY = config['alpaca']['api_key']
SECRET_KEY = config['alpaca']['secret_key']
BASE_URL = config['alpaca']['base_url']

# Initialize Alpaca API
api = tradeapi.REST(API_KEY, SECRET_KEY, BASE_URL, api_version='v2')

def get_account_info():
    """
    Retrieves account information.
    """
    account = api.get_account()
    print("Account Status:", account.status)
    print("Cash Balance:", account.cash)

if __name__ == "__main__":
    get_account_info()
