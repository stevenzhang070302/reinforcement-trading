# trade_execution/trade_executor.py

import alpaca_trade_api as tradeapi
import yaml
import logging
import os

class TradeExecutor:
    def __init__(self, config_path='config/config.yaml'):
        self.config = self.load_config(config_path)
        self.api = self.initialize_alpaca_api()
        self.logger = self.setup_logger()

    def load_config(self, config_path):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file {config_path} not found.")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)

    def initialize_alpaca_api(self):
        try:
            return tradeapi.REST(
                self.config['alpaca']['api_key'],
                self.config['alpaca']['secret_key'],
                self.config['alpaca']['base_url'],
                api_version='v2'
            )
        except Exception as e:
            raise ConnectionError(f"Failed to initialize Alpaca API: {e}")

    def setup_logger(self):
        log_file = os.path.join("trade_execution\logs", "trade_execution.log")
        logger = logging.getLogger("TradeExecutor")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            fh = logging.FileHandler(log_file)
            fh.setLevel(logging.INFO)
            formatter = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
        
        return logger

    def buy(self, symbol, qty=1):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='buy',
                type='market',
                time_in_force='gtc'
            )
            self.logger.info(f"BUY order submitted for {qty} shares of {symbol}.")
            return order
        except Exception as e:
            self.logger.error(f"Error executing BUY order for {symbol}: {e}")
            return None

    def sell(self, symbol, qty=1):
        try:
            order = self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side='sell',
                type='market',
                time_in_force='gtc'
            )
            self.logger.info(f"SELL order submitted for {qty} shares of {symbol}.")
            return order
        except Exception as e:
            self.logger.error(f"Error executing SELL order for {symbol}: {e}")
            return None
