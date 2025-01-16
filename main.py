# main.py (or rl_live_trading.py)

import pandas as pd
import numpy as np
from stable_baselines3 import PPO
from rl_agent.trading_env import TradingEnv
from trade_execution.trade_executor import TradeExecutor
import time

def main():
    # 1. Load Enhanced Data
    df = pd.read_csv("data_preprocessing/AAPL_day_features.csv", parse_dates=['date'])

    # 2. Load Trained RL Model
    model_path = "rl_agent/rl_agent/models/ppo_trading_agent"
    model = PPO.load(model_path)
    
    # 3. Initialize Environment (for state representation)
    env = TradingEnv(df, initial_balance=10000, lookback_window=50)
    obs = env.reset()

    # 4. Initialize Trade Executor (paper trading mode)
    executor = TradeExecutor(config_path="config/config.yaml")

    # 5. Simulate or Real-time Loop
    for step in range(50, len(df)):  # Example loop
        action, _states = model.predict(obs, deterministic=True)

        # Translate action to trade
        symbol = "AAPL"
        if action == 1:  # BUY
            executor.buy(symbol, qty=1)
        elif action == 2:  # SELL
            executor.sell(symbol, qty=1)
        else:
            print("HOLD")

        # Step environment to update state
        obs, reward, done, info = env.step(action)
        print(f"Step {step}, Reward: {reward}, Portfolio Value: {info['portfolio_value']:.2f}")

        if done:
            break
        # In real-time, wait for next bar or next data update
        time.sleep(1)

if __name__ == "__main__":
    main()
