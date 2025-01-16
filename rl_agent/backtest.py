# rl_agent/backtest.py

import os
import pandas as pd
from stable_baselines3 import PPO
from trading_env import TradingEnv

def load_test_data():
    """
    Loads the same 'AAPL_day_features.csv' for backtesting.
    """
    test_data_path = os.path.join("../data_preprocessing", "AAPL_day_features.csv")
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"Test data not found: {test_data_path}")
    return pd.read_csv(test_data_path, parse_dates=['date'])

def backtest():
    # 1. Load Data
    df = load_test_data()

    # 2. Create Environment
    env = TradingEnv(df, initial_balance=10000, lookback_window=50)

    # 3. Load Trained Model
    model_path = os.path.join("rl_agent", "models", "ppo_trading_agent.zip")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained model found at {model_path}")
    model = PPO.load(model_path)

    # 4. Backtest loop
    obs = env.reset()
    portfolio_values = []
    done = False
    step_count = 0
    max_steps = len(df) - env.lookback_window

    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        portfolio_values.append(info["portfolio_value"])

        print(f"Step {step_count}, Action: {action}, "
              f"Reward: {reward:.2f}, "
              f"Portfolio Value: {info['portfolio_value']:.2f}, "
              f"Balance: {info['balance']:.2f}, "
              f"Position: {info['position']}")

        step_count += 1

    # Final Value
    final_value = portfolio_values[-1] if portfolio_values else env.initial_balance
    pct_change = (final_value - env.initial_balance) / env.initial_balance * 100.0

    print("\n--- Backtest Summary ---")
    print(f"Initial Balance: {env.initial_balance}")
    print(f"Final Portfolio Value: {final_value:.2f}")
    print(f"Percentage Change: {pct_change:.2f}%")

    results_df = pd.DataFrame({"portfolio_value": portfolio_values})
    results_df.to_csv("backtest_results.csv", index=False)
    print("Backtest results saved to backtest_results.csv")

if __name__ == "__main__":
    backtest()
