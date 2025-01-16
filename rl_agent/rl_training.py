import os
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.logger import configure
from trading_env import TradingEnv
import torch

def load_data():
    """
    Load feature-enhanced data from the data_preprocessing directory.
    Must contain columns like 'close', 'MA_Short', 'MA_Long', 'RSI', etc.
    """
    features_path = os.path.join("../data_preprocessing", "AAPL_day_features.csv")
    if not os.path.exists(features_path):
        raise FileNotFoundError(f"Enhanced features file not found at {features_path}")
    df = pd.read_csv(features_path, parse_dates=["date"])
    return df

def main():
    df = load_data()

    # Create environment
    env = TradingEnv(df, initial_balance=10000, lookback_window=50)

    # Ensure the model uses GPU if available
    # device = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or torch.cuda.is_available() else "cpu"
    device = "cpu"

    # PPO Hyperparameters
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        n_steps=2048,         # Rollout steps before each update
        batch_size=64,
        learning_rate=8e-5,
        ent_coef=0.01,        # Encourage exploration
        clip_range=0.2,
        gae_lambda=0.95,
        device=device,        # Use CUDA or fallback to CPU
    )

    # Setup TensorBoard
    log_dir = os.path.join("logs", "tensorboard")
    os.makedirs(log_dir, exist_ok=True)
    from stable_baselines3.common.logger import configure
    new_logger = configure(log_dir, ["stdout", "tensorboard"])
    model.set_logger(new_logger)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=20_000,
        save_path=os.path.join("rl_agent", "models"),
        name_prefix="ppo_trading_agent_checkpoint"
    )

    # Train
    total_timesteps = 1_000_000  # Extend to 500k or 1M as required
    model.learn(total_timesteps=total_timesteps, callback=checkpoint_callback)

    # Save final model
    model_dir = os.path.join("rl_agent", "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "ppo_trading_agent.zip")
    model.save(model_path)
    print(f"Model saved at {model_path}")

if __name__ == "__main__":
    main()
