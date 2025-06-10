import os
import numpy as np
import pandas as pd
import torch
import gym
import matplotlib.pyplot as plt
import math

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gym import spaces
import optuna

# ------------------------------
# 1. Custom Feature Extractor (Optional)
# ------------------------------
class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """
    Custom feature extractor for trading environment.
    """
    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super(TradingFeaturesExtractor, self).__init__(observation_space, features_dim)
        n_input = np.prod(observation_space.shape)
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(n_input, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, features_dim),
            torch.nn.ReLU()
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        flat_obs = torch.flatten(observations, start_dim=1)
        return self.extractor(flat_obs)

# ------------------------------
# 2. Trading Environment
# ------------------------------
class TradingEnv(gym.Env):
    """
    Custom trading environment for a single asset.
    Observations consist of a lookback window of feature-engineered data plus [balance, position].
    Net worth is computed as balance + total_shares * current_price.
    """
    def __init__(self, df, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0       # 1 = holding, 0 = not holding
        self.total_shares = 0
        self.lookback_window = lookback_window
        self.current_step = lookback_window
        self.hold_steps = 0     # penalize prolonged holding
        
        # Action space: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = spaces.Discrete(3)
        # Observation space: (lookback_window x (num_features + 2))
        num_features = len(self.df.columns) - 1  # excluding date column
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf,
                                            shape=(lookback_window, num_features + 2),
                                            dtype=np.float32)
        # To track net worth over the episode
        self.episode_networth = []

    def _next_observation(self):
        start = self.current_step - self.lookback_window
        end = self.current_step
        obs_df = self.df.iloc[start:end].drop(columns=['date'], errors='ignore')
        obs = obs_df.values  # shape: (lookback_window, num_features)
        extra = np.array([[self.balance, self.position]] * self.lookback_window)
        return np.hstack((obs, extra)).astype(np.float32)

    def step(self, action):
        current_price = self.df.loc[self.current_step, 'close']
        prev_net = self.balance + self.total_shares * current_price

        # Execute action
        if action == 1:  # BUY
            if self.balance >= current_price:
                self.position = 1
                self.total_shares += 1
                self.balance -= current_price
        elif action == 2:  # SELL
            if self.position == 1 and self.total_shares > 0:
                self.position = 0
                self.total_shares -= 1
                self.balance += current_price

        new_net = self.balance + self.total_shares * current_price
        profit = new_net - prev_net

        # Risk-adjusted reward: amplify profits, penalize losses and prolonged holding
        reward = profit
        if action == 0:
            self.hold_steps += 1
            reward -= 0.02 * self.hold_steps
        else:
            self.hold_steps = 0
        reward *= 100.0
        if reward < 0:
            reward *= 2.0

        # Record net worth for evaluation (liquidation is inherently handled in net worth)
        self.episode_networth.append(new_net)

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)
        obs = self._next_observation()
        info = {'net_worth': new_net, 'balance': self.balance, 'position': self.position}
        return obs, reward, done, info

    def reset(self):
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.current_step = self.lookback_window
        self.hold_steps = 0
        self.episode_networth = []
        return self._next_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")

# ------------------------------
# 3. Custom Evaluation Callback
# ------------------------------
class TradingEvalCallback(BaseCallback):
    """
    Custom callback to evaluate trading performance at set intervals.
    It runs a full evaluation episode (with deterministic actions), and logs total profit
    and percentage return. Liquidation of remaining positions is inherently computed in net worth.
    """
    def __init__(self, eval_env, eval_freq: int = 10000, verbose=0):
        super(TradingEvalCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            total_profit, percent_return = self.evaluate_trading()
            self.logger.record("eval/total_profit", total_profit)
            self.logger.record("eval/percent_return", percent_return)
            if self.verbose > 0:
                print(f"Evaluation at step {self.num_timesteps}: Profit = {total_profit:.2f}, Return = {percent_return:.2f}%")
        return True

    def evaluate_trading(self):
        env = self.eval_env
        obs = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
        # Compute profit based on net worth (liquidation is automatic in net worth calc.)
        initial_net = env.episode_networth[0] if env.episode_networth else env.initial_balance
        final_net = env.episode_networth[-1] if env.episode_networth else env.balance
        total_profit = final_net - initial_net
        percent_return = ((final_net - initial_net) / initial_net) * 100
        return total_profit, percent_return

# ------------------------------
# 4. Data Loading Function
# ------------------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}.")
    df = pd.read_csv(file_path, parse_dates=['date'])
    df.columns = [col.lower() for col in df.columns]
    df.fillna(method='ffill', inplace=True)
    return df

# ------------------------------
# 5. Hyperparameter Tuning Objective with Optuna
# ------------------------------
def objective(trial):
    # Suggest hyperparameters for PPO
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.99)
    clip_range = trial.suggest_uniform('clip_range', 0.1, 0.3)
    gae_lambda = trial.suggest_uniform('gae_lambda', 0.90, 0.99)
    ent_coef = trial.suggest_uniform('ent_coef', 0.001, 0.05)
    
    policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    # Create training and evaluation environments
    df = load_data('data/SP500.csv')  # update with your path
    train_env = TradingEnv(df, initial_balance=10000, lookback_window=50)
    eval_env = TradingEnv(df, initial_balance=10000, lookback_window=50)
    
    # Setup a minimal TensorBoard logger (optional)
    log_dir = "./logs_optuna"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout"])
    
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=0,
        n_steps=2048,
        batch_size=64,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        clip_range=clip_range,
        gae_lambda=gae_lambda,
        policy_kwargs=policy_kwargs,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)
    
    # Create our evaluation callback; use a shorter eval frequency for tuning
    eval_callback = TradingEvalCallback(eval_env, eval_freq=5000, verbose=0)
    
    # Train the model for a limited number of timesteps for speed
    model.learn(total_timesteps=200000, callback=eval_callback)
    
    # After training, evaluate the model on the eval environment
    total_profit, percent_return = eval_callback.evaluate_trading()
    
    # Return the percent return as the objective to maximize
    return percent_return

# ------------------------------
# 6. Main Function for Hyperparameter Tuning
# ------------------------------
def main():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)
    
    print("Number of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value (Percent Return): {trial.value:.2f}%")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # Optionally, retrain the final model using the best parameters
    df = load_data('data/SP500.csv')
    train_env = TradingEnv(df, initial_balance=10000, lookback_window=50)
    eval_env = TradingEnv(df, initial_balance=10000, lookback_window=50)
    
    best_policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=64),
        net_arch=dict(pi=[64, 64], vf=[64, 64])
    )
    
    final_model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        learning_rate=trial.params['learning_rate'],
        gamma=trial.params['gamma'],
        ent_coef=trial.params['ent_coef'],
        clip_range=trial.params['clip_range'],
        gae_lambda=trial.params['gae_lambda'],
        policy_kwargs=best_policy_kwargs,
        tensorboard_log="./logs_final",
    )
    
    final_model.learn(total_timesteps=1000000)
    model_path = os.path.join("rl_agent", "models", "ppo_trading_agent_final.zip")
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    final_model.save(model_path)
    print(f"Final model saved at {model_path}")

if __name__ == "__main__":
    main()
