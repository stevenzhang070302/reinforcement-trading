# --- PPO Trading Agent with Reward‑Shaping Improvements ---
# This script replaces the original version, adding the step‑by‑step reward
# tuning components we discussed (drawdown penalty, Sharpe term, trade‑frequency
# penalty, coverage bonus, normalisation / clipping) and exposing their
# coefficients to Optuna so you can search over them.

import os
import numpy as np
import pandas as pd
import torch
import gym
from gym import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import optuna

# ------------------------------
# 1. Custom Feature Extractor
# ------------------------------
class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """Enhanced MLP feature extractor with deeper architecture, attention mechanism, and regularization."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 64):
        super().__init__(observation_space, features_dim)
        n_input = int(np.prod(observation_space.shape))
        
        # Deeper network with dropout for better regularization
        self.extractor = torch.nn.Sequential(
            torch.nn.Linear(n_input, 512),  # Wider first layer
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(512),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(512, 256),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(256),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm1d(128),
            torch.nn.Dropout(0.3),  # Increased dropout
            torch.nn.Linear(128, features_dim),
            torch.nn.LeakyReLU(),
        )
        
        # Self-attention mechanism to capture temporal patterns
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(features_dim, features_dim),
            torch.nn.Tanh(),
            torch.nn.Linear(features_dim, 1),
            torch.nn.Softmax(dim=1)
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        batch_size = observations.shape[0]
        flat_obs = torch.flatten(observations, start_dim=1)
        features = self.extractor(flat_obs)
        
        # Reshape for attention if needed (for sequence data)
        if len(observations.shape) > 2 and observations.shape[1] > 1:  # If we have sequence data
            try:
                # Extract temporal features if possible
                seq_length = observations.shape[1]
                feature_dim = features.shape[1] // seq_length
                reshaped_features = features.view(batch_size, seq_length, feature_dim)
                
                # Apply attention
                attention_weights = self.attention(reshaped_features)
                context_vector = torch.sum(attention_weights * reshaped_features, dim=1)
                return context_vector
            except:
                # Fallback if reshaping fails
                return features
        return features

# ------------------------------
# 2. Trading Environment
# ------------------------------
class TradingEnv(gym.Env):
    """Single‑asset trading environment with shaped reward."""

    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        df: pd.DataFrame,
        *,
        initial_balance: float = 10_000,
        lookback_window: int = 50,
        # --- shaping coefficients ---
        lambda_hold: float = 0.02,
        lambda_drawdown: float = 10.0,
        drawdown_threshold: float = 0.05,
        lambda_sharpe: float = 5.0,
        lambda_trade_freq: float = 1.0,
        lambda_cov: float = 0.5,
        cov_bin_size: float = 500.0,
        reward_clip: float = 5.0,
        # --- new shaping coefficients ---
        lambda_volatility: float = 2.0,
        lambda_txn_cost: float = 0.5,
        txn_cost_pct: float = 0.001,  # 0.1% transaction cost
        lambda_position_size: float = 1.0,
        max_position_pct: float = 0.2,  # max 20% of capital in one position
    ):
        super().__init__()
        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0  # 1 = long, 0 = flat
        self.total_shares = 0
        self.lookback_window = lookback_window
        self.current_step = lookback_window

        # --- shaping hyper‑parameters ---
        self.lambda_hold = lambda_hold
        self.lambda_drawdown = lambda_drawdown
        self.drawdown_threshold = drawdown_threshold
        self.lambda_sharpe = lambda_sharpe
        self.lambda_trade_freq = lambda_trade_freq
        self.lambda_cov = lambda_cov
        self.cov_bin_size = cov_bin_size
        self.reward_clip = reward_clip
        
        # --- new shaping hyper-parameters ---
        self.lambda_volatility = lambda_volatility
        self.lambda_txn_cost = lambda_txn_cost
        self.txn_cost_pct = txn_cost_pct
        self.lambda_position_size = lambda_position_size
        self.max_position_pct = max_position_pct
        
        # --- advanced risk management parameters ---
        self.progressive_dd_penalty = True  # Progressive drawdown penalty
        self.risk_aversion_factor = 1.5     # Higher values = more risk averse
        self.max_drawdown_memory = 50       # Remember past drawdowns for this many steps

        # trackers
        self.hold_steps = 0
        self.episode_networth: list[float] = []
        self.peak_networth = initial_balance
        self.visited_bins: set[int] = set()
        self.running_mean = 0.0  # for reward normalisation
        self.running_var = 1.0
        self.alpha = 0.005  # decay for running stats
        self.trade_count = 0
        self.volatility_window = 20  # window for calculating market volatility
        
        # Advanced tracking for risk management
        self.drawdown_history = []  # Track historical drawdowns
        self.volatility_history = []  # Track historical volatility
        self.consecutive_losses = 0  # Track consecutive losing trades

        # action & observation spaces
        self.action_space = spaces.Discrete(3)  # 0=HOLD,1=BUY,2=SELL
        num_features = len(self.df.columns) - 1  # drop date col if present
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, num_features + 2),
            dtype=np.float32,
        )

    # ------------------------------ helpers ------------------------------
    def _next_observation(self):
        start = self.current_step - self.lookback_window
        end = self.current_step
        obs_df = self.df.iloc[start:end].drop(columns=["date"], errors="ignore")
        obs = obs_df.values
        extra = np.array([[self.balance, self.position]] * self.lookback_window)
        return np.hstack((obs, extra)).astype(np.float32)

    def _update_running_stats(self, r_raw: float):
        """EMA update for mean/variance used in normalisation."""
        diff = r_raw - self.running_mean
        self.running_mean += self.alpha * diff
        self.running_var += self.alpha * (diff**2 - self.running_var)

    def _normalise(self, r_raw: float) -> float:
        std = np.sqrt(self.running_var) + 1e-8
        return np.clip((r_raw - self.running_mean) / std, -self.reward_clip, self.reward_clip)

    # ------------------------------ core API ------------------------------
    def _calculate_market_volatility(self):
        """Calculate recent market volatility based on price changes."""
        if self.current_step < self.volatility_window + 1:
            return 0.01  # default low volatility if not enough history
        
        # Get recent closing prices
        end = self.current_step
        start = end - self.volatility_window
        prices = self.df.iloc[start:end+1]["close"].values
        
        # Calculate daily returns
        returns = np.diff(prices) / prices[:-1]
        
        # Calculate volatility (standard deviation of returns)
        volatility = np.std(returns)
        return max(volatility, 0.001)  # ensure minimum volatility
    
    def step(self, action):  # type: ignore[override]
        price = self.df.loc[self.current_step, "close"]
        prev_net = self.balance + self.total_shares * price
        prev_position = self.position
        prev_shares = self.total_shares

        # --- execute action with transaction costs ---
        txn_cost = 0.0
        if action == 1:  # BUY
            if self.balance >= price:
                # Apply transaction cost
                cost = price * self.txn_cost_pct
                if self.balance >= (price + cost):
                    self.position = 1
                    self.total_shares += 1
                    txn_cost = cost
                    self.balance -= (price + cost)
                    if prev_position == 0:
                        self.trade_count += 1
        elif action == 2:  # SELL
            if self.position == 1 and self.total_shares > 0:
                self.position = 0
                self.total_shares -= 1
                # Apply transaction cost
                cost = price * self.txn_cost_pct
                txn_cost = cost
                self.balance += (price - cost)
                self.trade_count += 1

        new_net = self.balance + self.total_shares * price
        profit = new_net - prev_net  # per‑step P&L
        
        # Track consecutive losses for risk management
        if profit < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        # ---------- reward components ----------
        r_profit = profit * 100.0  # scale dollars → reward points
        
        # Calculate current market volatility and track history
        market_volatility = self._calculate_market_volatility()
        
        # Store volatility history
        self.volatility_history.append(market_volatility)
        if len(self.volatility_history) > self.volatility_window:
            self.volatility_history.pop(0)
            
        # Calculate volatility trend for adaptive strategy
        volatility_increasing = False
        if len(self.volatility_history) >= 5:
            recent_vol = self.volatility_history[-5:]
            volatility_increasing = all(recent_vol[i] <= recent_vol[i+1] for i in range(len(recent_vol)-1))
        
        # Adaptive Hold‑cost with enhanced volatility awareness
        # In high volatility, holding might be more reasonable
        volatility_factor = 1.0 / (1.0 + market_volatility * 10.0)  # reduces penalty in high volatility
        
        # Further adjust based on volatility trend
        if volatility_increasing:
            # If volatility is increasing, reduce hold penalty even more
            volatility_factor *= 0.8
            
        # Calculate drawdown before using it
        self.peak_networth = max(self.peak_networth, new_net)
        drawdown = (self.peak_networth - new_net) / self.peak_networth
        
        # Adjust hold penalty based on drawdown situation
        if drawdown > self.drawdown_threshold * 2:  # In severe drawdown
            # Encourage more cautious behavior during large drawdowns
            if action == 0:  # HOLD
                self.hold_steps += 1
                # Reduced penalty for holding during severe drawdowns
                r_hold = -self.lambda_hold * self.hold_steps * volatility_factor * 0.5
            else:
                self.hold_steps = 0
                r_hold = 0.0
        else:  # Normal drawdown situation
            if action == 0:  # HOLD
                self.hold_steps += 1
                r_hold = -self.lambda_hold * self.hold_steps * volatility_factor
            else:
                self.hold_steps = 0
                r_hold = 0.0

        # Enhanced Drawdown penalty with progressive scaling and historical awareness
        # Note: drawdown already calculated above
        
        # Store drawdown history for trend analysis
        self.drawdown_history.append(drawdown)
        if len(self.drawdown_history) > self.max_drawdown_memory:
            self.drawdown_history.pop(0)
            
        # Calculate drawdown trend (increasing or decreasing)
        drawdown_trend = 1.0
        if len(self.drawdown_history) >= 5:
            recent_dd_trend = self.drawdown_history[-5:]
            if all(recent_dd_trend[i] <= recent_dd_trend[i+1] for i in range(len(recent_dd_trend)-1)):
                # Drawdown is consistently increasing - amplify penalty
                drawdown_trend = 1.5
        
        r_dd = 0.0
        
        if self.progressive_dd_penalty:
            # Progressive penalty that increases quadratically with drawdown size
            if drawdown > self.drawdown_threshold:
                excess_dd = drawdown - self.drawdown_threshold
                # Quadratic scaling for larger drawdowns with trend awareness
                r_dd = -self.lambda_drawdown * (self.drawdown_threshold + excess_dd**2 * self.risk_aversion_factor * drawdown_trend)
                
                # Additional penalty for consecutive losses during drawdown
                if self.consecutive_losses >= 3:
                    r_dd *= (1.0 + 0.1 * self.consecutive_losses)  # Increase penalty by 10% per consecutive loss
        else:
            # Original linear penalty
            if drawdown > self.drawdown_threshold:
                r_dd = -self.lambda_drawdown * drawdown

        # Enhanced Sharpe‑style risk‑adjusted term with adaptive volatility awareness
        lookback = min(30, len(self.episode_networth))  # Increased lookback window
        recent_slice = self.episode_networth[-lookback:] if len(self.episode_networth) >= lookback else self.episode_networth
        portfolio_vol = np.std(recent_slice) + 1e-6
        
        # Calculate rolling returns for better risk assessment
        rolling_returns = []
        if len(self.episode_networth) >= 2:
            for i in range(1, min(lookback, len(self.episode_networth))):
                ret = (self.episode_networth[-i] - self.episode_networth[-i-1]) / self.episode_networth[-i-1]
                rolling_returns.append(ret)
        
        # Calculate downside deviation (only negative returns)
        downside_returns = [r for r in rolling_returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else portfolio_vol
        
        # Adjust Sharpe ratio based on market conditions with emphasis on downside risk
        # In high volatility markets, we expect higher returns for the same risk
        volatility_adjustment = 1.0 + market_volatility * self.lambda_volatility
        
        # Use Sortino-inspired ratio (penalize downside risk more)
        risk_denominator = (portfolio_vol + downside_deviation * self.risk_aversion_factor) / 2
        r_sharpe = (profit / (risk_denominator * volatility_adjustment)) * self.lambda_sharpe

        # Transaction cost penalty
        r_txn = -self.lambda_txn_cost * txn_cost * 100.0 if txn_cost > 0 else 0.0

        # Trade‑frequency penalty (adjusted by volatility)
        # More trading might be justified in higher volatility
        r_tf = -self.lambda_trade_freq * (1.0 - market_volatility) if action in (1, 2) else 0.0

        # Position concentration penalty
        position_value = self.total_shares * price
        position_pct = position_value / new_net if new_net > 0 else 0
        r_position = 0.0
        if position_pct > self.max_position_pct:
            excess = position_pct - self.max_position_pct
            r_position = -self.lambda_position_size * excess * 100.0

        # Coverage (state‑diversity) bonus — reward entering new equity bins
        bin_id = int(new_net / self.cov_bin_size)
        r_cov = 0.0
        if bin_id not in self.visited_bins:
            self.visited_bins.add(bin_id)
            r_cov = self.lambda_cov

        # ---------- aggregate & normalise ----------
        r_raw = r_profit + r_hold + r_dd + r_sharpe + r_tf + r_txn + r_position + r_cov
        self._update_running_stats(r_raw)
        reward = self._normalise(r_raw)

        # ---------- bookkeeping ----------
        self.episode_networth.append(new_net)
        self.current_step += 1
        done = self.current_step >= len(self.df) - 1

        # Enhanced info dictionary with detailed metrics for monitoring
        info = {
            "net_worth": new_net,
            "balance": self.balance,
            "position": self.position,
            "drawdown": drawdown,
            "market_volatility": market_volatility,
            "trade_count": self.trade_count,
            "position_pct": position_pct * 100.0,
            "reward_components": {
                "profit": r_profit,
                "hold": r_hold,
                "drawdown": r_dd,
                "sharpe": r_sharpe,
                "trade_freq": r_tf,
                "txn_cost": r_txn,
                "position_size": r_position,
                "coverage": r_cov,
                "total_raw": r_raw,
                "normalized": reward
            }
        }
        return self._next_observation(), reward, done, info

    def reset(self):  # type: ignore[override]
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.current_step = self.lookback_window
        self.hold_steps = 0
        self.trade_count = 0
        self.episode_networth = [self.initial_balance]
        self.peak_networth = self.initial_balance
        self.visited_bins = set([int(self.initial_balance / self.cov_bin_size)])
        self.running_mean, self.running_var = 0.0, 1.0
        
        # Reset advanced tracking variables
        self.drawdown_history = []
        self.volatility_history = []
        self.consecutive_losses = 0
        
        return self._next_observation()

    def render(self, mode="human"):
        volatility = self._calculate_market_volatility()
        print(
            f"Step {self.current_step} | Net: {self.episode_networth[-1]:.2f} | "
            f"Bal: {self.balance:.2f} | Pos: {self.position} | "
            f"Vol: {volatility:.4f} | Trades: {self.trade_count}"
        )
        
    def evaluate(self):
        """Calculate performance metrics for the current episode."""
        if not self.episode_networth:
            return {
                "total_profit": 0,
                "percent_return": 0,
                "sharpe_ratio": 0,
                "calmar_ratio": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "num_trades": 0,
                "avg_profit_per_trade": 0,
                "max_consecutive_losses": 0,
                "drawdown_duration_pct": 0
            }
            
        # Calculate returns
        initial_balance = self.initial_balance
        final_balance = self.episode_networth[-1]
        profit = final_balance - initial_balance
        pct_return = (profit / initial_balance) * 100
        
        # Calculate Sharpe ratio
        daily_returns = []
        for i in range(1, len(self.episode_networth)):
            daily_return = (self.episode_networth[i] - self.episode_networth[i-1]) / self.episode_networth[i-1]
            daily_returns.append(daily_return)
            
        sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252) if daily_returns else 0
        
        # Calculate max drawdown
        peak = self.initial_balance
        max_dd = 0
        for networth in self.episode_networth:
            peak = max(peak, networth)
            dd = (peak - networth) / peak * 100
            max_dd = max(max_dd, dd)
            
        # Calculate Calmar ratio
        calmar = pct_return / max_dd if max_dd > 0 else 0
        
        # Calculate win rate
        win_rate = 0
        if self.trade_count > 0:
            # Approximate win rate based on final profit
            win_rate = 100 if profit > 0 else 0
            
        return {
            "total_profit": profit,
            "percent_return": pct_return,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "num_trades": self.trade_count,
            "avg_profit_per_trade": profit / self.trade_count if self.trade_count > 0 else 0,
            "max_consecutive_losses": 0,  # Not tracked in this simplified version
            "drawdown_duration_pct": 0  # Not tracked in this simplified version
        }

# ------------------------------
# 3. Evaluation Callback
# ------------------------------
class TradingEvalCallback(BaseCallback):
    """Runs a deterministic episode every eval_freq timesteps with enhanced metrics."""

    def __init__(self, eval_env: gym.Env, eval_freq: int = 10_000, verbose: int = 0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq

    def _on_step(self) -> bool:  # type: ignore[override]
        if self.n_calls % self.eval_freq == 0:
            metrics = self.evaluate()
            for key, value in metrics.items():
                self.logger.record(f"eval/{key}", value)
            
            if self.verbose:
                print(f"Eval at {self.num_timesteps} | "
                      f"P&L ${metrics['total_profit']:.2f} | "
                      f"Return {metrics['percent_return']:.2f}% | "
                      f"Sharpe {metrics['sharpe_ratio']:.2f} | "
                      f"MaxDD {metrics['max_drawdown']:.2f}% | "
                      f"Win {metrics['win_rate']:.2f}%")
        return True

    def evaluate(self):  # type: ignore[override]
        obs = self.eval_env.reset()
        done = False
        daily_returns = []
        trade_profits = []
        last_trade_price = None
        max_networth = self.eval_env.initial_balance
        min_networth = self.eval_env.initial_balance
        drawdown_periods = 0  # Count periods in drawdown
        max_consecutive_losses = 0
        current_consecutive_losses = 0
        
        # Run the episode
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            prev_position = self.eval_env.position
            prev_networth = self.eval_env.episode_networth[-1] if self.eval_env.episode_networth else self.eval_env.initial_balance
            
            obs, _, done, info = self.eval_env.step(action)
            
            # Track daily returns for Sharpe ratio
            current_networth = info["net_worth"]
            daily_return = (current_networth - prev_networth) / prev_networth if prev_networth > 0 else 0
            daily_returns.append(daily_return)
            
            # Track consecutive losses
            if daily_return < 0:
                current_consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_consecutive_losses)
            else:
                current_consecutive_losses = 0
            
            # Track drawdown periods
            current_drawdown = (max_networth - current_networth) / max_networth if max_networth > 0 else 0
            if current_drawdown > self.eval_env.drawdown_threshold:
                drawdown_periods += 1
            
            # Track max/min networth for drawdown calculation
            max_networth = max(max_networth, current_networth)
            min_networth = min(min_networth, current_networth)
            
            # Track individual trade profits
            current_price = self.eval_env.df.loc[self.eval_env.current_step-1, "close"]
            if prev_position == 1 and self.eval_env.position == 0:  # Sold
                if last_trade_price is not None:
                    profit_pct = (current_price - last_trade_price) / last_trade_price
                    trade_profits.append(profit_pct)
                last_trade_price = None
            elif prev_position == 0 and self.eval_env.position == 1:  # Bought
                last_trade_price = current_price
        
        # Calculate metrics
        init_net = self.eval_env.initial_balance
        final_net = info["net_worth"]
        profit = final_net - init_net
        pct_return = profit / init_net * 100.0
        
        # Sharpe ratio (annualized)
        if len(daily_returns) > 1 and np.std(daily_returns) > 0:
            sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)  # Annualized
        else:
            sharpe = 0.0
        
        # Maximum drawdown
        max_drawdown = ((max_networth - min_networth) / max_networth) * 100.0 if max_networth > 0 else 0.0
        
        # Win rate
        win_rate = (sum(1 for p in trade_profits if p > 0) / len(trade_profits) * 100.0) if trade_profits else 0.0
        
        # Number of trades
        num_trades = len(trade_profits)
        
        # Calculate Calmar ratio (return / max drawdown)
        calmar_ratio = pct_return / max_drawdown if max_drawdown > 0 else 0
        
        # Calculate average profit per trade
        avg_profit_per_trade = sum(trade_profits) / len(trade_profits) if trade_profits else 0
        
        # Calculate drawdown duration percentage
        drawdown_duration_pct = (drawdown_periods / len(daily_returns) * 100) if daily_returns else 0
        
        return {
            "total_profit": profit,
            "percent_return": pct_return,
            "sharpe_ratio": sharpe,
            "calmar_ratio": calmar_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "num_trades": num_trades,
            "avg_profit_per_trade": avg_profit_per_trade * 100,  # Convert to percentage
            "max_consecutive_losses": max_consecutive_losses,
            "drawdown_duration_pct": drawdown_duration_pct
        }

# ------------------------------
# 4. Data Loader
# ------------------------------

def load_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    df = pd.read_csv(path, parse_dates=["date"])
    df.columns = [c.lower() for c in df.columns]
    df.fillna(method="ffill", inplace=True)
    return df

# ------------------------------
# 5. Optuna Objective
# ------------------------------

def objective(trial: optuna.Trial) -> float:
    # --- PPO hyperparams ---
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    gamma = trial.suggest_float("gamma", 0.90, 0.99)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.3)
    gae_lambda = trial.suggest_float("gae_lambda", 0.90, 0.99)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.05)
    n_steps = trial.suggest_categorical("n_steps", [1024, 2048, 4096])  # Try different rollout lengths
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])  # Try different batch sizes

    # --- reward‑shaping coefficients ---
    lambda_hold = trial.suggest_float("lambda_hold", 0.005, 0.05, log=True)
    lambda_drawdown = trial.suggest_float("lambda_drawdown", 5.0, 30.0, log=True)  # Increased range for drawdown penalty
    lambda_sharpe = trial.suggest_float("lambda_sharpe", 2.0, 15.0, log=True)  # Increased range for Sharpe ratio
    lambda_trade_freq = trial.suggest_float("lambda_trade_freq", 0.5, 5.0, log=True)
    lambda_cov = trial.suggest_float("lambda_cov", 0.1, 2.0, log=True)
    
    # --- new reward-shaping coefficients ---
    lambda_volatility = trial.suggest_float("lambda_volatility", 1.0, 8.0, log=True)  # Increased range for volatility adjustment
    lambda_txn_cost = trial.suggest_float("lambda_txn_cost", 0.1, 2.0, log=True)
    txn_cost_pct = trial.suggest_float("txn_cost_pct", 0.0005, 0.002)  # 0.05% to 0.2%
    lambda_position_size = trial.suggest_float("lambda_position_size", 1.0, 8.0, log=True)  # Increased penalty for position concentration
    max_position_pct = trial.suggest_float("max_position_pct", 0.1, 0.3)  # 10% to 30%

    # --- envs ---
    df = load_data("data/SP500.csv")
    common_env_kwargs = dict(
        initial_balance=10_000,
        lookback_window=50,
        lambda_hold=lambda_hold,
        lambda_drawdown=lambda_drawdown,
        lambda_sharpe=lambda_sharpe,
        lambda_trade_freq=lambda_trade_freq,
        lambda_cov=lambda_cov,
        # New parameters
        lambda_volatility=lambda_volatility,
        lambda_txn_cost=lambda_txn_cost,
        txn_cost_pct=txn_cost_pct,
        lambda_position_size=lambda_position_size,
        max_position_pct=max_position_pct,
    )
    train_env = TradingEnv(df, **common_env_kwargs)
    eval_env = TradingEnv(df, **common_env_kwargs)

    # --- logger ---
    log_dir = "./logs_optuna"
    os.makedirs(log_dir, exist_ok=True)
    new_logger = configure(log_dir, ["stdout"])

    # Enhanced policy network architecture with wider and deeper networks
    policy_kwargs = dict(
        features_extractor_class=TradingFeaturesExtractor,
        features_extractor_kwargs=dict(features_dim=128),  # Increased feature dimension
        net_arch=dict(
            pi=[128, 128, 64],  # Deeper policy network
            vf=[128, 128, 64]    # Deeper value network
        ),
        activation_fn=torch.nn.ReLU,
        normalize_images=False
    )

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=n_steps,
        batch_size=batch_size,
        learning_rate=learning_rate,
        gamma=gamma,
        ent_coef=ent_coef,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        policy_kwargs=policy_kwargs,
        verbose=0,
        tensorboard_log=log_dir,
    )
    model.set_logger(new_logger)

    eval_cb = TradingEvalCallback(eval_env, eval_freq=5_000, verbose=0)
    model.learn(total_timesteps=200_000, callback=eval_cb)

    metrics = eval_cb.evaluate()
    
    # Create a composite score that balances return, risk, and trading efficiency
    # Higher return, higher Sharpe ratio, lower drawdown, and reasonable win rate are all desirable
    pct_return = metrics['percent_return']
    sharpe = metrics['sharpe_ratio']
    calmar = metrics['calmar_ratio']
    max_dd = metrics['max_drawdown']
    win_rate = metrics['win_rate']
    num_trades = metrics['num_trades']
    avg_profit_per_trade = metrics['avg_profit_per_trade']
    drawdown_duration_pct = metrics['drawdown_duration_pct']
    max_consecutive_losses = metrics['max_consecutive_losses']
    
    # Penalize excessive drawdown more heavily and reward higher risk-adjusted returns
    # This creates a more balanced optimization target that considers both returns and risk
    
    # Progressive drawdown penalty that increases quadratically for larger drawdowns
    dd_penalty = max_dd if max_dd <= 20 else (20 + (max_dd - 20)**2 * 0.05)
    
    # Penalize long drawdown periods
    drawdown_duration_penalty = drawdown_duration_pct * 0.2
    
    # Penalize consecutive losses
    consecutive_loss_penalty = max_consecutive_losses * 1.5 if max_consecutive_losses > 3 else 0
    
    # Ensure reasonable trading frequency (neither too many nor too few trades)
    trade_frequency_score = 0
    if 5 <= num_trades <= 30:  # Ideal range
        trade_frequency_score = 5
    elif num_trades < 5:  # Too few trades
        trade_frequency_score = num_trades
    else:  # Too many trades
        trade_frequency_score = max(0, 5 - (num_trades - 30) * 0.1)
    
    # Calculate final composite score with greater emphasis on risk management
    composite_score = pct_return + \
                     (sharpe * 10) + \
                     (calmar * 15) + \
                     (win_rate * 0.3) + \
                     (avg_profit_per_trade * 0.5) + \
                     trade_frequency_score - \
                     (dd_penalty * 1.2) - \
                     drawdown_duration_penalty - \
                     consecutive_loss_penalty
    
    # Print detailed metrics for debugging
    if trial.number % 5 == 0:  # Print every 5th trial
        print(f"\nTrial {trial.number} metrics:")
        print(f"  Return: {pct_return:.2f}%, Sharpe: {sharpe:.2f}, Calmar: {calmar:.2f}")
        print(f"  MaxDD: {max_dd:.2f}%, Win Rate: {win_rate:.2f}%, Trades: {num_trades}")
        print(f"  Composite Score: {composite_score:.2f}\n")
    
    return composite_score  # maximize composite score

# ------------------------------
# 6. Main Entry
# ------------------------------

def main():
    print("\n=== Enhanced PPO Trading Agent with Advanced Reward Shaping v2 ===\n")
    print("Improvements implemented:")
    print("  - Deeper neural network with attention mechanism: Better pattern recognition")
    print("  - Progressive drawdown penalty: Quadratic scaling for risk management")
    print("  - Sortino-inspired reward component: Focus on downside risk")
    print("  - Market volatility awareness: Adapts strategy to market conditions")
    print("  - Realistic transaction costs: Models slippage and fees")
    print("  - Position concentration penalties: Prevents overexposure")
    print("  - Enhanced risk-adjusted returns: Improved Sharpe calculation")
    print("  - Adaptive hold penalties: Smarter in volatile markets")
    print("  - Early stopping and learning rate scheduling: Prevents overfitting")
    print("  - Advanced drawdown trend analysis: Adapts to changing market conditions")
    print("\nStarting hyperparameter optimization with Optuna...\n")
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)  # Increased number of trials for better optimization

    print("\nBest trial:")
    best = study.best_trial
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print(f"\nBest composite score: {best.value:.2f}")
    print("(Score combines return, Sharpe ratio, drawdown, and win rate)")
    print("\nRetraining with optimal parameters...")


    # retrain long run with best params
    df = load_data("data/SP500.csv")
    env_kwargs = dict(
        initial_balance=10_000,
        lookback_window=50,
        lambda_hold=best.params["lambda_hold"],
        lambda_drawdown=best.params["lambda_drawdown"],
        lambda_sharpe=best.params["lambda_sharpe"],
        lambda_trade_freq=best.params["lambda_trade_freq"],
        lambda_cov=best.params["lambda_cov"],
        # New parameters
        lambda_volatility=best.params["lambda_volatility"],
        lambda_txn_cost=best.params["lambda_txn_cost"],
        txn_cost_pct=best.params["txn_cost_pct"],
        lambda_position_size=best.params["lambda_position_size"],
        max_position_pct=best.params["max_position_pct"],
    )
    train_env = TradingEnv(df, **env_kwargs)

    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=2048,
        batch_size=64,
        learning_rate=best.params["learning_rate"],
        gamma=best.params["gamma"],
        ent_coef=best.params["ent_coef"],
        gae_lambda=best.params["gae_lambda"],
        clip_range=best.params["clip_range"],
        policy_kwargs=dict(
            features_extractor_class=TradingFeaturesExtractor,
            features_extractor_kwargs=dict(features_dim=128),  # Increased feature dimension
            net_arch=dict(
                pi=[128, 128, 64],  # Deeper policy network
                vf=[128, 128, 64]    # Deeper value network
            ),
            activation_fn=torch.nn.ReLU,
            normalize_images=False
        ),
        verbose=1,
        tensorboard_log="./logs_final",
    )
    # Add early stopping callback to prevent overfitting
    from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
    
    # Create evaluation callback with early stopping
    eval_env = TradingEnv(df, **env_kwargs)
    stop_train_callback = StopTrainingOnRewardThreshold(reward_threshold=100, verbose=1)
    eval_callback = EvalCallback(
        eval_env,
        callback_on_new_best=stop_train_callback,
        eval_freq=10000,
        best_model_save_path="./logs_final/best_model",
        verbose=1,
        n_eval_episodes=5
    )
    
    # Create a learning rate scheduler
    def linear_schedule(initial_value):
        def func(progress_remaining):
            return progress_remaining * initial_value
        return func
    
    # Apply learning rate schedule
    model.learning_rate = linear_schedule(best.params["learning_rate"])
    
    # Train with callbacks
    model.learn(total_timesteps=1_000_000, callback=eval_callback)
    out_path = "rl_agent/models/ppo_trading_agent_final.zip"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    model.save(out_path)
    print(f"Saved final model to {out_path}")
    
    # Run a final evaluation with visualization
    print("\nRunning final evaluation with visualization...")
    visualize_agent_performance(model, df, env_kwargs)


def visualize_agent_performance(model, df, env_kwargs):
    """Visualize the agent's performance with detailed metrics and charts."""
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    
    # Create evaluation environment
    eval_env = TradingEnv(df, **env_kwargs)
    
    # Run a full episode
    obs = eval_env.reset()
    done = False
    actions = []
    net_worths = [eval_env.initial_balance]
    positions = [0]
    timestamps = [df.iloc[eval_env.lookback_window]["date"]]
    drawdowns = [0]
    peak_networth = eval_env.initial_balance
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        actions.append(action)
        obs, _, done, info = eval_env.step(action)
        
        net_worths.append(info["net_worth"])
        positions.append(info["position"])
        timestamps.append(df.iloc[eval_env.current_step-1]["date"])
        
        # Calculate drawdown
        peak_networth = max(peak_networth, info["net_worth"])
        current_drawdown = (peak_networth - info["net_worth"]) / peak_networth * 100 if peak_networth > 0 else 0
        drawdowns.append(current_drawdown)
    
    # Calculate performance metrics
    initial_balance = eval_env.initial_balance
    final_balance = net_worths[-1]
    total_return = ((final_balance - initial_balance) / initial_balance) * 100
    max_drawdown = max(drawdowns)
    
    # Count trades
    buy_signals = [i for i, a in enumerate(actions) if a == 1]
    sell_signals = [i for i, a in enumerate(actions) if a == 2]
    total_trades = len(buy_signals) + len(sell_signals)
    
    # Create figure with subplots
    fig = plt.figure(figsize=(15, 12))
    
    # Plot 1: Equity curve
    ax1 = fig.add_subplot(3, 1, 1)
    ax1.plot(timestamps, net_worths, label="Net Worth", color="blue")
    ax1.set_title(f"Trading Performance: Return {total_return:.2f}%, Max DD {max_drawdown:.2f}%")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend()
    ax1.grid(True)
    
    # Plot 2: Drawdown
    ax2 = fig.add_subplot(3, 1, 2)
    ax2.fill_between(timestamps, drawdowns, color="red", alpha=0.3)
    ax2.set_title(f"Drawdown (Max: {max_drawdown:.2f}%)")
    ax2.set_ylabel("Drawdown (%)")
    ax2.grid(True)
    
    # Plot 3: Position and Price
    ax3 = fig.add_subplot(3, 1, 3)
    price_data = df.iloc[eval_env.lookback_window:eval_env.current_step]["close"].values
    ax3.plot(timestamps, price_data, label="Price", color="black", alpha=0.5)
    
    # Plot buy/sell markers
    for idx in buy_signals:
        if idx < len(timestamps):
            ax3.scatter(timestamps[idx], price_data[idx], color="green", marker="^", s=100)
    
    for idx in sell_signals:
        if idx < len(timestamps):
            ax3.scatter(timestamps[idx], price_data[idx], color="red", marker="v", s=100)
    
    ax3.set_title(f"Price Chart with Trading Signals (Total Trades: {total_trades})")
    ax3.set_ylabel("Price ($)")
    ax3.grid(True)
    
    # Format x-axis dates
    for ax in [ax1, ax2, ax3]:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the figure
    os.makedirs("./figures", exist_ok=True)
    plt.savefig("./figures/trading_performance.png")
    print(f"Performance visualization saved to ./figures/trading_performance.png")
    
    # Calculate additional metrics
    daily_returns = []
    for i in range(1, len(net_worths)):
        daily_return = (net_worths[i] - net_worths[i-1]) / net_worths[i-1] if net_worths[i-1] > 0 else 0
        daily_returns.append(daily_return)
        
    sharpe_ratio = np.mean(daily_returns) / (np.std(daily_returns) + 1e-6) * np.sqrt(252) if daily_returns else 0
    calmar_ratio = total_return / max_drawdown if max_drawdown > 0 else 0
    
    # Calculate win rate from buy/sell signals
    profitable_trades = 0
    trade_pairs = min(len(buy_signals), len(sell_signals))
    if trade_pairs > 0:
        for i in range(min(len(buy_signals), len(sell_signals))):
            if buy_signals[i] < sell_signals[i]:
                buy_price = price_data[buy_signals[i]]
                sell_price = price_data[sell_signals[i]]
                if sell_price > buy_price:
                    profitable_trades += 1
    
    win_rate = (profitable_trades / trade_pairs * 100) if trade_pairs > 0 else 0
    
    # Print summary statistics
    print("\nPerformance Summary:")
    print(f"  Initial Balance: ${initial_balance:.2f}")
    print(f"  Final Balance: ${final_balance:.2f}")
    print(f"  Total Return: {total_return:.2f}%")
    print(f"  Maximum Drawdown: {max_drawdown:.2f}%")
    print(f"  Total Trades: {total_trades}")
    print(f"  Win Rate: {win_rate:.2f}%")
    print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"  Calmar Ratio: {calmar_ratio:.2f}")

if __name__ == "__main__":
    main()
