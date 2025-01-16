import numpy as np
import pandas as pd
from gym import Env
from gym.spaces import Discrete, Box


class TradingEnv(Env):
    """
    A trading environment for a single asset with simplified buy/sell/hold mechanics.
    Reward includes immediate profit/loss and harsher penalties for poor actions.
    """

    def __init__(self, df, initial_balance=10000, lookback_window=50):
        super(TradingEnv, self).__init__()

        self.df = df.reset_index(drop=True)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.position = 0         # 1 = holding, 0 = not holding
        self.total_shares = 0
        self.current_step = lookback_window
        self.lookback_window = lookback_window
        self.hold_steps = 0       # Track consecutive HOLD actions

        # Actions: 0 = HOLD, 1 = BUY, 2 = SELL
        self.action_space = Discrete(3)

        # Observations: [lookback_window, (all features - 'date') + 2 for balance, position]
        num_features = len(self.df.columns) - 1  # excluding 'date'
        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(lookback_window, num_features + 2),
            dtype=np.float32
        )

    def _next_observation(self):
        """
        Returns the observation for the current step:
        The past 'lookback_window' rows of features + appended [balance, position].
        """
        start = self.current_step - self.lookback_window
        end = self.current_step
        obs_df = self.df.iloc[start:end].drop(columns=['date'], errors='ignore')
        obs = obs_df.values  # shape = (lookback_window, number_of_features)

        # Append balance, position to each row
        extra = np.array([[self.balance, self.position]] * self.lookback_window)
        obs = np.hstack((obs, extra))
        return obs.astype(np.float32)

    def step(self, action):
        """
        Execute the action and return next_obs, reward, done, info.
        """
        current_price = self.df.loc[self.current_step, 'close']
        previous_value = self.balance + self.total_shares * current_price

        # Execute the action
        if action == 1:  # BUY
            # Can only buy if we have enough balance
            if self.balance >= current_price:
                self.position = 1
                self.total_shares += 1
                self.balance -= current_price

        elif action == 2:  # SELL
            # Only sell if currently holding shares
            if self.position == 1 and self.total_shares > 0:
                self.position = 0
                self.total_shares -= 1
                self.balance += current_price
            else:
                # Invalid SELL: harsher penalty
                pass

        # Update net worth and calculate immediate profit/loss
        new_value = self.balance + self.total_shares * current_price
        profit_loss = new_value - previous_value

        # Reward Calculation
        reward = profit_loss
        if action == 0:  # HOLD
            self.hold_steps += 1
            reward -= 0.02 * self.hold_steps  # Increase penalty for consecutive HOLD actions
        elif action == 2:  # SELL
            if self.position == 0 and self.total_shares == 0:
                reward -= 0.1  # Harsher penalty for invalid SELL
        else:
            self.hold_steps = 0  # Reset HOLD step counter after BUY/SELL

        # Scale rewards
        reward *= 100.0  # Amplify meaningful signals
        if reward < 0:
            reward *= 2.0  # Double penalty for losses to encourage better decisions

        self.current_step += 1
        done = (self.current_step >= len(self.df) - 1)

        obs = self._next_observation()
        info = {
            'portfolio_value': new_value,
            'balance': self.balance,
            'position': self.position
        }
        return obs, reward, done, info

    def reset(self):
        """
        Resets the environment to initial conditions.
        """
        self.balance = self.initial_balance
        self.position = 0
        self.total_shares = 0
        self.current_step = self.lookback_window
        self.hold_steps = 0
        return self._next_observation()

    def render(self, mode='human'):
        print(f"Step: {self.current_step}, Balance: {self.balance}, Position: {self.position}")
