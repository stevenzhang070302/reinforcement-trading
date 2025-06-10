"""
Reinforcement Learning-Based Trading Agent using an Enhanced Dueling DQN.
Incorporates advanced feature engineering, risk-adjusted reward, improved network architecture,
and prioritized experience replay.

Author: Your Name
Date: YYYY-MM-DD
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import os
import warnings
import optuna
from collections import namedtuple
from optuna.trial import TrialState

# ------------------------------
# 0. Global Settings & Config
# ------------------------------
warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

close_prices = None
TOTAL_TRIALS = 20  # For hyperparameter tuning

# ------------------------------
# 1. Data Preparation
# ------------------------------
def load_data(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}.")
    dataset = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print("Dataset Head:")
    print(dataset.head())
    if dataset.isnull().values.any():
        dataset.fillna(method='ffill', inplace=True)
    # Assume columns are in lowercase already from feature engineering
    close_prices = dataset['close'].values.astype(float)
    print(f"Total data points: {len(close_prices)}")
    # Also return the full dataframe for extended state representations
    return close_prices, dataset

# ------------------------------
# 2. Define Enhanced DQN Networks (Dueling Architecture)
# ------------------------------
class DuelingDQN(nn.Module):
    """Implements Dueling Network Architecture"""
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.value_fc = nn.Linear(32, 1)
        self.advantage_fc = nn.Linear(32, output_dim)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values

# ------------------------------
# 3. Define Prioritized Replay Buffer
# ------------------------------
Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

class PrioritizedReplayBuffer:
    """Implements proportional prioritization experience replay"""
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.pos = 0
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def add(self, experience):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == 0:
            return None, None, None
        prios = self.priorities[:len(self.buffer)] ** self.alpha
        probs = prios / prios.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[idx] for idx in indices]
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        return experiences, indices, np.array(weights, dtype=np.float32)
    
    def update_priorities(self, indices, priorities):
        for idx, prio in zip(indices, priorities):
            self.priorities[idx] = prio
    
    def __len__(self):
        return len(self.buffer)

# ------------------------------
# 4. Helper Functions
# ------------------------------
def format_price(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_rsi(data, t, n):
    if t < 1:
        return 50
    delta = data[max(t - n + 1, 0):t + 1]
    delta = np.diff(delta)
    up = delta.clip(min=0)
    down = -1 * delta.clip(max=0)
    avg_gain = up.mean() if len(up) > 0 else 0
    avg_loss = down.mean() if len(down) > 0 else 0
    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def get_state(data, t, n):
    """
    Basic state from close prices.
    """
    d = t - n + 1
    if d < 0:
        block = [data[0]] * (-d) + list(data[0:t + 1])
    else:
        block = data[d:t + 1]
    res = [sigmoid(block[i + 1] - block[i]) for i in range(n - 1)]
    ma = np.mean(block)
    rsi = compute_rsi(data, t, n)
    return np.array([res + [ma, rsi]])

def get_extended_state(df, t, window):
    """
    Extended state representation using feature-engineered columns.
    This example uses: close, high, low, open, bollinger_high, bollinger_low, bollinger_width, obv, roc.
    You can expand or adjust this list as needed.
    """
    # Assume df has columns in lowercase
    features = ['close', 'high', 'low', 'open', 'bollinger_high', 'bollinger_low', 'bollinger_width', 'obv', 'roc']
    window_df = df.iloc[max(0, t-window+1):t+1][features]
    state = window_df.values.flatten()
    return np.array([state])

def plot_behavior(data_input, states_buy, states_sell, profit, percentage_change):
    plt.figure(figsize=(15,5))
    plt.plot(data_input, color='r', lw=2., label='Price')
    plt.plot(states_buy, [data_input[i] for i in states_buy], '^', markersize=10, color='g', label='Buy')
    plt.plot(states_sell, [data_input[i] for i in states_sell], 'v', markersize=10, color='b', label='Sell')
    plt.title(f'Total Profit: {format_price(profit)} | Percentage Change: {percentage_change:.2f}%')
    plt.legend()
    plt.show()

# ------------------------------
# 5. Enhanced Agent Class
# ------------------------------
class EnhancedAgent:
    def __init__(
        self, 
        state_size, 
        action_size, 
        device, 
        model_path=None, 
        is_eval=False, 
        learning_rate=0.0001, 
        gamma=0.95, 
        epsilon=1.0, 
        epsilon_min=0.01, 
        epsilon_decay=0.995,
        tau=0.001
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.is_eval = is_eval
        self.tau = tau

        self.model = DuelingDQN(state_size, action_size).to(device)
        self.target_model = DuelingDQN(state_size, action_size).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        
        if torch.cuda.device_count() > 1 and not self.is_eval:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            self.model = nn.DataParallel(self.model)
            self.target_model = nn.DataParallel(self.target_model)

        self.memory = PrioritizedReplayBuffer(10000)
        self.inventory = []
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        if not self.is_eval:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.SmoothL1Loss()
        if self.is_eval and model_path:
            self.load_model(model_path)

    def load_model(self, model_path):
        if os.path.exists(model_path):
            state_dict = torch.load(model_path, map_location=self.device)
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k.replace("module.", "") if k.startswith("module.") else k
                new_state_dict[new_key] = v
            self.model.load_state_dict(new_state_dict)
            self.model.eval()
            print(f"Loaded model from {model_path}")
        else:
            raise FileNotFoundError(f"Model file not found at {model_path}")

    def soft_update_target(self):
        for target_param, model_param in zip(self.target_model.parameters(), self.model.parameters()):
            target_param.data.copy_(self.tau * model_param.data + (1.0 - self.tau) * target_param.data)

    def act(self, state):
        if not self.is_eval and random.random() <= self.epsilon:
            return random.randrange(self.action_size)
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        exp = Experience(state, action, reward, next_state, done)
        self.memory.add(exp)

    def replay(self, batch_size, beta=0.4):
        if len(self.memory) < batch_size:
            return
        experiences, indices, weights = self.memory.sample(batch_size, beta=beta)
        if experiences is None:
            return
        states = torch.FloatTensor([e[0][0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3][0] for e in experiences]).to(self.device)
        dones = torch.FloatTensor([e[4] for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)

        current_q = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.model(next_states).max(1)[1]
            next_q = self.target_model(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)
        td_errors = current_q - target_q
        loss = (td_errors.pow(2) * weights).mean()

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()

        new_priorities = (td_errors.abs().cpu().detach().numpy() + 1e-6)
        self.memory.update_priorities(indices, new_priorities)

        self.soft_update_target()

        if not self.is_eval and self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# ------------------------------
# 6. Training Function with Improved Protocol
# ------------------------------
def train_agent(agent, X_train, df_features, window_size, batch_size, episode_count):
    last_saved_episode = None
    for episode in range(episode_count + 1):
        print(f"Running Experiment {episode}/{episode_count}")
        # Choose state representation: basic or extended.
        # For extended state, use the feature-engineered dataframe.
        state = get_extended_state(df_features, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        
        for t in range(len(X_train) - 1):
            action = agent.act(state)
            next_state = get_extended_state(df_features, t + 1, window_size + 1)
            
            reward = 0
            if action == 1:  # Buy
                agent.inventory.append(X_train[t])
            elif action == 2 and len(agent.inventory) > 0:  # Sell
                bought_price = agent.inventory.pop(0)
                profit = X_train[t] - bought_price
                reward = profit
                total_profit += profit
            else:
                reward = -0.01  # Penalty for holding

            done = (t == len(X_train) - 2)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            agent.replay(batch_size)
        
        print(f"Experiment {episode}/{episode_count} | Profit: {format_price(total_profit)} | ε: {agent.epsilon:.4f}")
        if episode % 10 == 0:
            save_path = f"enhanced_model_ep{episode}.pth"
            torch.save(agent.model.state_dict(), save_path)
            last_saved_episode = episode

    if last_saved_episode != episode_count:
        print(f"Final Experiment {episode_count}/{episode_count} | Saving final model.")
        final_save_path = f"enhanced_model_ep{episode_count}.pth"
        torch.save(agent.model.state_dict(), final_save_path)

# ------------------------------
# 7. Evaluation Function
# ------------------------------
def evaluate_agent(agent, X_test, df_features, window_size):
    initial_cash = 10000
    cash = initial_cash
    total_profit = 0
    agent.inventory = []
    state = get_extended_state(df_features, 0, window_size + 1)
    
    for t in range(len(X_test) - 1):
        action = agent.act(state)
        next_state = get_extended_state(df_features, t + 1, window_size + 1)
        current_price = X_test[t]
        if action == 1 and cash >= current_price:
            agent.inventory.append(current_price)
            cash -= current_price
        elif action == 2 and agent.inventory:
            bought_price = agent.inventory.pop(0)
            cash += current_price
            profit = current_price - bought_price
            total_profit += profit
        agent.remember(state, action, 0, next_state, t == len(X_test) - 2)
        state = next_state

    while agent.inventory:
        bought_price = agent.inventory.pop(0)
        last_price = X_test[-1]
        cash += last_price
        profit = last_price - bought_price
        total_profit += profit

    percentage_change = ((cash - initial_cash) / initial_cash) * 100
    print(f"Total Profit: {format_price(total_profit)} | Final Cash: {format_price(cash)} | Percentage Change: {percentage_change:.2f}%")
    return total_profit, percentage_change

# ------------------------------
# 8. Hyperparameter Tuning (Optuna Objective)
# ------------------------------
def objective(trial):
    tau = trial.suggest_uniform('tau', 0.001, 0.1)
    per_alpha = trial.suggest_uniform('per_alpha', 0.4, 0.8)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.90, 0.9999)
    batch_size = 512
    episode_count = trial.suggest_int('episode_count', 5, 100)
    window_size = trial.suggest_int('window_size', 5, 20)
    
    print(f"Running experiment {trial.number + 1}/{TOTAL_TRIALS}")
    VALIDATION_SIZE = 0.2
    TRAIN_SIZE = int(len(close_prices) * (1 - VALIDATION_SIZE))
    X_train = close_prices[:TRAIN_SIZE]
    X_test = close_prices[TRAIN_SIZE:]
    
    # Load the feature-engineered dataframe for extended state.
    # Assume the feature CSV is saved at "../data_preprocessing/SPY_day_features.csv"
    features_path = os.path.join("..", "data_preprocessing", "SPY_day_features.csv")
    df_features = pd.read_csv(features_path, parse_dates=['date'])
    df_features.columns = [col.lower() for col in df_features.columns]
    
    agent = EnhancedAgent(
        state_size=window_size * len(['close', 'high', 'low', 'open', 'bollinger_high', 
                                       'bollinger_low', 'bollinger_width', 'obv', 'roc']),
        action_size=3,
        device=DEVICE,
        is_eval=False,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay,
        tau=tau
    )
    agent.memory.alpha = per_alpha
    train_agent(agent, X_train, df_features, window_size, batch_size, episode_count)
    
    model_name = f"enhanced_model_ep{episode_count}.pth"
    if not os.path.exists(model_name):
        return -float('inf')
    
    agent_eval = EnhancedAgent(
        state_size=window_size * len(['close', 'high', 'low', 'open', 'bollinger_high', 
                                       'bollinger_low', 'bollinger_width', 'obv', 'roc']),
        action_size=3,
        device=DEVICE,
        model_path=model_name,
        is_eval=True,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=0.0,
        epsilon_min=0.0,
        epsilon_decay=epsilon_decay,
        tau=tau
    )
    
    total_profit, percentage_change = evaluate_agent(agent_eval, X_test, df_features, window_size)
    os.remove(model_name)
    return percentage_change

def run_hyperparameter_tuning():
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=TOTAL_TRIALS)
    print("\nNumber of finished trials:", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params:")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# ------------------------------
# 9. Main Function
# ------------------------------
def main():
    global close_prices
    DATA_PATH = 'data/SP500.csv'  # Update with your actual CSV path
    close_prices, _ = load_data(DATA_PATH)
    
    perform_tuning = True  # Set to False for standard training/testing
    
    if perform_tuning:
        run_hyperparameter_tuning()
    else:
        WINDOW_SIZE = 10
        df_features = pd.read_csv(os.path.join("..", "data_preprocessing", "SPY_day_features.csv"), parse_dates=['date'])
        df_features.columns = [col.lower() for col in df_features.columns]
        ACTION_SIZE = 3
        BATCH_SIZE = 64
        EPISODE_COUNT = 100
        VALIDATION_SIZE = 0.2
        TRAIN_SIZE = int(len(close_prices) * (1 - VALIDATION_SIZE))
        X_train = close_prices[:TRAIN_SIZE]
        X_test = close_prices[TRAIN_SIZE:]
        
        agent = EnhancedAgent(
            state_size=WINDOW_SIZE * len(['close', 'high', 'low', 'open', 'bollinger_high', 
                                           'bollinger_low', 'bollinger_width', 'obv', 'roc']),
            action_size=ACTION_SIZE,
            device=DEVICE,
            is_eval=False,
            learning_rate=0.0001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995,
            tau=0.005
        )
        train_agent(agent, X_train, df_features, WINDOW_SIZE, BATCH_SIZE, EPISODE_COUNT)
        
        model_path = f"enhanced_model_ep{EPISODE_COUNT}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}.")
        
        agent_eval = EnhancedAgent(
            state_size=WINDOW_SIZE * len(['close', 'high', 'low', 'open', 'bollinger_high', 
                                           'bollinger_low', 'bollinger_width', 'obv', 'roc']),
            action_size=ACTION_SIZE,
            device=DEVICE,
            model_path=model_path,
            is_eval=True,
            learning_rate=0.0001,
            gamma=0.95,
            epsilon=0.0,
            epsilon_min=0.0,
            epsilon_decay=0.995,
            tau=0.005
        )
        evaluate_agent(agent_eval, X_test, df_features, WINDOW_SIZE)

if __name__ == "__main__":
    main()
