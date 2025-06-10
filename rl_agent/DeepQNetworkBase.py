"""
Reinforcement Learning-Based Trading Agent using PyTorch with Enhanced Logging and Trading Strategies,
including Hyperparameter Tuning via Optuna and Multiple-GPU Support with DataParallel.

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
from collections import deque
import random
import os
import warnings
import optuna
from optuna.trial import TrialState

# ------------------------------
# 0. Global Settings & Config
# ------------------------------

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# This global variable will be assigned in main() so that Optuna's objective can access the dataset
close_prices = None

# ------------------------------
# 1. Data Preparation
# ------------------------------

def load_data(file_path):
    """
    Loads and preprocesses the stock data.

    Parameters:
        file_path (str): Path to the CSV file containing stock data.

    Returns:
        np.array: Array of closing prices.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at {file_path}. Please ensure the file exists.")
    
    dataset = pd.read_csv(file_path, index_col=0, parse_dates=True)
    print("Dataset Head:")
    print(dataset.head())
    
    # Check for null values
    if dataset.isnull().values.any():
        print("Null values found. Filling missing values using forward fill.")
        dataset.fillna(method='ffill', inplace=True)
    else:
        print("No null values found.")
    
    # Focus on the 'Close' price
    close_prices = dataset['Close'].values.astype(float)
    print(f"Total data points: {len(close_prices)}")
    
    return close_prices

# ------------------------------
# 2. Define the DQN Network
# ------------------------------

class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initializes the Deep Q-Network.

        Parameters:
            input_dim (int): Dimension of the input state.
            output_dim (int): Number of possible actions.
        """
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, output_dim)
        
    def forward(self, x):
        """
        Forward pass of the network.

        Parameters:
            x (torch.Tensor): Input state.

        Returns:
            torch.Tensor: Q-values for each action.
        """
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        return out

# ------------------------------
# 3. Define the Agent Class (DataParallel-compatible)
# ------------------------------

class Agent:
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
        epsilon_decay=0.995
    ):
        """
        Initializes the DQN agent.

        Parameters:
            state_size (int): Dimension of the input state.
            action_size (int): Number of possible actions.
            device (torch.device): Device to run computations on.
            model_path (str, optional): Path to a saved model (for evaluation mode).
            is_eval (bool, optional): Flag indicating evaluation mode.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor.
            epsilon (float): Initial epsilon for exploration.
            epsilon_min (float): Minimum epsilon after decay.
            epsilon_decay (float): Decay rate of epsilon.
        """
        self.state_size = state_size
        self.action_size = action_size
        self.device = device
        self.is_eval = is_eval

        self.memory = deque(maxlen=10000)
        self.inventory = []

        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Create the network
        self.model = DQN(state_size, action_size).to(self.device)
        
        # Use DataParallel if multiple GPUs are available and not in eval mode
        if torch.cuda.device_count() > 1 and not self.is_eval:
            print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
            self.model = nn.DataParallel(self.model)
        
        # If in evaluation mode, load the trained model
        if self.is_eval and model_path:
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found at {model_path}.")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print(f"Loaded model from {model_path}")
        # Otherwise, set up optimizer and loss
        elif not self.is_eval:
            self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
            self.criterion = nn.MSELoss()

    def act(self, state):
        """
        Chooses an action based on the current state using epsilon-greedy.

        Parameters:
            state (np.array): Current state.

        Returns:
            int: Selected action (0: Hold, 1: Buy, 2: Sell).
        """
        if (not self.is_eval) and (random.random() <= self.epsilon):
            return random.randrange(self.action_size)
        
        state = torch.FloatTensor(state).to(self.device)
        
        with torch.no_grad():
            q_values = self.model(state)
        
        return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        """
        Stores the experience in the replay memory.

        Parameters:
            state (np.array): Current state.
            action (int): Action taken.
            reward (float): Reward received.
            next_state (np.array): Next state.
            done (bool): Flag indicating end of episode.
        """
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        """
        Samples a batch of experiences from memory and trains the network.

        Parameters:
            batch_size (int): Number of experiences to sample for each training step.
        """
        if len(self.memory) < batch_size:
            return

        mini_batch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor([experience[0][0] for experience in mini_batch]).to(self.device)
        actions = torch.LongTensor([experience[1] for experience in mini_batch]).to(self.device)
        rewards = torch.FloatTensor([experience[2] for experience in mini_batch]).to(self.device)
        next_states = torch.FloatTensor([experience[3][0] for experience in mini_batch]).to(self.device)
        dones = torch.FloatTensor([experience[4] for experience in mini_batch]).to(self.device)

        # Current Q values
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values
        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * max_next_q_values * (1 - dones))
        
        # Compute loss
        loss = self.criterion(current_q_values, target_q_values)

        # Backprop and optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon
        if (not self.is_eval) and (self.epsilon > self.epsilon_min):
            self.epsilon *= self.epsilon_decay

# ------------------------------
# 4. Helper Functions
# ------------------------------

def format_price(n):
    """
    Formats a numerical value into a price string.
    """
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))

def sigmoid(x):
    """
    Applies the sigmoid function.
    """
    return 1 / (1 + np.exp(-x))

def compute_rsi(data, t, n):
    """
    Computes the Relative Strength Index (RSI) for a given time step.
    """
    if t < 1:
        return 50  # Neutral RSI for the first data point
    
    delta = data[max(t - n + 1, 0):t + 1]
    delta = np.diff(delta)
    up = delta.clip(min=0)
    down = -1 * delta.clip(max=0)

    avg_gain = up.mean() if len(up) > 0 else 0
    avg_loss = down.mean() if len(down) > 0 else 0

    if avg_loss == 0:
        return 100
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def get_state(data, t, n):
    """
    Generates the state representation for the agent by looking back 'n' steps
    and combining price differences (sigmoid), moving average, and RSI.
    """
    d = t - n + 1
    if d < 0:
        block = [data[0]] * (-d) + list(data[0:t + 1])
    else:
        block = data[d:t + 1]
    
    # Price differences normalized through sigmoid
    res = [sigmoid(block[i + 1] - block[i]) for i in range(n - 1)]
    
    # Technical indicators
    ma = np.mean(block)  # Moving Average
    rsi = compute_rsi(data, t, n)
    
    # Combine into a single state vector
    state = np.array([res + [ma, rsi]])
    return state

def plot_behavior(data_input, states_buy, states_sell, profit, percentage_change):
    """
    Plots the stock price along with buy and sell signals.
    """
    plt.figure(figsize=(15,5))
    plt.plot(data_input, color='r', lw=2., label='Price')
    plt.plot(states_buy, [data_input[i] for i in states_buy], '^', markersize=10, color='g', label='Buy')
    plt.plot(states_sell, [data_input[i] for i in states_sell], 'v', markersize=10, color='b', label='Sell')
    plt.title(f'Total Profit: {format_price(profit)} | Percentage Change: {percentage_change:.2f}%')
    plt.legend()
    plt.show()

# ------------------------------
# 5. Training the Agent
# ------------------------------

def train_agent(agent, X_train, window_size, batch_size, episode_count):
    """
    Trains the DQN agent over multiple episodes on the training set.
    """
    for episode in range(episode_count + 1):
        print(f"\nRunning Episode {episode}/{episode_count}")
        state = get_state(X_train, 0, window_size + 1)
        total_profit = 0
        agent.inventory = []
        total_rewards = 0
        
        states_buy = []
        states_sell = []
        
        for t in range(len(X_train) - 1):
            action = agent.act(state)
            next_state = get_state(X_train, t + 1, window_size + 1)
            
            reward = 0
            # BUY
            if action == 1:
                agent.inventory.append(X_train[t])
                states_buy.append(t)
            
            # SELL
            elif action == 2 and len(agent.inventory) > 0:
                bought_price = agent.inventory.pop(0)
                profit = X_train[t] - bought_price
                reward = profit
                total_profit += profit
                states_sell.append(t)

            total_rewards += reward
            done = (t == len(X_train) - 2)  # last iteration
            agent.remember(state, action, reward, next_state, done)
            
            state = next_state

            if done:
                avg_reward = total_rewards / (len(X_train) - 1)
                print("--------------------------------")
                print(f"Episode {episode} | Total Profit: {format_price(total_profit)} | Average Reward: {avg_reward:.4f}")
                print("--------------------------------")
                # Optionally, you can plot intermediate results here.
            
            # Perform training via replay
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)
        
        # Save model checkpoint at intervals
        if episode % 10 == 0 or episode == episode_count:
            model_path = f"model_ep{episode}.pth"
            # If using DataParallel, save the underlying module
            if isinstance(agent.model, nn.DataParallel):
                torch.save(agent.model.module.state_dict(), model_path)
            else:
                torch.save(agent.model.state_dict(), model_path)
            print(f"Model saved to {model_path}")

# ------------------------------
# 6. Evaluate & Test the Agent
# ------------------------------

def evaluate_agent(agent, X_test, window_size):
    """
    Evaluates the trained agent on the test dataset and returns performance metrics.
    This is used by the Optuna objective for hyperparameter tuning.
    
    Parameters:
        agent (Agent): The trained agent in evaluation mode.
        X_test (list or np.array): Test data (prices).
        window_size (int): Window size for state representation.
    
    Returns:
        float, float: (total_profit, percentage_change)
    """
    initial_cash = 10000
    cash = initial_cash
    total_profit = 0
    agent.inventory = []

    state = get_state(X_test, 0, window_size + 1)
    
    for t in range(len(X_test) - 1):
        action = agent.act(state)
        next_state = get_state(X_test, t + 1, window_size + 1)
        reward = 0
        
        current_price = X_test[t]
        
        # BUY
        if action == 1:
            if cash >= current_price:
                agent.inventory.append(current_price)
                cash -= current_price
        
        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            cash += current_price
            profit = current_price - bought_price
            reward = profit
            total_profit += profit
        
        total_profit += reward
        done = (t == len(X_test) - 2)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
    
    # Sell remaining inventory at the final price
    while agent.inventory:
        bought_price = agent.inventory.pop(0)
        last_price = X_test[-1]
        cash += last_price
        profit = last_price - bought_price
        total_profit += profit

    percentage_change = ((cash - initial_cash) / initial_cash) * 100
    return total_profit, percentage_change


def test_agent(agent, X_test, window_size):
    """
    Tests the trained agent on the test dataset with detailed logging and final plotting.
    This function is more verbose than 'evaluate_agent'.
    """
    print("\n--- Starting Evaluation ---\n")
    initial_cash = 10000
    cash = initial_cash
    total_profit = 0
    agent.inventory = []
    action_counts = {0: 0, 1: 0, 2: 0}

    states_buy_test = []
    states_sell_test = []

    state = get_state(X_test, 0, window_size + 1)
    
    for t in range(len(X_test) - 1):
        action = agent.act(state)
        action_counts[action] += 1
        next_state = get_state(X_test, t + 1, window_size + 1)
        
        reward = 0
        current_price = X_test[t]
        
        # BUY
        if action == 1:
            if cash >= current_price:
                agent.inventory.append(current_price)
                cash -= current_price
                states_buy_test.append(t)
                print(f"t={t}: Buy at {format_price(current_price)} | Cash: {format_price(cash)}")
            else:
                print(f"t={t}: Buy signal but insufficient cash: {format_price(cash)}")

        # SELL
        elif action == 2 and len(agent.inventory) > 0:
            bought_price = agent.inventory.pop(0)
            cash += current_price
            profit = current_price - bought_price
            reward = profit
            total_profit += profit
            states_sell_test.append(t)
            print(f"t={t}: Sell at {format_price(current_price)} | Profit: {format_price(profit)} | Cash: {format_price(cash)}")
        
        total_profit += reward
        done = (t == len(X_test) - 2)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        
        if done:
            percentage_change = ((cash - initial_cash) / initial_cash) * 100
            print("\n----------------------------------------")
            print(f"Total Profit so far: {format_price(total_profit)}")
            print(f"Final Cash: {format_price(cash)}")
            print(f"Percentage Change: {percentage_change:.2f}%")
            print("----------------------------------------\n")

    # Sell any remaining inventory at the final price
    while agent.inventory:
        bought_price = agent.inventory.pop(0)
        last_price = X_test[-1]
        cash += last_price
        profit = last_price - bought_price
        total_profit += profit
        states_sell_test.append(len(X_test) - 1)
        print(f"Final Sell: at {format_price(last_price)} | Profit: {format_price(profit)} | Cash: {format_price(cash)}")

    # Final metrics
    percentage_change = ((cash - initial_cash) / initial_cash) * 100
    print("\n------------------------------------------")
    print(f"Total Profit after Final Sell: {format_price(total_profit)}")
    print(f"Final Cash after Final Sell: {format_price(cash)}")
    print(f"Percentage Change: {percentage_change:.2f}%")
    print("------------------------------------------\n")
    
    print(f"Action Distribution: Hold={action_counts[0]}, Buy={action_counts[1]}, Sell={action_counts[2]}")

    # Plot the results
    plot_behavior(X_test, states_buy_test, states_sell_test, total_profit, percentage_change)


# ------------------------------
# 7. Hyperparameter Tuning (Optuna)
# ------------------------------

def objective(trial):
    """
    The objective function for Optuna hyperparameter tuning.
    Uses the global close_prices data, trains a new agent for each trial, 
    and returns the percentage change (or total profit) as the metric to maximize.
    """
    # Suggest hyperparameters
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)
    gamma = trial.suggest_uniform('gamma', 0.90, 0.99)
    epsilon_decay = trial.suggest_uniform('epsilon_decay', 0.90, 0.9999)
    batch_size =  512
    # batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
    # batch_size = 4096 * 2
    
    # Force episode_count to be at most 100
    episode_count = trial.suggest_int('episode_count', 1, 100)

    # Force window_size to be between 5 and 20, as before
    window_size = trial.suggest_int('window_size', 5, 20)
    
    # Split data
    VALIDATION_SIZE = 0.2
    TRAIN_SIZE = int(len(close_prices) * (1 - VALIDATION_SIZE))
    X_train, X_test = close_prices[:TRAIN_SIZE], close_prices[TRAIN_SIZE:]
    
    # Create agent
    agent = Agent(
        state_size=window_size + 2,  # +2 for MA & RSI
        action_size=3,              # 0: Hold, 1: Buy, 2: Sell
        device=DEVICE,
        is_eval=False,
        learning_rate=learning_rate,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=epsilon_decay
    )
    
    # Train the agent
    train_agent(agent, X_train, window_size, batch_size, episode_count)
    
    # Evaluate the agent
    model_name = f"model_ep{episode_count}.pth"
    if not os.path.exists(model_name):
        # If the model wasn't saved, return a very low score to penalize
        return -float('inf')
    
    # Create an evaluation agent
    agent_eval = Agent(
        state_size=window_size + 2,
        action_size=3,
        device=DEVICE,
        model_path=model_name,
        is_eval=True,
        learning_rate=learning_rate,  # not used in eval, but we keep the parameter for consistency
        gamma=gamma,
        epsilon=0.0,   # no exploration in evaluation
        epsilon_min=0.0,
        epsilon_decay=epsilon_decay
    )
    
    # Evaluate and retrieve metrics
    total_profit, percentage_change = evaluate_agent(agent_eval, X_test, window_size)
    
    # Clean up model file to save space
    os.remove(model_name)
    
    # Optuna will maximize the returned value
    return percentage_change  # or total_profit, if you prefer

def run_hyperparameter_tuning():
    """
    Sets up an Optuna study to find the best hyperparameters for the trading agent.
    """
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)  # Increase n_trials for a more thorough search

    print("\nNumber of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

# ------------------------------
# 8. Main Function
# ------------------------------

def main():
    """
    Main function to:
      - Load data
      - Either run hyperparameter tuning or do a standard train/test cycle
    """
    # Global so our objective function can see it
    global close_prices

    # Your CSV dataset path
    DATA_PATH = 'data/SP500.csv'  # Change this to your actual CSV file path

    # Load data
    close_prices = load_data(DATA_PATH)

    # Decide if we want to do hyperparameter tuning or just normal training/testing
    perform_tuning = True  # Set to False to skip tuning and do normal train/test

    if perform_tuning:
        run_hyperparameter_tuning()
    else:
        # Example of manual hyperparameter definitions
        WINDOW_SIZE = 10
        STATE_SIZE = WINDOW_SIZE + 2  # +2 for MA & RSI
        ACTION_SIZE = 3               # 0: Hold, 1: Buy, 2: Sell
        BATCH_SIZE = 64
        
        # Force a max of 100 training episodes
        EPISODE_COUNT = 100

        # Split data
        VALIDATION_SIZE = 0.2
        TRAIN_SIZE = int(len(close_prices) * (1 - VALIDATION_SIZE))
        X_train, X_test = close_prices[:TRAIN_SIZE], close_prices[TRAIN_SIZE:]
        
        # Train
        agent = Agent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            device=DEVICE,
            is_eval=False,
            learning_rate=0.0001,
            gamma=0.95,
            epsilon=1.0,
            epsilon_min=0.01,
            epsilon_decay=0.995
        )
        train_agent(agent, X_train, WINDOW_SIZE, BATCH_SIZE, EPISODE_COUNT)

        # Test using the last saved model (model_ep{EPISODE_COUNT}.pth)
        model_path = f"model_ep{EPISODE_COUNT}.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Trained model not found at {model_path}. Please ensure training is completed.")
        
        agent_eval = Agent(
            state_size=STATE_SIZE,
            action_size=ACTION_SIZE,
            device=DEVICE,
            model_path=model_path,
            is_eval=True
        )
        test_agent(agent_eval, X_test, WINDOW_SIZE)

# Standard Python entry point
if __name__ == "__main__":
    main()
