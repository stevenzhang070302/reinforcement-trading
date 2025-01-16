# **AI Trading Agent**

A simple AI trading agent using Alpaca Markets and Reinforcement Learning (RL). The agent is designed to make trading decisions based on market data, leveraging advanced algorithms and innovative methodologies to improve profitability and robustness. Recently, I saw a XiaoHongShu post on how someone was able to use Reinforcement Learning to make profitable trades, growing their portfolio from 90k to 110k. This reignited my interest in Algorithmic Trading and AI Trading, leading me to start this project.

---

## **Overview**

This project combines reinforcement learning, real-time data, and AI techniques to automate trading decisions. It integrates with Alpaca Markets for execution, providing a foundation for advanced trading strategies. The current focus is on developing an agent that can achieve consistent profitability in simulated environments before transitioning to live trading.

---

## **Progress**

### **Current Status**
#### **1. Reinforcement Learning Agent**
- **Environment**: Custom trading environment developed with `gym`.
- **Current Model**: PPO (Proximal Policy Optimization) with action space [Buy, Sell, Hold].
- **Reward Function**: Recently optimized for harsher penalties and profit amplification.
- **Training**: ~500k timesteps completed. Further steps include improving reward scaling and training longer with better metrics tracking.

#### **2. Data Pipeline**
- **Historical Data**: Leveraging cleaned and feature-rich datasets with key indicators such as RSI, MACD, Bollinger Bands, etc.
- **Live Data**: Integration with Alpaca for real-time market feeds in progress.

#### **3. Backtesting**
- **Results**:
  - **Initial Balance**: $10,000  
  - **Final Portfolio Value**: $10,532.57  
  - **Percentage Change**: +5.33%  
  - **Observation**: While profitability was achieved, there is room for improvement in sustained performance. The agent is currently stuck at a reward mean of ~0.001, indicating a need for further model refinement. In addition, this is only the backtest - more aggressive improvements are needed to ensure profitability in live trading.
  - Backtest results saved to `backtest_results.csv`.

- **Planned Improvements**: Focused on refining the reward function, improving the model, and testing longer time horizons.

---

## **What's Next**

### **Next Steps**
1. **Achieving Profitability**:
   - Refine the RL model to break past the reward mean plateau.
   - Experiment with alternative RL algorithms such as:
     - Soft Actor-Critic (SAC)
     - TD3 (Twin Delayed Deep Deterministic Policy Gradient)
     - Rainbow DQN (if sticking to discrete actions).
   - Conduct systematic hyperparameter tuning using tools like Optuna or Ray Tune.
   - Experiment with different machine learning models for feature extraction and prediction.

2. **Real-Time Data Stream**:
   - Set up a live data pipeline using Alpaca’s API for paper trading.
   - Ensure compatibility with the RL environment for seamless transition to live testing.

3. **Dashboard UI**:
   - Create a real-time dashboard to monitor trades, portfolio performance, and key metrics.

4. **Continuous Training**:
   - Implement online learning to allow the model to adapt to new data dynamically.
   - Incorporate Bayesian methods to leverage priors from current batches to predict future trends.

5. **Incorporate RAG and Sentiment Analysis**:
   - Experiment with retrieval-augmented generation (RAG) for integrating external market data.
   - Use sentiment analysis from news and social media to inform trading decisions.

6. **Advanced Models**:
   - Explore time-series-compatible large language models (LLMs) or custom-built LLMs tailored for trading.
   - Investigate hybrid approaches combining time-series models with reinforcement learning agents.

---