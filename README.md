# **AI Trading Agent**

A simple AI trading agent using Alpaca Markets and Reinforcement Learning (RL). The agent is designed to make trading decisions based on market data, leveraging advanced algorithms and innovative methodologies to improve profitability and robustness.

---

## **Overview**

This project combines reinforcement learning, real-time data, and AI techniques to automate trading decisions. It integrates with Alpaca Markets for execution, providing a foundation for advanced trading strategies. The current focus is on developing an agent that can achieve consistent profitability in simulated environments before transitioning to live trading.

---

## **Progress**

### **1. Reinforcement Learning Agent**
- **Environment**: Custom trading environment developed with `gym`.
- **Current Model**: PPO (Proximal Policy Optimization) with action space [Buy, Sell, Hold].
- **Reward Function**: Recently optimized for harsher penalties and profit amplification.
- **Training**: ~500k timesteps completed. Further steps include improving reward scaling and training longer with better metrics tracking.

### **2. Data Pipeline**
- **Historical Data**: Leveraging cleaned and feature-rich datasets with key indicators such as RSI, MACD, Bollinger Bands, etc.
- **Live Data**: Integration with Alpaca for real-time market feeds in progress.

### **3. Backtesting**
- **Results**: Current agent achieves minor gains but struggles with sustained profitability (reward mean capped around 0).
- **Planned Improvements**: Focused on refining the reward function, improving the model, and testing longer time horizons.

---

## **What's Next**

### **Immediate Goals**
1. **Achieving Profitability**:
   - Refine the RL model to break past the reward mean plateau.
   - Experiment with alternative RL algorithms such as:
     - Soft Actor-Critic (SAC)
     - TD3 (Twin Delayed Deep Deterministic Policy Gradient)
     - Rainbow DQN (if sticking to discrete actions).
   - Conduct systematic hyperparameter tuning using tools like Optuna or Ray Tune.

2. **Real-Time Data Stream**:
   - Set up a live data pipeline using Alpaca’s API for paper trading.
   - Ensure compatibility with the RL environment for seamless transition to live testing.

3. **Dashboard UI**:
   - Create a real-time dashboard to monitor trades, portfolio performance, and key metrics.

### **Mid-Term Goals**
1. **Continuous Training**:
   - Implement online learning to allow the model to adapt to new data dynamically.
   - Incorporate Bayesian methods to leverage priors from current batches to predict future trends.

2. **Incorporate RAG and Sentiment Analysis**:
   - Experiment with retrieval-augmented generation (RAG) for integrating external market data.
   - Use sentiment analysis from news and social media to inform trading decisions.

3. **Advanced Models**:
   - Explore time-series-compatible large language models (LLMs) or custom-built LLMs tailored for trading.
   - Investigate hybrid approaches combining time-series models with reinforcement learning agents.

---

## **Future Aspirations**
- Transition from paper trading to live trading on Alpaca Markets.
- Develop ensemble models for robustness across various market conditions.
- Open-source the project for community contributions and enhancements.

---