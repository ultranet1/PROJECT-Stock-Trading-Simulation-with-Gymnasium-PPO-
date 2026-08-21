## Stock Trading Simulation with Gymnasium (PPO)

This repository implements a reinforcement learning trading simulation using Gymnasium and Stable-Baselines3 (PPO) on Apple (AAPL) daily prices. It is inspired by and based on the DataCamp project “stock trading simulation with gymnasium” (`https://app.datacamp.com/learn/projects/2468`), but reproduced and extended in my own local environment with reusable code and results for portfolio strategy exploration.

### What’s included
- A runnable Jupyter notebook: `StockTradingRL.ipynb`
- Data sample reference: `AAPL copy.csv` (Date, Close)
- Saved result charts in `charts/` after you run the notebook:
  - `charts/price_actions.png`: Price with buy/sell markers
  - `charts/portfolio_value.png`: Portfolio value over time

### Why this is useful for portfolio strategy
- Demonstrates end-to-end RL workflow: data ingestion, environment setup (`gym-anytrading`), PPO training, evaluation, and visualization.
- Provides a clear baseline to iterate on features (technical indicators, multi-asset training, risk constraints) and training regimes.
- The notebook cleanly separates training from an execution loop with percentage-based position sizing, making it easy to port to other datasets.

### Environment and dependencies
Install the core dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` includes Gymnasium, gym-anytrading, Stable-Baselines3, NumPy, Pandas, and Matplotlib.

### Data
- Expected CSV format with two columns: `Date`, `Close`.
- The notebook is configured to point to the included file path:
  - `/Users/sachabrouck/StockTradingSimDatacamp/AAPL copy.csv`
- If you clone this repo elsewhere, update the `csv_path` in the notebook to your local path.

### How to run
Option A) Open the notebook:

```bash
jupyter notebook StockTradingRL.ipynb
```

2) Run all cells. The notebook will:
- Load the AAPL data
- Create the `stocks-v0` trading environment
- Train a PPO agent for `10_000` timesteps
- Evaluate with a 10% position-sizing policy per action
- Save charts to `charts/`

Option B) CLI one-liner (no notebook):

```bash
python generate_charts.py
```

This will train PPO and save the two charts to `charts/`.

### Results
After running, you’ll find and can view these charts below:

![AAPL Price with Buy/Sell Signals (PPO)](charts/price_actions.png)

![Portfolio Value Over Time](charts/portfolio_value.png)

### Notes on performance and limitations
- With daily Close-only data and a short training budget (10k steps), performance is typically modest and regime-dependent. In strongly trending periods, buy-and-hold can dominate; in choppy markets, frequent trades may hurt performance.
- `stocks-v0` is intentionally simplified. Consider adding features (volume, volatility, indicators), richer reward shaping, transaction costs and slippage modeling, and cross-asset training windows for more realistic results.

### Attribution
- Based on DataCamp’s “stock trading simulation with gymnasium” project: `https://app.datacamp.com/learn/projects/2468`
- Re-implemented and extended locally to create a reusable baseline for data science portfolio strategy experiments.


