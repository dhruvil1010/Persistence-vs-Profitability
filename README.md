# Persistence vs. Profitability: A Cointegration Study of NIFTY Stock Pairs

![Project Status](https://img.shields.io/badge/Status-Research_Complete-success)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

## 1. Executive Summary
This project is an algorithmic trading research framework designed to analyze **Statistical Arbitrage (Pairs Trading)** opportunities within the NIFTY 50 universe. 

Unlike standard backtesting engines that simply check for profitability, this project was built to empirically test a specific quantitative hypothesis: **"Does the historical duration of a cointegration relationship predict its future stability and profitability?"**

The system implements an end-to-end pipeline: from data ingestion and Engle-Granger cointegration testing to a rolling Z-score trading strategy with out-of-sample (OOS) validation.

---

## 2. Research Hypotheses
The core objective was to disentangle two competing drivers of alpha in mean-reversion strategies:

### Hypothesis A: The Duration Theory (Persistence)
* **Premise:** A pair that has been cointegrated for 10+ years (e.g., 2010–2022) represents a fundamental economic link and will be more robust to regime changes in 2023 than a pair with only 3 years of history.
* **Metric:** Length of the Formation Window (Years).

### Hypothesis B: The Mean Reversion Speed Theory (Half-Life)
* **Premise:** Profitability is driven not by how *long* the relationship has existed, but by how *fast* the spread reverts to the mean (Half-Life). Faster reversion creates more trading opportunities and reduces exposure time.
* **Metric:** Trading Frequency (Empirical proxy for Ornstein-Uhlenbeck $\lambda$).

---

## 3. Mathematical Framework (Corrected & Precise)

This project implements a statistical arbitrage framework based on the **Engle–Granger two-step cointegration methodology**, combined with a rolling mean-reversion trading rule. 

The objective is not just to maximize profit, but to empirically study how cointegration window stability (Endogenous Formation Length) and mean-reversion speed (Half-Life proxy) relate to out-of-sample performance.

### 3.1 Step 1: Linear Relationship Estimation (OLS)
Individual stock prices are typically non-stationary processes (integrated of order one, $I(1)$). However, if two assets share a stable economic relationship, a linear combination of their prices may be stationary ($I(0)$).

For each stock pair, we estimate the linear relationship using **Ordinary Least Squares (OLS)** over candidate formation windows ending in 2022:

$$Y_t = \alpha + \beta X_t + \epsilon_t$$

Where:
* $Y_t$: Adjusted close price of Stock A
* $X_t$: Adjusted close price of Stock B
* $\beta$: **Hedge Ratio**, representing the relative exposure between the two assets
* $\epsilon_t$: Regression residuals

### 3.2 Step 2: Cointegration Testing (Engle–Granger Procedure)
To verify whether the estimated relationship is statistically meaningful, we apply the **Engle–Granger cointegration test** (`statsmodels.tsa.stattools.coint`), which internally performs an Augmented Dickey–Fuller (ADF) test on the regression residuals.

* **Null Hypothesis ($H_0$):** The residuals are non-stationary (no cointegration).
* **Alternative Hypothesis ($H_1$):** The residuals are stationary (cointegrated).

A pair is considered cointegrated if the **p-value < 0.05**.

**Sliding Formation Window Logic:**
Instead of assuming a fixed historical period (e.g., arbitrarily starting in 2010), the model:
1.  Iterates over multiple candidate start years.
2.  Keeps the end date fixed at **Dec 31, 2022** (to prevent look-ahead bias).
3.  Selects the **earliest formation window** that satisfies the cointegration threshold ($p < 0.05$).
*This allows the cointegration window length to emerge endogenously from the data.*

### 3.3 Step 3: Spread Construction
Once cointegration is confirmed, we construct the **Spread** ($S_t$) during the trading period (2023–present):

$$S_t = Y_t - \beta X_t$$

**Interpretation:**
* **Long:** 1 unit of Stock A
* **Short:** $\beta$ units of Stock B
* $S_t$ represents the value of this synthetic portfolio.

**Why the intercept ($\alpha$) is excluded:**
The intercept is a constant offset. Since all trading signals are generated using a **rolling mean** (in Step 4), any constant bias is naturally removed by the subtraction of $\mu_t$. Therefore, explicitly including $\alpha$ is unnecessary for the signal generation.

### 3.4 Step 4: Normalization via Rolling Z-Score
Although the spread is stationary, its scale varies with price levels. To standardize deviations, we compute a **Rolling Z-Score**:

$$Z_t = \frac{S_t - \mu_t}{\sigma_t}$$

Where:
* $\mu_t$: Rolling **Mean** of the spread (30-day window).
* $\sigma_t$: Rolling **Standard Deviation** of the spread (30-day window).

*This rolling normalization allows the strategy to adapt to changing volatility regimes in the Out-of-Sample period.*

### 3.5 Step 5: Mean-Reversion Trading Logic
The strategy assumes that extreme deviations of the Z-score will revert toward zero.

* **Short Spread Entry ($Z_t > 2.0$):** Spread is expensive.
    * *Action:* Sell Stock A, Buy $\beta$ Stock B.
* **Long Spread Entry ($Z_t < -2.0$):** Spread is cheap.
    * *Action:* Buy Stock A, Sell $\beta$ Stock B.
* **Exit Condition ($|Z_t| < 0.5$):** Mean reversion achieved.
    * *Action:* Close all positions.

---

## 4. Technical Implementation & Robustness

### 4.1 The "Cold Start" Engineering Fix
A critical flaw in standard rolling-window backtests is the "Cold Start" bias. If the backtest begins on Jan 1st, 2023, a 30-day rolling window returns `NaN` for the first month, effectively blinding the strategy.

* **Solution:** Implemented a **Warmup Buffer**.
* **Mechanism:** The system fetches data from Q4 2022 but suppresses trading signals until Jan 1st, 2023.
* **Result:** The Z-score is fully converged on Day 1 of the trading period, allowing for immediate trade execution (e.g., capturing January volatility).

### 4.2 Out-of-Sample (OOS) Validation
To prevent overfitting:
1.  **Formation (Train):** 2010 – Dec 31, 2022. All parameter selection (Hedge Ratio, Cointegration check) happens here.
2.  **Trading (Test):** Jan 1, 2023 – Present. No parameters are updated; the strategy runs "blind" on unseen data.

---

## 5. Key Findings & Visualizations

### Result 1: "History Length" is Irrelevant (Hypothesis A Disproven)
Contrary to intuition, pairs with 12+ years of cointegration history did **not** outperform pairs with 3 years of history. 

The chart below demonstrates a statistically flat trendline, proving that **Stationarity $\neq$ Profitability**. A pair can be statistically stable for a decade but fail to generate alpha if the spread volatility is too low or the mean reversion is too slow.

![Duration Analysis](result/profit_vs_cointegrationWindow.png)
*Figure 1: Window Duration (X-Axis) vs. Profit (Y-Axis). The flat trendline indicates zero correlation between history length and future returns.*

### Result 2: "Speed" Drives Alpha (Hypothesis B Confirmed)
The chart below reveals a strong positive correlation between **Trading Frequency** and **Profit**.
* Pairs with a **Short Half-Life** (high trade count) snapped back quickly, generating consistent profits.
* This confirms that for this strategy, **Mean Reversion Speed (Ornstein-Uhlenbeck $\lambda$)** is the dominant factor for success, overshadowing the duration of the economic relationship.

![Frequency Analysis](result/profit_vs_frequency.png)
*Figure 2: Trading Frequency (X-Axis) vs. Profit (Y-Axis). The upward trendline confirms that higher activity (faster mean reversion) leads to higher alpha.*

---

## 6. Limitations & Risk Factors
* **Multiple Hypothesis Testing:** We use a fixed $p=0.05$ threshold across all NIFTY pairs. We acknowledge the risk of False Discovery Rate (FDR) inflation, though strict OOS testing mitigates this.
* **Regime Shift:** The strategy assumes the $\beta$ (Hedge Ratio) remains constant from 2022 to 2023. In a production environment, a Kalman Filter or Rolling OLS would be preferred for dynamic $\beta$ adjustment.

## 7. Tech Stack
* **Statsmodels:** Cointegration tests (ADF) and OLS Regression.
* **Pandas/NumPy:** Vectorized backtesting engine.
* **Streamlit:** Interactive UI for regime analysis.
* **Plotly:** High-dimensional scatter plots for hypothesis testing.
* **YFinance:** Real-time adjusted OHLCV data.

## Author: Dhruvil Katharotiya | 2025
