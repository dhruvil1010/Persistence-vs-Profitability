# Cointegration Window Selection Logic

def find_cointegration_window(stock1_data, stock2_data, end_year, min_start_year):
    earliest_year = stock1_data.index.min().year

    for start_year in range(earliest_year, min_start_year + 1):
        start_date = f"{start_year}-01-01"
        end_date = f"{end_year}-12-31"

        window_data = concat(stock1_data, stock2_data)
        window_data = window_data.loc[start_date:end_date].dropna()

        if len(window_data) < 252:
            continue

        pvalue = cointegration_test(
            window_data[:, 0],
            window_data[:, 1]
        )

        if pvalue < 0.05:
            return start_date, end_date, pvalue

    return None, None, None

# Mean Reversion Backtest Logic (FULL ACCOUNTING + WARMUP)

def run_backtest(
    trading_data,
    hedge_ratio,
    warmup_data,
    initial_capital,
    entry_threshold,
    exit_threshold,
    rolling_window,
    allocation,
    fee_bps
):
    if trading_data is empty:
        return initial_capital, 0, 0, 0, 0, [], []

    if warmup_data exists:
        full_data = concatenate(warmup_data, trading_data)
        cutoff = length(warmup_data)
    else:
        full_data = trading_data
        cutoff = 0

    price1 = full_data[:, 0]
    price2 = full_data[:, 1]

    spread = price1 - hedge_ratio * price2
    mean = rolling_mean(spread, rolling_window)
    std = rolling_std(spread, rolling_window)
    zscore = (spread - mean) / std

    zscore = zscore[cutoff:]
    price1 = price1[cutoff:]
    price2 = price2[cutoff:]

    cash = initial_capital
    q1 = 0
    q2 = 0
    position = 0
    trade_count = 0
    equity_curve = []
    trade_pnls = []
    entry_equity = None

    for t in range(length(zscore)):
        equity = cash + q1 * price1[t] + q2 * price2[t]
        equity_curve.append(equity)

        if zscore[t] is not finite:
            continue

        if position == 0:
            if abs(zscore[t]) > entry_threshold:
                deploy = allocation * equity
                gross = abs(price1[t]) + abs(hedge_ratio * price2[t])
                k = deploy / gross

                if zscore[t] < 0:
                    q1 =  k
                    q2 = -hedge_ratio * k
                else:
                    q1 = -k
                    q2 =  hedge_ratio * k

                cash -= (q1 * price1[t] + q2 * price2[t])
                cash -= fee_bps * (abs(q1 * price1[t]) + abs(q2 * price2[t]))

                entry_equity = equity
                trade_count += 1
                position = 1

        else:
            if abs(zscore[t]) < exit_threshold:
                cash += (q1 * price1[t] + q2 * price2[t])
                cash -= fee_bps * (abs(q1 * price1[t]) + abs(q2 * price2[t]))

                exit_equity = cash
                trade_pnls.append(exit_equity - entry_equity)

                q1 = 0
                q2 = 0
                position = 0
                entry_equity = None

    final_equity = equity_curve[-1]
    pnl = final_equity - initial_capital
    pnl_pct = pnl / initial_capital

    win_rate = count(p > 0 for p in trade_pnls) / length(trade_pnls)

    return final_equity, pnl, pnl_pct, trade_count, win_rate, equity_curve, zscore

# Full Analysis Loop Logic (Pairwise Research)

for each pair (stock1, stock2):
    formation_data = prices before 2023

    start, end, pvalue = find_cointegration_window(
        stock1, stock2
    )

    if no cointegration:
        continue

    hedge_ratio = OLS(stock1 ~ stock2 on formation window)

    warmup_data = prices from Oct–Dec 2022
    trading_data = prices from 2023 onward

    results = run_backtest(
        trading_data,
        hedge_ratio,
        warmup_data
    )

    window_length = end_year - start_year + 1

    store:
        - window_length
        - pvalue
        - hedge_ratio
        - trade_count
        - win_rate
        - pnl
        - pnl_pct
