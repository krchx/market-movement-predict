import pandas as pd
import matplotlib.pyplot as plt
import os

def start_trading(df, plot=True, save_path="results/trades_plot.png"):
    """Very simple virtual trading simulation.

    Rules:
      predicted_index 0 (Down)  -> take a short position (-1)
      predicted_index 2 (Up)    -> take a long position (+1)
      predicted_index 1 (Side)  -> flat (0)

    On each signal change we close the previous position at current close price
    and open the new one. PnL is (exit_price - entry_price) * position_size
    where position_size is +1 for long, -1 for short (so short gains when price falls).
    """
    if df.empty:
        print("No data passed to start_trading.")
        return

    required_cols = {"predicted_index", "close"}
    missing = required_cols - set(df.columns)
    if missing:
        print(f"Missing required columns for trading: {missing}")
        return

    position = 0          # current position size (-1, 0, 1)
    balance = 0.0          # realized PnL
    trades = []            # record of executed trades
    charges = 0.0013

    for idx, row in df.iterrows():
        signal = int(row["predicted_index"])
        price = float(row["close"]) if not pd.isna(row["close"]) else None
        if price is None:
            continue

        if signal == 0:
            balance += price
            balance -= price * charges  # Deduct trading charges, one time for each trade, during sell
            position -= 1
            trades.append({
                "index": idx,
                "action": "SELL",
                "price": price,
                "balance": balance
            })
        elif signal == 2:
            balance -= price
            position += 1
            trades.append({
                "index": idx,
                "action": "BUY",
                "price": price,
                "balance": balance
            })

    # Close final open position at last price
    if position != 0:
        last_price = float(df.iloc[-1]["close"])
        pnl = last_price * position
        pnl -= last_price * charges if position > 0 else 0
        position = 0
        balance += pnl
        trades.append({
            "index": df.index[-1],
            "action": "FINAL_CLOSE",
            "price": last_price,
            "balance": balance
        })

    print(f"Final realized balance: {balance:.2f}; final position: {position}")

    if plot and trades:
        trades_df = pd.DataFrame(trades)
        # Determine x-axis (prefer datetime if present)
        if "datetime" in df.columns:
            x_full = df["datetime"]
        else:
            x_full = df.index

        price_series = df["close"].astype(float)

        # Map trade indices to x values
        trades_df["x"] = trades_df["index"].map(lambda i: df.loc[i, "datetime"] if "datetime" in df.columns else i)

        plt.figure(figsize=(12, 6))
        plt.plot(x_full, price_series, label="Close Price", color="black", linewidth=1)

        # Plot markers
        def scatter_subset(action, marker, color, label):
            subset = trades_df[trades_df.action == action]
            if not subset.empty:
                plt.scatter(subset["x"], subset["price"], marker=marker, color=color, s=80, label=label, edgecolor="k")

        scatter_subset("BUY", marker="^", color="green", label="Open Long")
        scatter_subset("SELL", marker="v", color="red", label="Open Short")
        scatter_subset("FINAL_CLOSE", marker="X", color="black", label="final settle")

        plt.title("Trade Execution Plot")
        plt.xlabel("Time")
        plt.ylabel("Price")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()

        # Ensure directory exists
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=120)
            print(f"Trade plot saved to {save_path}")
        else:
            plt.show()

    # Return raw trade list for further analysis
    return trades
