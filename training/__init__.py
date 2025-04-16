"""Training and evaluation functions."""

from .train import train_model
from .evaluate import evaluate_model, generate_trading_signals, backtest_strategy, plot_results

__all__ = [
    "train_model",
    "evaluate_model",
    "generate_trading_signals",
    "backtest_strategy",
    "plot_results",
]
