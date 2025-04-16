import torch
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
from tqdm import tqdm

def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate the model on the test set."""
    model = model.to(device)
    model.eval()
    
    predictions = []
    probabilities = []
    targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in tqdm(test_loader, desc="Evaluating"):
            X_batch = X_batch.to(device)
            
            # Forward pass
            y_pred, _ = model(X_batch)  # Output shape: (batch_size, 3)
            
            # Store results
            probs = F.softmax(y_pred, dim=1).cpu().numpy()  # Convert logits to probabilities
            preds = torch.argmax(y_pred, dim=1).cpu().numpy()  # Get predicted class indices
            
            predictions.extend(preds)
            probabilities.extend(probs)
            targets.extend(y_batch.cpu().numpy())
    
    # Calculate metrics
    acc = accuracy_score(targets, predictions)
    f1 = f1_score(targets, predictions, average='weighted')
    cm = confusion_matrix(targets, predictions)
    
    print(f"Test Accuracy: {acc:.4f}")
    print(f"Test F1 Score: {f1:.4f}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'accuracy': acc,
        'f1': f1,
        'confusion_matrix': cm,
        'predictions': np.array(predictions),
        'probabilities': np.array(probabilities),
        'targets': np.array(targets)
    }

def generate_trading_signals(probabilities, threshold_buy=0.6, threshold_sell=0.6):
    """Generate trading signals based on model output probabilities for 3 classes."""
    signals = np.zeros(len(probabilities))
    
    for i, probs in enumerate(probabilities):
        # probs is array of shape (3,) with [P(down), P(sideways), P(up)]
        if probs[2] >= threshold_buy:  # High probability for "up" class (index 2)
            signals[i] = 1  # Buy signal
        elif probs[0] >= threshold_sell:  # High probability for "down" class (index 0)
            signals[i] = -1  # Sell signal
        # Otherwise signal remains 0 (hold)
    
    return signals

def backtest_strategy(close_prices, signals, transaction_cost=0.0005):
    """Simple backtesting of a trading strategy."""
    # Initialize portfolio and positions
    portfolio_value = np.ones(len(signals) + 1)  # Start with $1
    position = 0  # 0 = no position, 1 = long, -1 = short
    trades = 0
    
    # Metrics to track
    daily_returns = []
    
    # Execute trades based on signals
    for i in range(len(signals)):
        # Current portfolio value (before potential trade)
        current_value = portfolio_value[i]
        
        # Check for a trade signal
        if signals[i] == 1 and position <= 0:  # Buy signal
            # Close short position if exists
            if position == -1:
                # Close short position
                returns = 2 * transaction_cost - (close_prices[i] / close_prices[i-1] - 1)
                portfolio_value[i+1] = current_value * (1 + returns)
                trades += 1
                
            # Open long position
            position = 1
            portfolio_value[i+1] = current_value * (1 - transaction_cost)
            trades += 1
            
        elif signals[i] == -1 and position >= 0:  # Sell signal
            # Close long position if exists
            if position == 1:
                # Close long position
                returns = (close_prices[i] / close_prices[i-1] - 1) - 2 * transaction_cost
                portfolio_value[i+1] = current_value * (1 + returns)
                trades += 1
                
            # Open short position
            position = -1
            portfolio_value[i+1] = current_value * (1 - transaction_cost)
            trades += 1
            
        else:
            # No change in position, just update portfolio value based on current position
            if position == 1:  # Long position
                portfolio_value[i+1] = current_value * (1 + (close_prices[i] / close_prices[i-1] - 1))
            elif position == -1:  # Short position
                portfolio_value[i+1] = current_value * (1 - (close_prices[i] / close_prices[i-1] - 1))
            else:  # No position
                portfolio_value[i+1] = current_value
        
        # Calculate daily return
        daily_returns.append(portfolio_value[i+1] / portfolio_value[i] - 1)
    
    # Calculate performance metrics
    total_return = portfolio_value[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(signals)) - 1
    sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns) if np.std(daily_returns) > 0 else 0
    max_drawdown = np.max(np.maximum.accumulate(portfolio_value) - portfolio_value) / np.max(portfolio_value)
    
    print(f"Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
    print(f"Annualized Return: {annual_return:.4f} ({annual_return*100:.2f}%)")
    print(f"Sharpe Ratio: {sharpe_ratio:.4f}")
    print(f"Max Drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
    print(f"Total Trades: {trades}")
    
    return {
        'portfolio_value': portfolio_value,
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'trades': trades,
        'daily_returns': daily_returns
    }

def plot_results(history, backtest_results):
    """Plot training history and backtest results."""
    # Create figure with 2x2 subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training history - Loss
    axs[0, 0].plot(history['train_loss'], label='Train Loss')
    axs[0, 0].plot(history['val_loss'], label='Validation Loss')
    axs[0, 0].set_title('Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot training history - Accuracy
    axs[0, 1].plot(history['train_acc'], label='Train Accuracy')
    axs[0, 1].plot(history['val_acc'], label='Validation Accuracy')
    axs[0, 1].set_title('Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot training history - F1 Score
    axs[1, 0].plot(history['train_f1'], label='Train F1 Score')
    axs[1, 0].plot(history['val_f1'], label='Validation F1 Score')
    axs[1, 0].set_title('F1 Score')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('F1 Score')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot backtest results - Portfolio Value
    axs[1, 1].plot(backtest_results['portfolio_value'])
    axs[1, 1].set_title('Portfolio Value')
    axs[1, 1].set_xlabel('Time')
    axs[1, 1].set_ylabel('Value ($)')
    axs[1, 1].grid(True)
    
    # Add backtest metrics as text
    metrics_text = (
        f"Total Return: {backtest_results['total_return']:.4f}\n"
        f"Annual Return: {backtest_results['annual_return']:.4f}\n"
        f"Sharpe Ratio: {backtest_results['sharpe_ratio']:.4f}\n"
        f"Max Drawdown: {backtest_results['max_drawdown']:.4f}\n"
        f"Trades: {backtest_results['trades']}"
    )
    axs[1, 1].text(
        0.05, 0.05, metrics_text, transform=axs[1, 1].transAxes,
        bbox=dict(facecolor='white', alpha=0.7)
    )
    
    plt.tight_layout()
    plt.savefig('results/performance.png')
    plt.show()
