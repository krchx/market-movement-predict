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

# ...existing code...
def plot_results(history, save_path='results/performance.png', show=True):
    """Plot training history (loss, accuracy, F1)."""
    required = ['train_loss','val_loss','train_acc','val_acc','train_f1','val_f1']
    missing = [k for k in required if k not in history]
    if missing:
        raise KeyError(f"Missing history keys: {missing}")

    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    # Loss
    axs[0].plot(history['train_loss'], label='Train')
    axs[0].plot(history['val_loss'], label='Val')
    axs[0].set_title('Loss')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    axs[0].grid(True)

    # Accuracy
    axs[1].plot(history['train_acc'], label='Train')
    axs[1].plot(history['val_acc'], label='Val')
    axs[1].set_title('Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    axs[1].grid(True)

    # F1
    axs[2].plot(history['train_f1'], label='Train')
    axs[2].plot(history['val_f1'], label='Val')
    axs[2].set_title('F1 Score')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('F1')
    axs[2].legend()
    axs[2].grid(True)

    # Optional annotation
    axs[0].text(0.02, 0.95,
                f"Final Val Acc: {history['val_acc'][-1]:.3f}\nFinal Val F1: {history['val_f1'][-1]:.3f}",
                transform=axs[0].transAxes,
                va='top', ha='left',
                bbox=dict(facecolor='white', alpha=0.6, edgecolor='none', pad=4),
                fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
