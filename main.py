import torch
import os
import time
import pandas as pd

os.putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")

from config import cfg # Import configuration
from data import prepare_data
from models import HybridCNNGRUAttentionModel
from training import (
    train_model, 
    evaluate_model, 
    plot_results
)

def main():
    """Main function to run the market prediction model."""
    
    # Determine device
    # Device is now determined in config.py
    device = cfg.DEVICE
    print(f"Using device: {device}")
    
    # Create output directories if they don't exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Prepare data
    print("Preparing data...")
    train_loader, val_loader, test_loader, aligned_test_original_features, input_dim = prepare_data(
        file_path=cfg.DATA_PATH,
        seq_length=cfg.SEQ_LENGTH,
        horizon=cfg.HORIZON,
        batch_size=cfg.BATCH_SIZE
    )
    print(f"Input dimension: {input_dim}")
    
    # Create model
    print("Creating model...")
    model = HybridCNNGRUAttentionModel(
        input_dim=input_dim,
        cnn_channels=cfg.CNN_CHANNELS,
        rnn_hidden=cfg.RNN_HIDDEN,
        dropout=cfg.DROPOUT,
        use_lstm=cfg.USE_LSTM
    )
    
    # Train model
    print("Training model...")
    start_time = time.time()
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=cfg.EPOCHS,
        learning_rate=cfg.LR,
        weight_decay=cfg.WEIGHT_DECAY,
        patience=cfg.PATIENCE,
        device=device
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    # Save model
    torch.save(model.state_dict(), f'models/hybrid_model_{time.strftime("%Y%m%d_%H%M%S")}.pth')
    
    # Evaluate model
    print("Evaluating model...")
    eval_results = evaluate_model(model, test_loader, device=device)
    
    # Plot results
    print("Plotting results...")
    plot_results(history)
    
    # Save results
    results = {
        # Store configuration used for this run
        'model_config': {k: v for k, v in vars(cfg).items() if not k.startswith('_') and k != 'DEVICE'},
        'training_time': training_time,
        'accuracy': eval_results['accuracy'],
        'f1_score': eval_results['f1'],
        'confusion_matrix': eval_results['confusion_matrix'].tolist(),
    }
    
    # Convert results to DataFrame and save as CSV
    results_df = pd.DataFrame([results])
    results_df.to_csv(f'results/results_{time.strftime("%Y%m%d_%H%M%S")}.csv', index=False)
    
    print("Done!")

if __name__ == "__main__":
    main()
