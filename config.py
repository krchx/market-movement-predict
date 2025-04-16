import torch

class Config:
    # --- Data Parameters ---
    DATA_PATH = 'data/data.csv'          # Path to the input CSV data file
    SEQ_LENGTH = 60                 # Input sequence length (e.g., number of minutes)
    HORIZON = 5                     # Prediction horizon (e.g., number of minutes into the future)

    # --- Model Parameters ---
    CNN_CHANNELS = 32               # Number of output channels for the CNN layers
    RNN_HIDDEN = 64                 # Number of hidden units in the RNN (GRU or LSTM) layers
    DROPOUT = 0.3                   # Dropout rate for regularization
    USE_LSTM = False                # If True, use LSTM; otherwise, use GRU

    # --- Training Parameters ---
    BATCH_SIZE = 32                 # Number of samples per batch during training
    EPOCHS = 50                     # Maximum number of training epochs
    LR = 1e-4                       # Learning rate for the optimizer
    WEIGHT_DECAY = 1e-4             # Weight decay (L2 penalty) for the optimizer
    PATIENCE = 5                    # Number of epochs to wait for improvement before early stopping
    CPU = False                     # If True, force CPU usage even if CUDA is available

    # --- Trading Parameters ---
    THRESHOLD_BUY = 0.6            # Probability threshold to generate a buy signal
    THRESHOLD_SELL = 0.6           # Probability threshold to generate a sell signal (or hold/neutral)
    TRANSACTION_COST = 0.0005       # Simulated transaction cost per trade (e.g., 0.05%)

    # --- Runtime Determined ---
    # Automatically determine the device based on availability and the CPU flag
    DEVICE = 'cuda' if torch.cuda.is_available() and not CPU else 'cpu'
    # DEVICE = 'cpu'  # Force CPU usage for testing purposes

# Instantiate the configuration class
cfg = Config()

# You can easily override specific configurations after importing, e.g.:
# from config import cfg
# cfg.EPOCHS = 50
# cfg.DATA_PATH = 'new_data.csv'
