import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import numpy as np
import os
import torch # Required if using the model
# ...existing code...
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.preprocessing import load_and_clean_data, calculate_features, create_sequences # Add scale_features if needed
from models.architecture import HybridCNNGRUAttentionModel
from training.evaluate import generate_trading_signals
from config import cfg # Import the configuration object
# ...existing code...

# --- Configuration ---
# DATA_FILE = 'data/data.csv' # Use cfg.DATA_PATH
MODEL_DIR = 'models'
# TARGET_DATE = '2025-04-10' # Will be determined dynamically
USE_MODEL_SIGNALS = False # Set to True to use the trained model

# --- Model Hyperparameters (Required if USE_MODEL_SIGNALS=True) ---
# These are now sourced from cfg where applicable
# SEQ_LENGTH = 60 # Use cfg.SEQ_LENGTH
# INPUT_DIM = 10 # Determined dynamically from features
# CNN_CHANNELS = 32 # Use cfg.CNN_CHANNELS
# RNN_HIDDEN = 64 # Use cfg.RNN_HIDDEN
# DROPOUT = 0.3 # Use cfg.DROPOUT
# USE_LSTM = False # Use cfg.USE_LSTM
# THRESHOLD_BUY = 0.55 # Use cfg.THRESHOLD_BUY
# THRESHOLD_SELL = 0.45 # Use cfg.THRESHOLD_SELL
# SCALER_PATH = 'models/scaler.joblib' # Example path if scaler was saved

# --- Load and Prepare Data ---
print(f"Loading data from {cfg.DATA_PATH}...")
df = load_and_clean_data(cfg.DATA_PATH)

# Determine the last date in the dataset
if df.empty:
    print("Error: Data file is empty.")
    exit()
target_dt = df['datetime'].max().date()
TARGET_DATE = target_dt.strftime('%Y-%m-%d') # For printing and title
print(f"Filtering data for the last available date: {TARGET_DATE}")

# Filter for the target day
# target_dt = pd.to_datetime(TARGET_DATE).date() # Already have target_dt
df_day = df[df['datetime'].dt.date == target_dt].copy()

if df_day.empty:
    # This should ideally not happen if target_dt was derived from df
    print(f"Error: No data found for {TARGET_DATE}. This might indicate an issue.")
    exit()

# Convert datetime to matplotlib format
df_day['num_date'] = df_day['datetime'].apply(mdates.date2num)

# --- Generate Signals ---
if USE_MODEL_SIGNALS:
    print("Generating signals using trained model...")
    # ** 1. Find and Load Model **
    try:
        model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith('.pth')]
        if not model_files:
            raise FileNotFoundError("No model (.pth) files found in models directory.")
        latest_model_file = max(model_files, key=lambda f: os.path.getmtime(os.path.join(MODEL_DIR, f)))
        model_path = os.path.join(MODEL_DIR, latest_model_file)
        print(f"Loading model: {model_path}")

        # device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use cfg.DEVICE
        model = HybridCNNGRUAttentionModel(
            input_dim=INPUT_DIM, # Determined after feature calculation
            cnn_channels=cfg.CNN_CHANNELS,
            rnn_hidden=cfg.RNN_HIDDEN,
            dropout=cfg.DROPOUT,
            use_lstm=cfg.USE_LSTM
        )
        model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
        model.to(cfg.DEVICE)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to simple heuristic signals.")
        USE_MODEL_SIGNALS = False # Fallback

    # ** 2. Prepare Data for Model (Features, Scaling, Sequences) **
    # Note: This requires data *before* the target day for initial sequences
    if USE_MODEL_SIGNALS:
        try:
            print("Preparing data for model prediction...")
            # Select data up to the end of the target day + buffer for sequences
            start_index = df.index.get_loc(df_day.index[0])
            buffer_start_index = max(0, start_index - cfg.SEQ_LENGTH + 1)
            df_for_sequences = df.iloc[buffer_start_index : start_index + len(df_day)].copy()

            # Calculate features
            df_for_sequences = calculate_features(df_for_sequences)
            feature_cols = [col for col in df_for_sequences.columns if col not in ['datetime', 'timestamp', 'open', 'high', 'low', 'close', 'volume', 'hour', 'minute', 'day', 'target', 'num_date']]
            INPUT_DIM = len(feature_cols) # Set INPUT_DIM based on actual features
            print(f"Determined INPUT_DIM from features: {INPUT_DIM}")

            # --- Scaling ---
            # Option 1: Load saved scaler (Recommended)
            # scaler = joblib.load(SCALER_PATH)
            # df_for_sequences[feature_cols] = scaler.transform(df_for_sequences[feature_cols])

            # Option 2: Fit scaler on this subset (Less Ideal - introduces lookahead bias if not careful)
            # scaler = StandardScaler()
            # df_for_sequences[feature_cols] = scaler.fit_transform(df_for_sequences[feature_cols])

            # Option 3: Skip scaling (If model was trained on unscaled data or for quick viz)
            print("Skipping feature scaling for this plot (add scaler loading if needed).")


            # Create sequences
            X, _, sequence_indices = create_sequences(df_for_sequences[feature_cols].values, None, seq_length=cfg.SEQ_LENGTH, return_indices=True)
            X_tensor = torch.tensor(X, dtype=torch.float32).to(cfg.DEVICE)

            # ** 3. Get Predictions **
            print("Running model inference...")
            with torch.no_grad():
                y_pred, _ = model(X_tensor)
                probabilities = torch.sigmoid(y_pred.squeeze(1)).cpu().numpy()

            # ** 4. Generate Signals **
            model_signals = generate_trading_signals(probabilities, cfg.THRESHOLD_BUY, cfg.THRESHOLD_SELL)

            # ** 5. Align Signals to Daily Data **
            # The signal from sequence `i` (ending at index `idx`) applies to the candle *after* `idx`.
            # Map signals back to the original df_day index.
            df_day['signal'] = 0 # Default to hold
            for i, signal in enumerate(model_signals):
                # Get the index in the *original* df corresponding to the end of the sequence
                original_df_index = df_for_sequences.index[sequence_indices[i]]
                # If that index is within our target day, assign the signal to the *next* candle
                if original_df_index in df_day.index[:-1]: # Exclude last index as signal applies after it
                    signal_target_index = df_day.index[df_day.index.get_loc(original_df_index) + 1]
                    df_day.loc[signal_target_index, 'signal'] = signal

            print(f"Generated {len(model_signals)} signals from model.")

        except Exception as e:
            print(f"Error during model signal generation: {e}")
            print("Falling back to simple heuristic signals.")
            USE_MODEL_SIGNALS = False # Fallback

# --- Simple Heuristic Signals (Fallback or if USE_MODEL_SIGNALS=False) ---
if not USE_MODEL_SIGNALS:
    print("Generating signals using simple heuristic (Close vs Open)...")
    df_day['signal'] = df_day.apply(lambda row: 1 if row['close'] > row['open'] else (-1 if row['close'] < row['open'] else 0), axis=1)

# --- Plotting ---
print("Plotting candlestick chart...")
fig, ax = plt.subplots(figsize=(15, 7))
candle_width = 0.0008 # Adjust as needed based on time axis density

# Define colors
colors = {1: 'green', -1: 'red', 0: 'yellow'}

# Plot each candle
for _, row in df_day.iterrows():
    color = colors.get(row['signal'], 'gray') # Use gray for any unexpected signal values
    # Draw wick (high-low line)
    ax.plot([row['num_date'], row['num_date']], [row['low'], row['high']], color='black', linewidth=0.8)
    # Draw body (open-close rectangle)
    body_bottom = min(row['open'], row['close'])
    body_height = abs(row['close'] - row['open'])
    # Avoid plotting zero-height rectangles if open == close
    if body_height > 0:
        rect = Rectangle((row['num_date'] - candle_width / 2, body_bottom), candle_width, body_height, color=color, zorder=3)
        ax.add_patch(rect)
    else:
        # Draw a thin line if open == close
        ax.plot([row['num_date'] - candle_width / 2, row['num_date'] + candle_width / 2], [row['open'], row['close']], color=color, linewidth=1.5, zorder=3)


# Format x-axis to show time
ax.xaxis_date()
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
plt.xticks(rotation=45)

# Add labels and title
signal_type = "Model" if USE_MODEL_SIGNALS and 'signal' in df_day.columns and df_day['signal'].nunique() > 1 else "Heuristic (O/C)"
plt.title(f'Candlestick Chart for {TARGET_DATE} (Signals: {signal_type})')
plt.xlabel('Time')
plt.ylabel('Price')
plt.grid(True, linestyle='--', alpha=0.5)

# Add legend manually
legend_elements = [
    Rectangle((0, 0), 1, 1, color='green', label='Buy Signal'),
    Rectangle((0, 0), 1, 1, color='red', label='Sell Signal'),
    Rectangle((0, 0), 1, 1, color='yellow', label='Hold Signal')
]
ax.legend(handles=legend_elements, loc='upper left')

plt.tight_layout()
plt.show()

print("Plot displayed.")