import torch
import pandas as pd
import joblib
import numpy as np
from config import cfg
from models.architecture import HybridCNNGRUAttentionModel
from data.preprocessing import load_and_clean_data, calculate_features, split_features, create_target
from test.start_trading import start_trading

# --- Configuration ---
DEVICE = cfg.DEVICE
SEQ_LENGTH = cfg.SEQ_LENGTH
MODEL_PATH = 'models/hybrid_model_20250824_135449.pth' # Path to the trained model
SCALER_PATH = 'data/scaler.joblib' # Path to the scaler used during training
DATA_PATH = 'usemodel/data.csv' # Path to the new day's data

# --- Load Model ---
print("Loading model...")
INPUT_DIM = 12
model = HybridCNNGRUAttentionModel(
    input_dim=INPUT_DIM,
    cnn_channels=cfg.CNN_CHANNELS,
    rnn_hidden=cfg.RNN_HIDDEN,
    dropout=cfg.DROPOUT, # Dropout is part of the architecture, needed for instantiation
    use_lstm=cfg.USE_LSTM
)
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
except FileNotFoundError:
    print(f"Error: Model file not found at {MODEL_PATH}")
    exit()
except Exception as e:
    print(f"Error loading model state_dict: {e}")
    exit()

model.to(DEVICE)
model.eval()
print("Model loaded.")

# --- Load Scaler ---
print("Loading scaler...")
try:
    scaler = joblib.load(SCALER_PATH)
    print("Scaler loaded.")
except FileNotFoundError:
    print(f"Error: Scaler file not found at {SCALER_PATH}")
    print("This file should have been saved during the training's scale_features step.")
    print("Cannot proceed without the scaler fitted on the original training data.")
    exit()
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# --- Define feature columns ---
FEATURE_COLS = ['open_return', 'high_return', 'low_return', 'close_return',
                'volume_change', 'volume_velocity', 'relative_position', 'body_wick_ratio',
                'minute_sin', 'minute_cos', 'day_sin', 'day_cos']
if INPUT_DIM != len(FEATURE_COLS):
    print(f"Warning: INPUT_DIM ({INPUT_DIM}) does not match the number of FEATURE_COLS ({len(FEATURE_COLS)}). Check definitions.")

print(f"Loading and preprocessing new data from {DATA_PATH}...")
try:
    # Load and perform initial cleaning
    new_df_raw = load_and_clean_data(DATA_PATH)

    # Calculate features
    df_with_features = calculate_features(new_df_raw.copy()) # Use copy to avoid warnings

    # Create Target
    df_with_target = create_target(df_with_features)

    # Drop unused columns and keep only the relevant features
    new_df_features, df_original = split_features(df_with_target)

    if new_df_features.empty:
        print("Error: No data left after loading and initial feature calculation (DataFrame is empty).")
        exit()

    print(f"Loaded and calculated features for {len(new_df_features)} time steps.")

except FileNotFoundError:
    print(f"Error: New data file not found at {DATA_PATH}")
    exit()
except Exception as e:
    print(f"Error during data loading or feature calculation: {e}")
    exit()

# --- Scale features ---
print("Scaling features...")
try:
    # Ensure all required feature columns exist after preprocessing
    missing_cols = [col for col in FEATURE_COLS if col not in new_df_features.columns]
    if missing_cols:
        print(f"Error: Missing required feature columns after preprocessing: {missing_cols}")
        print("Check the output of calculate_features and the FEATURE_COLS list.")
        exit()

    # Select only the features the scaler and model expect, in the correct order
    features_to_scale = new_df_features[FEATURE_COLS]

    # Apply the loaded scaler
    scaled_features = scaler.transform(features_to_scale)
    print("Scaling complete.")
except Exception as e:
    print(f"Error during scaling: {e}")
    exit()

# --- Prepare data for model input ---
print("Creating input sequences...")
sequences = []
# Store corresponding indices from the feature dataframe for mapping results
result_indices = []

if len(scaled_features) < SEQ_LENGTH:
    print(f"Error: Not enough data ({len(scaled_features)} rows) to form even one sequence of length {SEQ_LENGTH}.")
    exit()

# Create sequences using a sliding window
for i in range(len(scaled_features) - SEQ_LENGTH + 1):
    seq = scaled_features[i : i + SEQ_LENGTH]
    sequences.append(seq)
    # Store the index of the *last* element in the sequence from the original feature df
    # This index corresponds to the time step for which the prediction is made
    result_indices.append(new_df_features.index[i + SEQ_LENGTH - 1])

print(f"Created {len(sequences)} sequences.")

print("Performing inference on sequences...")
all_logits = []

with torch.no_grad(): # Ensure no gradients are calculated during inference
    for seq_np in sequences:
        # 1. Convert sequence to tensor
        input_tensor = torch.tensor(seq_np, dtype=torch.float32)
        # 2. Add batch dimension -> [1, SEQ_LENGTH, INPUT_DIM]
        input_tensor = input_tensor.unsqueeze(0)
        # 3. Move to device
        input_tensor = input_tensor.to(DEVICE)

        # 4. Get model output (logits)
        # The model returns (output, attention_weights), we only need output here
        logits, _ = model(input_tensor) # Shape: [1, 3]
        all_logits.append(logits.cpu()) # Move logits back to CPU for storage

# Concatenate all logits into a single tensor
all_logits_tensor = torch.cat(all_logits, dim=0) # Shape: [num_sequences, 3]
print("Inference complete.")

# --- Convert logits to probabilities ---
print("Processing results...")
# Apply softmax to get probabilities
probabilities = torch.softmax(all_logits_tensor, dim=1).numpy() # Shape: [num_sequences, 3]

# Get predicted class index for each sequence
predicted_indices = np.argmax(probabilities, axis=1) # Shape: [num_sequences]

# Map indices to labels
class_map = {0: 'Down', 1: 'Sideways', 2: 'Up'} # Assuming 0:Down, 1:Sideways, 2:Up
predicted_labels = [class_map.get(idx, 'Unknown') for idx in predicted_indices]

# Create a results DataFrame aligned with prediction timestamps
# We want OHLCV (from df_original) plus optionally the feature columns used for the model.
base_df = df_original.loc[result_indices].copy()
feature_subset = new_df_features.loc[result_indices].copy()
results_df = pd.concat([base_df, feature_subset], axis=1)

# Add prediction results
results_df['predicted_index'] = predicted_indices
results_df['predicted_label'] = predicted_labels
# results_df['prob_down'] = probabilities[:, 0]
# results_df['prob_sideways'] = probabilities[:, 1]
# results_df['prob_up'] = probabilities[:, 2]

# Start trading process
print("########################################################## Starting virtual Trading ###############################################################")
start_trading(results_df)

# # Display or save the results
# print("\nSample Predictions (showing original data alongside predictions):")
# # Select relevant columns to display (adjust based on your df columns)
# display_cols = ['target', 'predicted_index', 'predicted_label']
# print(results_df[display_cols].head())
# print(results_df[display_cols].tail())


# # Optionally save results to CSV
# output_filename = 'results/live_predictions_output.csv'
# results_df[display_cols].to_csv(output_filename, index=True) # Keep index if it's the datetime
# print(f"\nPredictions saved to {output_filename}")
