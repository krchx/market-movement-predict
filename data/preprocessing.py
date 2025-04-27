import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import joblib

def load_and_clean_data(file_path):
    """Load and clean the OHLCV data."""
    df = pd.read_csv(file_path)
    
    # Rename columns to lowercase if they exist with capital letters
    column_mapping = {
        'Timestamp': 'timestamp',
        'Open': 'open',
        'High': 'high', 
        'Low': 'low',
        'Close': 'close',
        'Volume': 'volume'
    }
    df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
    
    # Convert timestamp to datetime if it exists
    if 'timestamp' in df.columns:
        # Convert Unix timestamp to datetime
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
        # If not shorted, sort by datetime
        # df = df.sort_values('datetime')
    
    # Drop any duplicates or NaN values
    df = df.drop_duplicates()
    df = df.dropna()
    
    return df

def calculate_features(df):
    """Calculate all required features from the OHLCV data."""

    # percentage change for open, high, low, close --> four features
    for col in ['open', 'high', 'low', 'close']:
        if col in df.columns:
            df[f'{col}_return'] = df[col].pct_change()
    
    # Volume log-transformation, log(volume + 1) --> one feature
    if 'volume' in df.columns:
        df['volume_change'] = np.log1p(df['volume'])
    
    # Volume Velocity (% change in Volume from previous minute) --> one feature
    if 'volume' in df.columns:
        df['volume_velocity'] = df['volume'].pct_change()
    
    # Relative Position in Range ((C-L)/(H-L)) --> one feature
    if 'high' in df.columns and 'low' in df.columns and 'close' in df.columns:
        df['relative_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])

    # Body/Wick Ratio (C-O)/(H-L) --> one feature
    if 'high' in df.columns and 'low' in df.columns and 'open' in df.columns and 'close' in df.columns:
        df['body_wick_ratio'] = (df['close'] - df['open']) / (df['high'] - df['low'])

    # Cyclical time features, from 3:45 to 10:00, sin and cos --> two features
    minutes_in_day = 10 * 60 - (3 * 60 + 45)
    minute_of_day = (df['datetime'].dt.hour - 3) * 60 + (df['datetime'].dt.minute - 45)
    df['minute_sin'] = np.sin(2 * np.pi * minute_of_day / minutes_in_day)
    df['minute_cos'] = np.cos(2 * np.pi * minute_of_day / minutes_in_day)

    # Cyclical day of year features, sin and cos --> two features
    days_in_year = 365
    day_of_year = df['datetime'].dt.dayofyear
    df['day_sin'] = np.sin(2 * np.pi * day_of_year / days_in_year)
    df['day_cos'] = np.cos(2 * np.pi * day_of_year / days_in_year)
    
    # Drop rows with NaN (first row after pct_change)
    df = df.dropna()
    
    return df

def create_target(df, horizon=5, threshold=0.0015):
    """
    Create target labels for price direction prediction based on future price movement.

    Args:
        df (pd.DataFrame): DataFrame with 'high', 'low', 'close' columns.
        horizon (int): Number of future minutes to look ahead.
        threshold (float): Percentage threshold for price movement (e.g., 0.001 for 0.1%).

    Returns:
        pd.DataFrame: DataFrame with the added 'target' column and rows with NaN targets dropped.
    """
    # Explicitly create a copy to avoid SettingWithCopyWarning
    df = df.copy()

    # Calculate the max high and min low over the next 'horizon' periods
    # Shift(-horizon) looks into the future
    future_max_high = df['high'].rolling(window=horizon).max().shift(-horizon)
    future_min_low = df['low'].rolling(window=horizon).min().shift(-horizon)

    # Calculate the threshold bounds based on the current close
    upper_bound = df['close'] * (1 + threshold)
    lower_bound = df['close'] * (1 - threshold)

    # Define conditions for target labels
    conditions = [
        (future_max_high - df['close'] >= df['close'] - future_min_low) & (future_max_high >= upper_bound),  # Price goes up significantly
        (df['close'] - future_min_low > future_max_high - df['close']) & (future_min_low <= lower_bound),  # Price goes down significantly
    ]

    # Define choices corresponding to conditions: 2 for up, 0 for down
    choices = [2, 0]  # 2 for up, 0 for down

    # Apply conditions using np.select, default is 1 (sideways)
    # Assign directly to the copied DataFrame
    df['target'] = np.select(conditions, choices, default=1)

    # Drop last 'horizon' rows where target cannot be calculated
    df = df[:-horizon]

    # Ensure target is integer type
    df['target'] = df['target'].astype(np.int64)

    df = df.dropna()  # Drop any rows with NaN values in the target column

    return df

def split_features(df):
    """Create two different dataframes: one with input features and other with all features."""
    # Create a copy of the DataFrame to avoid modifying the original
    df_features = df.copy()

    # Create a df with (timestamp, open, high, low, close, volume)
    df_original = df_features[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    
    # Remove columns (timestamp, open, high, low, close, volume, datetime) in df_features
    columns_to_remove = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'datetime']
    df_features.drop(columns=columns_to_remove, inplace=True, errors='ignore')
    
    return df_features, df_original


def split_data(df, train_size=0.7, val_size=0.15):
    """Split data chronologically into train, validation, and test sets."""
    n = len(df)
    train_end = int(n * train_size)
    val_end = int(n * (train_size + val_size))
    
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    test_original_indices = df.index[val_end:].tolist()
    
    return train_df, val_df, test_df, test_original_indices

def scale_features(train_df, val_df, test_df):
    """Scale the features using StandardScaler fit on training data."""
    # Get numerical feature columns (exclude target and datetime)
    feature_cols = train_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ['target']]
    
    # Fit scaler on training data
    scaler = StandardScaler()
    scaler.fit(train_df[feature_cols])

    # Save the scaler as scaler.joblib
    joblib.dump(scaler, 'data/scaler.joblib')
    
    # Transform all datasets
    train_df[feature_cols] = scaler.transform(train_df[feature_cols])
    val_df[feature_cols] = scaler.transform(val_df[feature_cols])
    test_df[feature_cols] = scaler.transform(test_df[feature_cols])
    
    return train_df, val_df, test_df, feature_cols

def create_sequences(df, seq_length=60):
    """Create sequences for time series model input."""
    feature_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    feature_cols = [col for col in feature_cols if col not in ['target']]
    
    # Convert to numpy arrays
    data_array = df[feature_cols].values
    target_array = df['target'].values
    
    X, y = [], []
    for i in range(len(df) - seq_length + 1):
        X.append(data_array[i:i+seq_length])
        y.append(target_array[i+seq_length-1])
        
    return np.array(X), np.array(y)

def create_dataloaders(X_train, y_train, X_val, y_val, X_test, y_test, batch_size=32):
    """Create PyTorch DataLoaders for train, validation, and test sets."""
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def prepare_data(file_path, seq_length=60, horizon=5, batch_size=32):
    """Full data preparation pipeline."""
    # Load and clean data
    df = load_and_clean_data(file_path)
    
    # Calculate features
    df = calculate_features(df)
    
    # Create target
    df = create_target(df, horizon=horizon)

    # Split features for model input
    df_features, df_original = split_features(df)
    
    # Split data
    train_df, val_df, test_df, test_original_indices  = split_data(df_features, train_size=0.7, val_size=0.15)

    # Ensure the test set is aligned with the original data
    if len(test_original_indices) != len(test_df):
        print(f"Warning: Length mismatch between test_original_indices ({len(test_original_indices)}) and test_df ({len(test_df)}).")
    
    # Scale features
    train_df, val_df, test_df, feature_cols = scale_features(train_df, val_df, test_df)
    
    # Create sequences
    X_train, y_train = create_sequences(train_df, seq_length=seq_length)
    X_val, y_val = create_sequences(val_df, seq_length=seq_length)
    X_test, y_test = create_sequences(test_df, seq_length=seq_length)

    # Align original indices & features with sequences
    aligned_test_original_indices = test_original_indices[seq_length-1:]
    aligned_test_original_df = df_original.loc[aligned_test_original_indices]

    # Ensure the test set is aligned with the original data
    if len(y_test) != len(aligned_test_original_df):
         print(f"Warning: Length mismatch between y_test ({len(y_test)}) and aligned_test_original_df ({len(aligned_test_original_df)}). Truncating.")
         min_len = min(len(y_test), len(aligned_test_original_df))
         y_test = y_test[:min_len]
         X_test = X_test[:min_len]
         aligned_test_original_df = aligned_test_original_df[:min_len]
    
    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, X_test, y_test, batch_size=batch_size
    )
    
    input_dim = X_train.shape[2]
    
    return train_loader, val_loader, test_loader, aligned_test_original_df, input_dim