import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionMechanism(nn.Module):
    def __init__(self, hidden_size):
        super(AttentionMechanism, self).__init__()
        self.attention = nn.Linear(hidden_size, 1)
        
    def forward(self, rnn_output):
        # rnn_output shape: (batch, seq_len, hidden_size)
        attn_weights = self.attention(rnn_output)  # (batch, seq_len, 1)
        attn_weights = F.softmax(attn_weights, dim=1)  # (batch, seq_len, 1)
        
        # Apply attention weights to RNN output
        context = torch.sum(rnn_output * attn_weights, dim=1)  # (batch, hidden_size)
        
        return context, attn_weights

class HybridCNNGRUAttentionModel(nn.Module):
    def __init__(self, input_dim, cnn_channels=64, rnn_hidden=128, dropout=0.3, use_lstm=False):
        super(HybridCNNGRUAttentionModel, self).__init__()
        
        # CNN blocks for feature extraction
        self.conv1 = nn.Conv1d(input_dim, cnn_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(cnn_channels)
        self.conv2 = nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_channels)
        self.dropout = nn.Dropout(dropout)
        
        # RNN layer (GRU or LSTM)
        if use_lstm:
            self.rnn = nn.LSTM(
                input_size=cnn_channels,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
        else:
            self.rnn = nn.GRU(
                input_size=cnn_channels,
                hidden_size=rnn_hidden,
                num_layers=1,
                batch_first=True,
                bidirectional=False
            )
        
        # Attention mechanism
        self.attention = AttentionMechanism(rnn_hidden)
        
        # Output layer
        self.fc = nn.Linear(rnn_hidden, 3)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        batch_size = x.shape[0]
        
        # Permute for CNN (batch, features, seq_len)
        x = x.permute(0, 2, 1)
        
        # CNN feature extraction
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        
        # Permute back for RNN (batch, seq_len, features)
        x = x.permute(0, 2, 1)
        
        # RNN sequence processing
        rnn_out, _ = self.rnn(x)
        
        # Apply attention
        context, attn_weights = self.attention(rnn_out)
        
        # Final prediction
        out = self.fc(context)
        
        return out, attn_weights
