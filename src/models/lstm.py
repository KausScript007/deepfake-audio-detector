import torch
import torch.nn as nn
class AudioLSTMClassifier(nn.Module):
    def __init__(
        self,
        input_size: int = 40,
        hidden_size: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        # 1. LSTM Temporal Backbone
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        # 2. Classifier Head
        classifier_input_dim = hidden_size * self.num_directions
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),  # Output 1 logit
        )
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Parameters:
        -----------
        x : torch.Tensor
            Batch of MFCC sequences of shape (batch_size, time_frames, input_size).
        Returns:
        --------
        logits : torch.Tensor
            Raw output logits of shape (batch_size, 1).
        """
        # x shape: (B, T=126, D=40)
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out shape: (B, T, hidden_size * num_directions)
        # h_n shape: (num_layers * num_directions, B, hidden_size)
        if self.bidirectional:
            # Concat the last hidden state of forward direction and backward direction
            # Forward last: h_n[-2, :, :], Backward last: h_n[-1, :, :]
            last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)  # shape: (B, hidden_size * 2)
        else:
            last_hidden = h_n[-1]  # shape: (B, hidden_size)
        # Pass through classification head
        logits = self.head(last_hidden)  # shape: (B, 1)
        return logits
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Inference helper: converts logits to probability of class 1 (FAKE).
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities
