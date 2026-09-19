import torch
import torch.nn as nn


class AudioLSTMClassifier(nn.Module):

    def __init__(
        self,
        input_dim: int = 40,
        hidden_dim: int = 128,
        num_layers: int = 2,
        bidirectional: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # 1. Recurrent Temporal Backbone (BiLSTM)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # 2. MLP Classification Head
        classifier_input_dim = hidden_dim * self.num_directions
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(classifier_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
  
        lstm_out, (h_n, c_n) = self.lstm(x)

        if self.bidirectional:
            last_hidden = torch.cat(
                (h_n[-2], h_n[-1]), dim=1
            )  # shape: (B, hidden_dim * 2)
        else:
            last_hidden = h_n[-1]  # shape: (B, hidden_dim)

        logits = self.head(last_hidden)  # shape: (B, 1)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = torch.sigmoid(logits)
        return probabilities