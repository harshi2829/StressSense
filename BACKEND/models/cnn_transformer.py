import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNTransformer(nn.Module):
    """
    CNN → Transformer Encoder → FC classifier for 4‑channel physiological time series.

    Input shape:
        x: (batch_size, in_channels=4, time_steps)

    Output:
        logits: (batch_size, num_classes)
    """

    def __init__(
        self,
        in_channels: int = 4,
        conv_channels=(32, 64, 128),
        conv_kernel_size: int = 5,
        conv_pool: int = 2,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()

        # -----------------------------
        # CNN feature extractor
        # -----------------------------
        conv_layers = []
        prev_c = in_channels
        for c in conv_channels:
            conv_layers.append(
                nn.Conv1d(
                    in_channels=prev_c,
                    out_channels=c,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                )
            )
            conv_layers.append(nn.BatchNorm1d(c))
            conv_layers.append(nn.ReLU(inplace=True))
            conv_layers.append(nn.MaxPool1d(kernel_size=conv_pool))
            conv_layers.append(nn.Dropout(dropout))
            prev_c = c

        self.conv_net = nn.Sequential(*conv_layers)
        self.conv_out_channels = prev_c

        # Project CNN channels to Transformer d_model if needed
        if self.conv_out_channels != d_model:
            self.to_d_model = nn.Linear(self.conv_out_channels, d_model)
        else:
            self.to_d_model = nn.Identity()

        # -----------------------------
        # Positional encoding
        # -----------------------------
        self.positional_encoding = SinusoidalPositionalEncoding(d_model=d_model, dropout=dropout)

        # -----------------------------
        # Transformer encoder
        # -----------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,  # input: (batch, seq, d_model)
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_transformer_layers,
        )

        # -----------------------------
        # Classification head
        # -----------------------------
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, channels=4, time_steps)
        """
        # CNN: (B, C_in, T) -> (B, C_out, T_reduced)
        x = self.conv_net(x)  # (B, C_out, T_reduced)

        # Permute to (B, T_reduced, C_out)
        x = x.permute(0, 2, 1)  # (B, T, C_out)

        # Project channels to d_model if needed
        x = self.to_d_model(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.positional_encoding(x)  # (B, T, d_model)

        # Transformer encoder
        x = self.transformer(x)  # (B, T, d_model)

        # Pool over time (mean pooling)
        x = x.mean(dim=1)  # (B, d_model)

        # Classification head
        logits = self.fc(x)  # (B, num_classes)
        return logits


class SinusoidalPositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding, adapted for batch_first inputs:
        x: (batch, seq_len, d_model)
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 10000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )  # (d_model/2)

        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        # Register as buffer so it moves with .to(device) but is not a parameter
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)
