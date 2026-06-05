import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=5, padding=2, pool_kernel=2):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel)

    def forward(self, x):
        # x: (batch, channels, time)
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.pool(x)
        return x


class Attention(nn.Module):
    """
    Simple additive attention over time.
    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim)  (weighted sum)
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x):
        # x: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attn(x))          # (batch, seq_len, hidden_dim)
        scores = self.v(scores).squeeze(-1)        # (batch, seq_len)
        weights = torch.softmax(scores, dim=1)     # (batch, seq_len)
        # weighted sum: (batch, hidden_dim)
        context = torch.bmm(weights.unsqueeze(1), x).squeeze(1)
        return context, weights


class CNNLSTM(nn.Module):
    """
    CNN -> BiLSTM -> Attention -> FC classifier for multichannel time-series.
    Input: (batch, channels, time_steps)
    """
    def __init__(
        self,
        in_channels=4,
        conv_channels=(32, 64, 128),
        conv_kernel_size=5,
        conv_pool=2,
        lstm_hidden=128,
        lstm_layers=1,
        fc_hidden=128,
        dropout=0.3,
        num_classes=2,          # 2 classes (baseline, stress)
        bidirectional=True,     # BiLSTM
        freeze_cnn=False
    ):
        super().__init__()

        # Build conv blocks
        blocks = []
        prev_ch = in_channels
        for ch in conv_channels:
            blocks.append(
                ConvBlock1D(
                    prev_ch,
                    ch,
                    kernel_size=conv_kernel_size,
                    padding=conv_kernel_size // 2,
                    pool_kernel=conv_pool,
                )
            )
            prev_ch = ch
        self.cnn = nn.Sequential(*blocks)

        # LSTM
        self.bidirectional = bidirectional
        self.lstm_input_size = prev_ch  # channels after last conv

        self.lstm = nn.LSTM(
            input_size=self.lstm_input_size,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,      # (batch, seq_len, features)
            bidirectional=bidirectional,
        )

        self.lstm_directions = 2 if bidirectional else 1
        lstm_out_dim = lstm_hidden * self.lstm_directions

        # Attention over LSTM outputs
        self.attention = Attention(lstm_out_dim)

        # Fully connected layers
        self.fc1 = nn.Linear(lstm_out_dim, fc_hidden)
        self.dropout = nn.Dropout(dropout)
        self.fc_out = nn.Linear(fc_hidden, num_classes)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.Linear, nn.LSTM)):
                # default init is fine for now
                pass

    def forward(self, x):
        """
        x: (batch, channels, time_steps)
        returns logits: (batch, num_classes)
        """
        # CNN
        x = self.cnn(x)                      # (batch, channels_out, time_out)
        x = x.permute(0, 2, 1).contiguous()  # (batch, seq_len, features)

        # LSTM
        lstm_out, _ = self.lstm(x)           # (batch, seq_len, hidden*dirs)

        # Attention over time
        context, _ = self.attention(lstm_out)  # (batch, hidden*dirs)

        # FC
        x = self.fc1(context)
        x = F.relu(x)
        x = self.dropout(x)
        logits = self.fc_out(x)
        return logits

    def predict_proba(self, x):
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=-1)
        return probs


if __name__ == "__main__":
    # quick shape test
    model = CNNLSTM(in_channels=4, num_classes=2, bidirectional=True)
    dummy = torch.randn(8, 4, 7000)  # batch, channels, time
    out = model(dummy)
    print("Out shape:", out.shape)  # should be (8, 2)
