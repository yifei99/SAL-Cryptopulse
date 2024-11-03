import math

import torch
import torch.nn as nn


class InputEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, dropout_rate):
        super(InputEmbedding, self).__init__()

        self.value_embed = SeriesConvEmbedding(in_channels, d_model)
        self.position_embed = LenExpDecayPositionalEmbedding(5000, d_model)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x_data, x_time):
        x = self.dropout(self.value_embed(x_data) + self.position_embed(x_data))
        return x


class LenExpDecayPositionalEmbedding(nn.Module):
    """Positional encoding with exponential decay along the temporal dimension."""

    def __init__(self, max_len, d_model):
        super(LenExpDecayPositionalEmbedding, self).__init__()

        pe = exp_decay_positional_encoding(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x_data):
        return self.pe[:, : x_data.size(1), :]


class SeriesConvEmbedding(nn.Module):
    def __init__(self, in_channels, d_model, kernel_size=3):
        super(SeriesConvEmbedding, self).__init__()

        self.series_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=d_model,
            kernel_size=kernel_size,
            padding="same",
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x_data):
        return self.series_conv(x_data.permute(0, 2, 1)).transpose(1, 2)


def exp_decay_positional_encoding(pos, out_channels):
    c = math.log(1e4) / out_channels
    freq = torch.exp(c * (-torch.arange(0, out_channels, 2).float()))
    # calculate pe
    PE = torch.zeros(pos, out_channels, requires_grad=False).float()
    pos_v = torch.arange(0, pos).float().unsqueeze(1)
    PE[:, 0::2] = torch.sin(pos_v * freq)
    PE[:, 1::2] = torch.cos(pos_v * freq)
    return PE
