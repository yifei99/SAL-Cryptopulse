import torch
import torch.nn as nn

from layers.emb_layers import InputEmbedding
from layers.auto_correlation import AutoCorrelationMH


class CryptoPulse(nn.Module):
    def __init__(self, configs):
        super(CryptoPulse, self).__init__()

        ob_len = configs.ob_len
        pred_len = configs.pred_len
        d_model = configs.d_model

        # handle market sentiment
        self.sent_embed = InputEmbedding(3, d_model, 0.05)
        self.sent_proj = nn.Sequential(nn.Linear(ob_len, pred_len), nn.GELU())
        self.sent_trend = nn.Sequential(nn.Linear(d_model, 1), nn.Tanh())
        # embedding for target crypto
        self.Q_embed = InputEmbedding(12, d_model, 0.05)
        # embedding for macro market environment
        self.K_embed = InputEmbedding(25, d_model, 0.05)
        # cross-correlation & FF
        self.cross_correlation = AutoCorrelationMH(d_model, configs.n_heads)
        self.ff = nn.Sequential(
            nn.Conv1d(
                in_channels=d_model, out_channels=2048, kernel_size=1, bias=False
            ),
            nn.GELU(),
            nn.Dropout(0.05),
            nn.Conv1d(
                in_channels=2048, out_channels=d_model, kernel_size=1, bias=False
            ),
            nn.Dropout(0.05),
        )
        # macro market environment -> change 1
        self.proj1 = nn.Sequential(nn.Linear(ob_len, pred_len), nn.GELU())
        self.delta1 = nn.Sequential(nn.Linear(d_model, 1), nn.Softplus())
        # target crypto -> change 2
        self.delta2 = nn.Sequential(nn.Linear(ob_len, pred_len), nn.Softplus())
        # ombination of two predictions
        self.gate_ = nn.Sequential(nn.Linear(ob_len, pred_len), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, x):
        Q = torch.cat([x[:, :, :11], x[:, :, -1:]], dim=-1)  # target
        K = x[:, :, 14:-1]  # macro market environment
        S = x[:, :, 11:14]  # sentiment
        ob_seq_last = x[:, -1:, :]
        # sentiment scaler prediction
        S_ = self.sent_embed(S, None)
        sent = self.sent_proj(S_.permute(0, 2, 1)).permute(0, 2, 1)
        sent_trend = self.sent_trend(sent)
        # macro market environment driven prediction
        Q_, K_ = self.Q_embed(Q, None), self.K_embed(K, None)
        V, _ = self.cross_correlation(Q_, K_, K_)
        V = self.ff(V.transpose(-1, 1)).transpose(-1, 1)
        delta1 = self.delta1(self.proj1(V.permute(0, 2, 1)).permute(0, 2, 1))
        pred1 = ob_seq_last + delta1 * sent_trend
        # target crypto driven prediction
        x = x - ob_seq_last
        delta2 = self.delta2(x.permute(0, 2, 1)).permute(0, 2, 1)
        pred2 = ob_seq_last + delta2 * sent_trend
        # combination
        environment = Q_ + S_
        gamma = self.gate(self.gate_(environment.permute(0, 2, 1)).permute(0, 2, 1))
        pred = gamma * pred1 + (1 - gamma) * pred2
        return pred
