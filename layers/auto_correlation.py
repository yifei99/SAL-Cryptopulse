import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoCorrelationMH(nn.Module):
    def __init__(self, d_model, n_heads):
        super(AutoCorrelationMH, self).__init__()

        self.n_heads = n_heads
        # split to multi-heads
        d_K = d_model // n_heads
        d_V = d_model // n_heads
        self.q_mh = nn.Linear(d_model, d_K * self.n_heads)
        self.k_mh = nn.Linear(d_model, d_K * self.n_heads)
        self.v_mh = nn.Linear(d_model, d_V * self.n_heads)
        self.output = nn.Linear(d_V * n_heads, d_model)

    def time_delay_agg(self, values, corr):
        # (batch, n_heads, d_V, len)
        values = values.permute(0, 2, 3, 1).contiguous()
        if self.training:
            values = self.time_delay_agg_training(values, corr)
        else:
            values = self.time_delay_agg_inference(values, corr)
        # (batch, len, n_heads, d_V)
        values = values.permute(0, 3, 1, 2)
        return values

    def time_delay_agg_training(self, values, corr):
        heads, channels, length = values.size()[1:]
        r_qk = corr.mean(dim=1).mean(dim=1)  # (batch, len)
        topk = torch.topk(
            r_qk.mean(dim=0), int(torch.log(torch.tensor(length))), dim=-1
        )[1]
        top_r_qk = r_qk.index_select(1, topk)
        corr_ = torch.softmax(top_r_qk, dim=-1)
        delays_agg = [
            values.roll(-tao.item(), -1)
            * corr_[:, i].view(-1, 1, 1, 1).repeat(1, heads, channels, length)
            for i, tao in enumerate(topk)
        ]
        delays_agg = torch.stack(delays_agg).sum(dim=0)
        return delays_agg

    def time_delay_agg_inference(self, values, corr):
        batch, num_heads, channels, length = values.size()
        r_qk = corr.mean(dim=1).mean(dim=1)  # (batch, len)
        corr_, topk = torch.topk(
            r_qk, int(torch.log(torch.tensor(length))), dim=-1
        )
        corr_ = torch.softmax(corr_, dim=-1)
        init_index = (
            torch.arange(length)
            .view(1, 1, 1, -1)
            .repeat(batch, num_heads, channels, 1)
            .to(values.device)
        )
        values_ = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(topk.size(1)):
            roll_index = init_index + topk[:, i].view(-1, 1, 1, 1).repeat(
                1, num_heads, channels, length
            )
            pattern = torch.gather(values_, dim=-1, index=roll_index)
            delays_agg += pattern * corr_[:, i].view(-1, 1, 1, 1).repeat(
                1, num_heads, channels, length
            )
        return delays_agg

    def forward(self, Q, K, V):
        # split to multi-heads (batch, len, n_heads, d_q / d_k / d_v)
        Q = self.q_mh(Q).view(Q.size(0), Q.size(1), self.n_heads, -1)
        K = self.k_mh(K).view(K.size(0), K.size(1), self.n_heads, -1)
        V = self.v_mh(V).view(V.size(0), V.size(1), self.n_heads, -1)
        # align the lengths of Q and V
        diff = Q.size(1) - V.size(1)
        if diff > 0:  # Q is longer, pad V and K with 0s at the end of each seq
            V, K = (F.pad(tmp, (0, 0, 0, 0, 0, diff, 0, 0)) for tmp in (V, K))
        elif diff < 0:  # V is longer, trim V and K
            V, K = (tmp[:, : Q.size(1), :, :] for tmp in (V, K))
        # identify period-based dependencies
        r_qk = cross_correlation_fast(
            Q.permute(0, 2, 3, 1).contiguous(), K.permute(0, 2, 3, 1).contiguous()
        )
        # time delay aggregation
        V = self.time_delay_agg(V, r_qk)
        out = self.output(V.contiguous().view(Q.size(0), Q.size(1), -1))
        return out, r_qk.permute(0, 3, 1, 2)


def cross_correlation_fast(series1, series2):
    s1 = torch.fft.rfft(series1, dim=-1)
    s2 = torch.fft.rfft(series2, dim=-1)
    return torch.fft.irfft(s1 * torch.conj(s2), dim=-1)
