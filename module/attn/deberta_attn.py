import torch
import math
import torch.nn.functional as F
import torch.nn as nn

__all__ = ['DisentangledSelfAttention']

from module import _get_clones, transpose_for_scores, build_relative_position


class DisentangledSelfAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(DisentangledSelfAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linear_layers = _get_clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, rel_pos=None,
                rel_q=None, rel_k=None, rel_v=None):
        query, key, value = [transpose_for_scores(l(x), self.h) for l, x in zip(self.linear_layers,
                                                                                (query, key, value))]

        output, _ = self.rel_attn(query, key, value, mask,
                                  rel_pos, rel_q, rel_k, rel_v)

        output = output.permute(0, 2, 1, 3).contiguous()
        new_value_shape = output.size()[:-2] + (-1,)
        output = output.view(*new_value_shape)
        output = self.linear_layers[-1](output)

        return output, None

    def rel_attn(self, q, k, v, mask=None, rel_pos=None, rel_q=None, rel_k=None, rel_v=None):
        assert q.size(-2) == k.size(-2)
        batch_size, num_heads, length, per_head = q.size()
        if rel_pos.dim() == 3:
            rel_pos = rel_pos.unsqueeze(1)
        rel_pos = rel_pos.expand([batch_size, num_heads, length, length])

        scores = torch.matmul(q, k.transpose(-1, -2))
        scale_factor = 1
        if rel_q is not None:
            c2p_att = torch.matmul(q, rel_k.transpose(-1, -2))
            c2p_att = torch.gather(c2p_att, dim=-1, index=rel_pos)
            scores += c2p_att
            scale_factor += 1
        if rel_k is not None:
            p2c_att = torch.matmul(k, rel_q.transpose(-1, -2))
            p2c_att = torch.gather(p2c_att, dim=-1, index=rel_pos.transpose(-1, -2)).transpose(-1, -2)
            scores += p2c_att
            scale_factor += 1

        scores = scores / (scale_factor * math.sqrt(per_head))
        if mask is not None:
            # 给需要mask的地方设置一个负无穷（因为接下来要输入到softmax层，如果是0还是会有影响）
            if mask.dim() == 2:
                mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(mask, -1e9)

        p_attn = F.softmax(scores, dim=-1)  # [batch_size, num_heads, len, len]
        p_attn = self.dropout(p_attn)

        # 和V做点积
        context = torch.matmul(p_attn, v)  # [batch_size ,num_head, node_len, node_len]
        if rel_v is not None:
            # relative_v shape [batch_size, node_len, node_len, dim]
            new_x_shape = rel_v.size()[:-1] + (num_heads, -1)
            rel_v = rel_v.view(*new_x_shape)
            rel_v = rel_v.permute(0, 3, 1, 2, 4)
            v_attn = p_attn.unsqueeze(-2)
            context_v = torch.matmul(v_attn, rel_v).squeeze(-2)
            context += context_v

        return context, p_attn



