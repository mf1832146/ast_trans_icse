import torch.nn as nn

from module import Embeddings,  FeedForward, _get_clones, \
    SublayerConnection, DebertaRelEmbeddings, build_relative_position, DecoderLayer, BaseDecoder, Generator
from module.attn.deberta_attn import DisentangledSelfAttention
from module.base_seq2seq import BaseTrans


__all__ = ['CodeTrans']


class CodeTrans(BaseTrans):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout, state_dict=None):
        super(CodeTrans, self).__init__()
        self.num_heads = num_heads

        self.pos_type = pos_type.split('_')

        self.src_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=src_vocab_size,
                                        dropout=dropout,
                                        with_pos=False)

        self.tgt_embedding = Embeddings(hidden_size=hidden_size,
                                        vocab_size=tgt_vocab_size,
                                        dropout=dropout,
                                        with_pos=True)

        encoder_layer = RobertaEncoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout)
        self.encoder = RobertaEncoder(encoder_layer, num_layers, num_heads, self.pos_type,
                                      max_rel_pos, hidden_size, dropout=dropout)

        decoder_layer = DecoderLayer(hidden_size, self.num_heads, dim_feed_forward, dropout, activation="gelu")
        self.decoder = BaseDecoder(decoder_layer, num_layers, norm=nn.LayerNorm(hidden_size))
        self.generator = Generator(tgt_vocab_size, hidden_size, dropout)

        print('Init or load model.')
        if state_dict is None:
            for p in self.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)
        else:
            self.load_state_dict(state_dict)


class RobertaEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, num_heads, pos_type, max_rel_pos,
                 hidden_size, dropout=0.2):
        super(RobertaEncoder, self).__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(hidden_size)
        self.num_heads = num_heads
        d_k = hidden_size // num_heads
        self.max_rel_pos = max_rel_pos
        self.rel_emb = DebertaRelEmbeddings(d_k, num_heads, max_rel_pos, pos_type, dropout=dropout)

    def forward(self, data):
        output = data.src_emb
        src_mask = data.src_mask
        seq_length = output.size(1)
        rel_pos = data.rel_pos
        if rel_pos is None:
            rel_pos = build_relative_position(seq_length, seq_length, self.max_rel_pos, output.device)
        rel_q, rel_k, rel_v = self.rel_emb(rel_pos)

        for i, layer in enumerate(self.layers):
            output = layer(output, src_mask, rel_pos, rel_q, rel_k, rel_v)

        return self.norm(output)


class RobertaEncoderLayer(nn.Module):
    def __init__(self, hidden_size, num_heads, dim_feed_forward, dropout):
        super(RobertaEncoderLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_size = hidden_size
        self.self_attn = DisentangledSelfAttention(num_heads, hidden_size, dropout)
        self.feed_forward = FeedForward(hidden_size, dim_feed_forward, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

        self.sublayer = _get_clones(SublayerConnection(hidden_size, dropout), 2)

    def forward(self, src, src_mask, rel_pos, rel_q, rel_k, rel_v):
        src, attn_weights = self.sublayer[0](src, lambda x: self.self_attn(x, x, x, src_mask,
                                                                           rel_pos, rel_q, rel_k, rel_v))
        src, _ = self.sublayer[1](src, self.feed_forward)
        return src
