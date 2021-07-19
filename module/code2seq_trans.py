import torch
import torch.nn as nn

from dataset import make_std_mask
from module import BaseTrans
from module.code_trans import RobertaEncoderLayer, RobertaEncoder
from module.components import PositionalEncoding, DecoderLayer, BaseDecoder, Generator, Embeddings, process_data
from utils import PAD

__all__ = ['Code2SeqTrans']


class Code2SeqTrans(BaseTrans):
    def __init__(self, src_vocab_size, tgt_vocab_size, hidden_size, num_heads,
                 max_rel_pos, pos_type, num_layers, dim_feed_forward, dropout, state_dict=None):
        super(Code2SeqTrans, self).__init__()
        self.num_heads = num_heads
        self.pos_type = pos_type.split('_')

        self.src_embedding = Code2SeqEmbeddings(hidden_size=hidden_size,
                                                vocab_size=src_vocab_size,
                                                num_heads=self.num_heads,
                                                dropout=dropout)

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

    def process_data(self, data):
        process_data(data)

        start_vec = data.start_vec
        end_vec = data.end_vec
        path_vec = data.path_vec  # [batch_size, max_src_len, max_path_len]

        data.src_mask = torch.sum(path_vec, dim=-1).eq(PAD)
        data.src_emb = self.src_embedding(start_vec, path_vec, end_vec)

        if data.tgt_seq is not None:
            tgt_seq = data.tgt_seq
            data.tgt_mask = make_std_mask(tgt_seq, PAD)
            data.tgt_emb = self.tgt_embedding(tgt_seq)


class Code2SeqEmbeddings(nn.Module):
    def __init__(self, hidden_size, vocab_size, num_heads, dropout):
        super(Code2SeqEmbeddings, self).__init__()
        """
        start_token, end_token embedding size is hidden_size // 4
        path_token embedding size is hidden // 2
        it is just to keep the dim of output same as hidden_size
        """
        self.node_embeddings = nn.Embedding(vocab_size, hidden_size // 2, padding_idx=PAD)
        self.token_linear = nn.Embedding(hidden_size // 2, hidden_size // 4)
        self.pos_emb = PositionalEncoding(hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.path_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.dropout = dropout

    def get_embed(self, tokens, need_pos=False, need_linear=False):
        token_emb = self.node_embeddings(tokens)
        if need_pos and self.pos_emb is not None:
            token_emb = self.pos_emb(token_emb)
        if need_linear:
            token_emb = self.token_linear(token_emb)
        token_emb = self.norm(token_emb)
        token_emb = self.dropout(token_emb)
        return token_emb

    def forward(self, start_tokens, path_tokens, end_tokens):
        """
        :param start_tokens: [batch_size, max_src_len, max_token_len]
        :param path_tokens: [batch_size, max_src_len, max_path_len]
        :param end_tokens: [batch_size, max_src_len, max_token_len]
        :return: output: [batch_size, max_src_len, hidden_size]
        """

        #  1. emb start_tokens and end_tokens, then sum them up
        start_token_emb = self.get_embed(start_tokens, need_pos=False)
        end_token_emb = self.get_embed(end_tokens, need_pos=False)

        start_token_emb = torch.sum(start_token_emb, dim=-1)
        end_token_emb = torch.sum(end_token_emb, dim=-1)

        #  2. use transformer encoder to encode path_tokens, then use the first encode output
        batch_size, max_src_len, max_path_len = path_tokens.size()
        path_tokens = path_tokens.view(batch_size * max_src_len, -1)
        path_token_mask = path_tokens.eq(PAD)
        path_token_emb = self.get_embed(path_tokens, need_pos=True)
        path_token_emb = path_token_emb.permute(1, 0, 2)
        path_encode_output = self.path_encoder(src=path_token_emb, src_key_padding_mask=path_token_mask)
        path_encode_output = path_encode_output.permute(1, 0, 2)

        path_encode_output = path_encode_output.view(batch_size, max_src_len, max_path_len, -1)
        path_encode_output = path_encode_output[:, :, 0, :]

        # 3. concat [start_token_emb, path_encode_output, end_token_emb]
        output = torch.cat([start_token_emb, path_encode_output, end_token_emb], dim=-1)

        return output
