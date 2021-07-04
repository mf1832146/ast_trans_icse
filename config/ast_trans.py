import torch.nn as nn
from py_config_runner import Schema
import torch.utils.data as data
from dataset.fast_ast_data_set import FastASTDataSet
from utils import Vocab


class ASTTrans(Schema):
    seed: int

    data_set: data.Dataset
    src_vocab: Vocab
    tgt_vocab: Vocab

    model: nn.Module

    max_src_len: int
    max_tgt_len: int
    data_type: str

    # model
    par_heads: list
    num_heads: int
    pos_type: list
    max_rel_pos: list

    schema: Schema


project_name = 'ast_trans_fast'
task_name = 'ast_trans_fast'

seed = 2021
# data
data_dir = '../data_set/java'
max_tgt_len = 50
max_src_len = 200
data_type = 'ast'

# model
pos_type = ['', 'p2q', 'p2q_p2k', 'p2q_p2k_p2v']
par_heads = range(0, 8)
num_heads = 8
max_rel_pos = [1, 3, 5, 7]

hidden_size = 512
dim_feed_forward = 2048
num_layers = [2, 4, 6]
dropout = 0.2

data_set = FastASTDataSet
model = ?






