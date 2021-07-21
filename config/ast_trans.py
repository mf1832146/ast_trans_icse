from pathlib import Path
from dataset.fast_ast_data_set import FastASTDataSet
from module import FastASTTrans
import os
from utils import LabelSmoothing, PAD

use_clearml = True
project_name = 'ast_trans_fast'
task_name = 'ast_trans_fast'

seed = 2021
# data
data_dir = '../data_set/processed/java'
max_tgt_len = 30
max_src_len = 150
data_type = 'pot'

is_split = True
is_test = False

# model
hype_parameters = {
    'pos_type': 'p2q_p2k_p2v',  # ['', 'p2q_p2k', 'p2q_p2k_p2v']
    'par_heads': 8,   # [0,8]
    'max_rel_pos': 1,  # [1, 3, 5, 7]
    'num_layers': 6,  # [2, 4, 6]
    'data_dir': '../data_set/processed/py',  # java, py
    'is_split': True,  # need split
    'is_test': False
}


num_heads = 8
pos_type = ''
par_heads = 0
max_rel_pos = 3
max_par_rel_pos = 3
max_bro_rel_pos = 3
num_layers = 4
hidden_size = 256
dim_feed_forward = 2048
is_ignore = True
dropout = 0.2

# train
batch_size = 64
num_epochs = 200
num_threads = 2
config_filepath = Path('./config/ast_trans.py')
es_patience = 20
load_epoch_path = ''
val_interval = 5
data_set = FastASTDataSet
model = FastASTTrans
fast_mod = False
logger = ['tensorboard', 'clear_ml']

# optimizer
learning_rate = 1e-3
warmup = 0.01

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.1)

g = '0'
# src_vocab, _, tgt_vocab = load_vocab(data_dir, is_split)
#
checkpoint = None
# if load_epoch_path != '':
#     file_name = load_epoch_path
#     if torch.cuda.is_available():
#         checkpoint = torch.load(file_name)
#     else:
#         checkpoint = torch.load(file_name, map_location='cpu')








