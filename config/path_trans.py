from pathlib import Path

from dataset import PathDataSet
from module import Code2SeqTrans
from utils import LabelSmoothing, PAD

use_clearml = True
project_name = 'baselines'
task_name = 'path_trans'
is_split = True  # only action when data_type in ['sbt', 'pot']
is_test = False

seed = 2021
# data
data_dir = '../data_set/processed/java'
max_tgt_len = 30
max_src_len = 150
max_token_len = 5
max_path_len = 9
data_type = 'path'

num_heads = 8
pos_type = 'p2q_p2k_p2v'
max_rel_pos = 5
num_layers = 4
hidden_size = 256
dim_feed_forward = 2048
is_ignore = True
dropout = 0.2

# train
batch_size = 4
num_epochs = 500
num_threads = 2
config_filepath = Path('./config/path_trans.py')
es_patience = 5
load_epoch_path = ''
val_interval = 1
data_set = PathDataSet
model = Code2SeqTrans
fast_mod = False
logger = ['tensorboard', 'clear_ml']

# optimizer
learning_rate = 1e-3
warmup = 0.01

# criterion
criterion = LabelSmoothing(padding_idx=PAD, smoothing=0.1)
checkpoint = None
