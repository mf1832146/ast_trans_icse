import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data.dataloader import Collater

from utils import PAD, UNK
from torch.utils.data.dataset import T_co
punc = string.punctuation

__all__ = ['BaseASTDataSet', 'clean_nl', 'subsequent_mask', 'make_std_mask', 'get_data_set']


def get_data_set(config):
    train_data_set = config.data_set(config.data_dir, config.max_src_len, config.max_tgt_len,
                                     config.max_rel_pos, config.is_ignore, 'train', config.src_vocab, config.tgt_vocab)
    eval_data_set = config.data_set(config.data_dir, config.max_src_len, config.max_tgt_len,
                                    config.max_rel_pos, config.is_ignore, 'dev', config.src_vocab, config.tgt_vocab)

    return train_data_set, eval_data_set


class BaseASTDataSet(data.Dataset):
    def __init__(self,  data_dir, max_ast_size, max_nl_len, max_rel_pos, is_ignore, data_set_name, ast_vocab, nl_vocab):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = data_dir + '/' + data_set_name + '/'

        self.ignore_more_than_k = is_ignore
        self.max_rel_pos = max_rel_pos
        self.max_ast_size = max_ast_size
        self.max_nl_len = max_nl_len

        self.ast_data = self.load_ast(data_dir + 'root_first.seq')
        self.nl_data = self.load_nl(data_dir + 'nl.original')
        self.matrices_data = self.load_matrices(data_dir + 'matrices.npz')

        self.data_set_len = len(self.ast_data)
        self.ast_vocab = ast_vocab
        self.nl_vocab = nl_vocab
        self.collector = Collater([], [])

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[:self.max_ast_size]
        ast_vec = [self.ast_vocab.w2i[x] if x in self.ast_vocab.w2i else UNK for x in ast_seq]
        ast_vec = ast_vec + [PAD for i in range(self.max_ast_size - len(ast_vec))]
        ast_vec = torch.tensor(ast_vec, dtype=torch.long)
        return ast_vec

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_nl_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        nl_vec = [self.nl_vocab.w2i[x] if x in self.nl_vocab.w2i else UNK for x in nl]
        nl_vec = nl_vec + [PAD for i in range(self.max_nl_len - len(nl_vec))]
        nl_vec = torch.tensor(nl_vec, dtype=torch.long)
        return nl_vec

    @staticmethod
    def load_ast(file_path):
        _data = []
        print('loading asts...')
        with open(file_path, 'r') as f:
            for line in f.readlines():
                _data.append(eval(line))
        return _data

    @staticmethod
    def load_matrices(file_path):
        print('loading matrices...')
        matrices = np.load(file_path, allow_pickle=True)
        return matrices

    @staticmethod
    def load_nl(file_path):
        data_ = []
        print('loading nls...')
        with open(file_path, 'r') as f:
            for line in f.readlines():
                nl_ = clean_nl(line)
                if len(nl_) >= 2:
                    data_.append(nl_)
                else:
                    data_.append(line.split())
        return data_


def clean_nl(s):
    s = s.strip()
    s = re.sub("[<].+?[>]", "", s)
    s = re.sub("[\[\]\%]", "", s)
    s = s[0:1].lower() + s[1:]
    processed_words = []
    for w in s.split():
        if w not in punc:
            processed_words.extend(wordninja.split(w))
        else:
            processed_words.append(w)
    return processed_words


def subsequent_mask(size):
    attn_shape = (1, size, size)
    sub_sequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(sub_sequent_mask) != 0


def make_std_mask(nl, pad):
    "Create a mask to hide padding and future words."
    nl_mask = (nl == pad).unsqueeze(-2)
    nl_mask = nl_mask | Variable(
        subsequent_mask(nl.size(-1)).type_as(nl_mask.data))
    return nl_mask



