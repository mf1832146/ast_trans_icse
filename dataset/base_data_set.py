import random

import torch
import numpy as np
import torch.utils.data as data
import re
import wordninja
import string
from torch.autograd import Variable
from torch_geometric.data import Data
from torch_geometric.data.dataloader import Collater
from tqdm import tqdm
from utils import PAD, UNK
from torch.utils.data.dataset import T_co
punc = string.punctuation

__all__ = ['BaseASTDataSet', 'BaseCodeDataSet', 'PathDataSet', 'clean_nl', 'subsequent_mask', 'make_std_mask']


class PathDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(PathDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = config.data_dir + '/' + data_set_name + '/'
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len
        self.max_token_len = config.max_token_len
        self.max_path_len = config.max_path_len
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        self.collector = Collater([], [])

        src_path = data_dir + 'paths.seq'
        path_data = load_seq(src_path)
        nl_data = load_seq(data_dir + 'nl.original')

        self.data_set_len = len(nl_data)
        #self.data_set_len = 40

        self.items = self.collect_data(path_data, nl_data)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index):
        return self.items[index], self.items[index].target

    def collect_data(self, path_data, nl_data):
        items = []
        for i in tqdm(range(self.data_set_len)):
            start_vec_list = torch.zeros((self.max_src_len, self.max_token_len))
            end_vec_list = torch.zeros((self.max_src_len, self.max_token_len))
            path_vec_list = torch.zeros((self.max_src_len, self.max_path_len))

            paths = path_data[i]
            nl = nl_data[i]

            paths = paths[0].split(';')

            random.shuffle(paths)
            paths = paths[:self.max_src_len]

            for j, path in enumerate(paths):
                tokens = [p.split('|') for p in path.split(',')]
                start_tokens, path_nodes, end_tokens = [], [], []
                if len(tokens) == 1:
                    path_nodes = tokens[0]
                elif len(tokens) == 2:
                    if tokens[0][0].startswith('SimpleName'):
                        end_tokens = tokens[1]
                        path_nodes = tokens[0]
                    else:
                        start_tokens = tokens[0]
                        path_nodes = tokens[1]
                elif len(tokens) == 3:
                    start_tokens = tokens[0]
                    path_nodes = tokens[1]
                    end_tokens = tokens[2]
                else:
                    raise Exception('Unexpect tokens length, expect length <= 3.')
                start_vec = self.convert_path_to_tensor(start_tokens, self.max_token_len)
                path_vec = self.convert_path_to_tensor(path_nodes, self.max_path_len)
                end_vec = self.convert_path_to_tensor(end_tokens, self.max_token_len)

                start_vec_list[j] = start_vec
                end_vec_list[j] = end_vec
                path_vec_list[j] = path_vec

            nl_vec = self.convert_nl_to_tensor(nl)

            d = Data(start_vec=start_vec_list.long(),
                     end_vec=end_vec_list.long(),
                     path_vec=path_vec_list.long(),
                     tgt_seq=nl_vec[:-1],
                     target=nl_vec[1:])

            items.append(d)
        return items

    def convert_path_to_tensor(self, token_seq, max_len):
        token_seq = token_seq[:max_len]
        return word2tensor(token_seq, max_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_tgt_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)

    def collect_fn(self, batch):
        return self.collector.collate(batch)


class BaseCodeDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(BaseCodeDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = config.data_dir + '/' + data_set_name + '/'
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        self.collector = Collater([], [])

        if config.data_type == 'code':
            src_path = data_dir + 'code.seq'
        elif config.data_type == 'pot':
            src_path = data_dir + 'split_pot.seq' if config.is_split else 'un_split_pot.seq'
        elif config.data_type == 'sbt':
            src_path = data_dir + 'split_sbt.seq' if config.is_split else 'un_split_sbt.seq'

        if config.data_type == 'code':
            code_data = load_seq(src_path)
        else:
            code_data = load_list(src_path)
        nl_data = load_seq(data_dir + 'nl.original')

        self.items = self.collect_data(code_data, nl_data)
        self.data_set_len = len(self.items)
        # self.data_set_len = 40

    def collect_data(self, code_data, nl_data):
        items = []
        for i in tqdm(range(len(code_data))):
            code_seq = code_data[i]
            nl = nl_data[i]

            ast_vec = self.convert_code_to_tensor(code_seq)
            nl_vec = self.convert_nl_to_tensor(nl)

            d = Data(src_seq=ast_vec,
                     rel_pos=None,
                     tgt_seq=nl_vec[:-1],
                     target=nl_vec[1:])

            items.append(d)
        return items

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index):
        return self.items[index], self.items[index].target

    def convert_code_to_tensor(self, code_seq):
        code_seq = code_seq[:self.max_src_len]
        return word2tensor(code_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_tgt_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


class BaseASTDataSet(data.Dataset):
    def __init__(self, config, data_set_name):
        super(BaseASTDataSet, self).__init__()
        self.data_set_name = data_set_name
        print('loading ' + data_set_name + ' data...')
        data_dir = config.data_dir + '/' + data_set_name + '/'

        self.ignore_more_than_k = config.is_ignore
        self.max_rel_pos = config.max_rel_pos
        self.max_src_len = config.max_src_len
        self.max_tgt_len = config.max_tgt_len

        ast_path = data_dir + 'split_pot.seq' if config.is_split else 'un_split_pot.seq'
        matrices_path = data_dir + 'split_matrices.npz' if config.is_split else 'un_split_matrices.npz'

        self.ast_data = load_list(ast_path)
        self.nl_data = load_seq(data_dir + 'nl.original')
        self.matrices_data = load_matrices(matrices_path)

        self.data_set_len = len(self.ast_data)
        self.src_vocab = config.src_vocab
        self.tgt_vocab = config.tgt_vocab
        self.collector = Collater([], [])

    def collect_fn(self, batch):
        return self.collector.collate(batch)

    def __len__(self):
        return self.data_set_len

    def __getitem__(self, index) -> T_co:
        pass

    def convert_ast_to_tensor(self, ast_seq):
        ast_seq = ast_seq[:self.max_src_len]
        return word2tensor(ast_seq, self.max_src_len, self.src_vocab)

    def convert_nl_to_tensor(self, nl):
        nl = nl[:self.max_tgt_len - 2]
        nl = ['<s>'] + nl + ['</s>']
        return word2tensor(nl, self.max_tgt_len, self.tgt_vocab)


def word2tensor(seq, max_seq_len, vocab):
    seq_vec = [vocab.w2i[x] if x in vocab.w2i else UNK for x in seq]
    seq_vec = seq_vec + [PAD for i in range(max_seq_len - len(seq_vec))]
    seq_vec = torch.tensor(seq_vec, dtype=torch.long)
    return seq_vec


def load_list(file_path):
    _data = []
    print(f'loading {file_path}...')
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            _data.append(eval(line))
    return _data


def load_seq(file_path):
    data_ = []
    print(f'loading {file_path} ...')
    with open(file_path, 'r') as f:
        for line in tqdm(f.readlines()):
            data_.append(line.split())
    return data_


def load_matrices(file_path):
    print('loading matrices...')
    matrices = np.load(file_path, allow_pickle=True)
    return matrices


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
    processed_words.append('.')
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



