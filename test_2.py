
#  if config.is_split else 'un_split_matrices.npz'
from tqdm import tqdm
import matplotlib.pyplot as plt
from dataset.base_data_set import load_matrices

data_dir = '../data_set/processed/java/train/'
matrices_path = data_dir + 'split_matrices.npz'
matrices_data = load_matrices(matrices_path)

par_edge_data = matrices_data['parent']
bro_edge_data = matrices_data['brother']


def edge2list(edges):
    edge_nums = {}
    for key in edges.keys():
        value = edges.get(key)
        if value in edge_nums:
            edge_nums[value] += 1
        else:
            edge_nums[value] = 1

    return edge_nums


par = {i: 0 for i in range(30)}
bro = {i: 0 for i in range(30)}

max_len_p = 0
max_len_b = 0

for i in tqdm(range(len(par_edge_data))):
    par_edges = par_edge_data[i]
    bro_edges = bro_edge_data[i]

    par_edge_list = edge2list(par_edges)
    bro_edge_list = edge2list(bro_edges)

    for k, v in par_edge_list.items():
        if k > max_len_p:
            max_len_p = k
        if k in par:
            par[k] += v
        else:
            par[k] = v

    for k, v in bro_edge_list.items():
        if k > max_len_b:
            max_len_b = k
        if k in bro:
            bro[k] += v
        else:
            bro[k] = v

new_p = {}
new_b = {}

for k in par.keys():
    if par[k] > 0:
        if k <= 20:
            new_p[k] = par[k]
        else:
            new_p[20] += par[k]
for k in bro.keys():
    if bro[k]>0:
        if k <= 5:
            new_b[k]=bro[k]
        else:
            new_b[5] += bro[k]

print('new_p', new_p)
print('new_b', new_b)

fig, (ax0, ax1) = plt.subplots(2, 1)

ax0.bar(new_p.keys(), new_p.values())
ax1.bar(new_b.keys(), new_b.values())
plt.show()