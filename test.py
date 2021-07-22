import random
import numpy as np
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt


def generate_rand_tree(tree_size):
    par_adj = np.zeros((tree_size, tree_size))
    bro_adj = np.zeros((tree_size, tree_size))

    prufer = [random.randint(0, tree_size-1) for i in range(tree_size - 2)]

    vertices = tree_size
    vertex_set = [0] * tree_size
    for i in range(vertices - 2):
        vertex_set[prufer[i] - 1] += 1

    for i in range(tree_size - 2):
        for j in range(tree_size):
            if vertex_set[j] == 0:
                vertex_set[j] = -1
                par_adj[prufer[i] - 1][j] = 1
                vertex_set[prufer[i] - 1] -= 1
                break

    j = 0

    # For the last element
    end_node = -1
    for i in range(vertices):
        if vertex_set[i] == 0 and j == 0:
            end_node = i
            j += 1
        elif vertex_set[i] == 0 and j == 1:
            par_adj[i][end_node] = 1

    for i in range(vertices):
        current_node = -1
        for j in range(vertices):
            if par_adj[i][j] == 1:
                if current_node > 0:
                    bro_adj[current_node][j] = 1
                current_node = j

    return par_adj, bro_adj


x = []
y = []

fig, (ax0, ax1) = plt.subplots(2, 1)

if __name__ == '__main__':
    k = 20
    run_times = 1000

    results_x = [[] for i in range(k)]
    results_y = [[] for i in range(k)]
    results_z = [[] for i in range(k)]
    for tree_size in range(100, 120):
        par_num = [0] * k
        bro_num = [0] * k
        for a in tqdm(range(run_times)):
            par_ajd, bro_adj = generate_rand_tree(tree_size)
            tmp_par = par_ajd
            tmp_bro = bro_adj
            for cur_k in range(2, k+1):
                tmp_par = np.matmul(tmp_par, par_ajd)
                tmp_bro = np.matmul(tmp_bro, bro_adj)

                par_num[cur_k-1] += np.sum(tmp_par)
                bro_num[cur_k-1] += np.sum(tmp_bro)
        par_num = [p / run_times for p in par_num]
        bro_num = [b / run_times for b in bro_num]

        for cur_k in range(k):
            results_x[cur_k].append(tree_size)
            results_y[cur_k].append(par_num[cur_k])
            results_z[cur_k].append(bro_num[cur_k])

    for cur_k in range(1, k):
        ax0.plot(results_x[cur_k], results_y[cur_k], label='par_k={}'.format(cur_k+1))
        ax1.plot(results_x[cur_k], results_z[cur_k], label='bro_k={}'.format(cur_k + 1))

    ax0.legend()
    ax1.legend()
    plt.show()