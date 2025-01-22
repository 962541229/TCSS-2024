import dgl
import torch
import numpy as np
from dgl.nn.pytorch.factory import KNNGraph
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# def construct_dis_graph(config):
#     kg_words = KNNGraph(config.each_word_out_degree)
#     g = kg_words(config.embedding_pretrained, dist='euclidean')
#     g = dgl.add_self_loop(g)
#     g = dgl.to_simple(g)
#     dgl.save_graphs(config.word_word_Dis_dir, g)


def euclidean_distance(vec1, vec2):

    return np.sqrt(np.sum((vec1 - vec2) ** 2))


def top_n_indices(lst, n=100):
    indices = sorted(range(len(lst)), key=lambda i: lst[i], reverse=False)[:n]
    return indices


def construct_dis_graph_numpy(config):
    words, embedding_pretraineds = config.words, np.array(config.embedding_pretrained, dtype=np.float32)

    s_nodes, g_nodes = [], []

    stop_word_indices = [i for i, w in enumerate(words) if w in stop_words]

    dist_matrix = np.einsum('ij,kj->ik', embedding_pretraineds, embedding_pretraineds)
    dist_matrix *= -2.0
    dist_matrix += np.sum(np.square(embedding_pretraineds), axis=1, keepdims=True)
    dist_matrix += np.sum(np.square(embedding_pretraineds), axis=1)

    for i in range(len(words)):
        if i not in stop_word_indices:
            distances = dist_matrix[i]
            distances[stop_word_indices] = float('inf')
            indices = top_n_indices(distances, n=config.each_word_out_degree_dis)

            s_nodes.extend(indices)
            g_nodes.extend([i]*len(indices))

    g = dgl.graph((s_nodes, g_nodes), num_nodes=len(config.words))
    g = dgl.add_self_loop(g)
    g = dgl.to_simple(g)

    dgl.save_graphs(config.word_word_Dis_dir, g)