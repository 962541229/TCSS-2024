import dgl
import torch
import math
import numpy as np
import scipy.sparse as sp
from collections import Counter
from utils import process_dataset_sequence
from dgl.nn.pytorch.conv import SAGEConv
from scipy.sparse import dok_matrix
from nltk.corpus import stopwords

stop_words = stopwords.words('english')


# def construct_pmi_graph(config):
#     word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
#     train_indexs = np.zeros(y_train.shape[0])
#
#     word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
#     test_indexs = np.ones(y_test.shape[0])
#
#     labels_total = np.concatenate((y_train, y_test)).astype(np.int)
#     word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
#     length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.int)
#
#     '''
#     word-word pmi graph
#     '''
#
#     # word co-occurence with context windows
#     windows = []
#
#     for i in range(word_documents_adjmatrix_total.shape[0]):
#         words = word_documents_adjmatrix_total[i][0:length_id_total[i][0]]
#         length = len(words)
#         if length <= config.PMI_windows_size:
#             windows.append(words)
#         else:
#             # print(length, length - window_size + 1)
#             for j in range(length - config.PMI_windows_size + 1):
#                 window = words[j: j + config.PMI_windows_size]
#                 windows.append(window)
#
#     word_window_freq = {}
#     for window in windows:
#         appeared = set()
#         for i in range(len(window)):
#             if window[i] in appeared:
#                 continue
#             if window[i] in word_window_freq:
#                 word_window_freq[window[i]] += 1
#             else:
#                 word_window_freq[window[i]] = 1
#             appeared.add(window[i])
#
#     word_pair_count = {}
#     for window in windows:
#         for i in range(1, len(window)):
#             for j in range(0, i):
#                 if window[i] == window[j]:
#                     continue
#                 word_pair_str = str(window[i]) + ',' + str(window[j])
#                 if word_pair_str in word_pair_count:
#                     word_pair_count[word_pair_str] += 1
#                 else:
#                     word_pair_count[word_pair_str] = 1
#                 # two orders
#                 word_pair_str = str(window[j]) + ',' + str(window[i])
#                 if word_pair_str in word_pair_count:
#                     word_pair_count[word_pair_str] += 1
#                 else:
#                     word_pair_count[word_pair_str] = 1
#
#     s_nodes = []
#     g_nodes = []
#     weight = []
#     # pmi as weights
#
#     num_window = len(windows)
#
#     for key in word_pair_count:
#         temp = key.split(',')
#         i = int(temp[0])
#         j = int(temp[1])
#         count = word_pair_count[key]
#         word_freq_i = word_window_freq[i]
#         word_freq_j = word_window_freq[j]
#         pmi = math.log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
#         if pmi <= 0:
#             continue
#         s_nodes.append(i)
#         g_nodes.append(j)
#         weight.append(pmi)
#
#     g = dgl.graph((s_nodes, g_nodes), num_nodes=len(config.words))
#     g = dgl.add_self_loop(g)
#     g = dgl.to_simple(g)
#
#     # g.edata['w'] = torch.tensor(weight, dtype=torch.float32)
#     # if g.num_edges() != g.edata['w'].size():
#     #     print('边的特征发生错误')
#     dgl.save_graphs(config.word_word_Pmi_dir, g)
#     (g,), _ = dgl.load_graphs(config.word_word_Pmi_dir)
#     # print(g.edata['w'].shape)
#     # print(g.number_of_edges())
#     # print(g.number_of_nodes())
#     #
#     # feature = torch.randn(g.number_of_nodes(), 300)
#     # SAGEConv = SAGEConv(300, 300, aggregator_type='mean')
#     #
#     # result = SAGEConv(g, feature, edge_weight=g.edata['w'])
#     # print(result)


def construct_pmi_graph_remove_stopword(config):
    word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
    train_indexs = np.zeros(y_train.shape[0])

    word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
    test_indexs = np.ones(y_test.shape[0])

    labels_total = np.concatenate((y_train, y_test)).astype(np.int)
    word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
    length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.int)

    '''
    word-word pmi graph
    '''

    # word co-occurence with context windows
    windows = []

    for i in range(word_documents_adjmatrix_total.shape[0]):
        words = word_documents_adjmatrix_total[i][0:length_id_total[i][0]]
        length = len(words)

        if length <= config.PMI_windows_size:
            windows.append(words)
        else:
            for j in range(length - config.PMI_windows_size + 1):
                window = words[j: j + config.PMI_windows_size]
                windows.append(window)



    word_window_freq = {}
    for window in windows:
        appeared = set()
        for i in range(len(window)):
            if window[i] in appeared:
                continue
            if window[i] in word_window_freq:
                word_window_freq[window[i]] += 1
            else:
                word_window_freq[window[i]] = 1
            appeared.add(window[i])

    word_pair_count = {}
    for window in windows:
        for i in range(1, len(window)):
            for j in range(0, i):
                if window[i] == window[j]:
                    continue
                word_pair_str = str(window[i]) + ',' + str(window[j])
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1
                # two orders
                word_pair_str = str(window[j]) + ',' + str(window[i])
                if word_pair_str in word_pair_count:
                    word_pair_count[word_pair_str] += 1
                else:
                    word_pair_count[word_pair_str] = 1

    s_nodes = []
    d_nodes = []
    weight = []

    pmi_value = {}
    # pmi as weights

    num_window = len(windows)

    for key in word_pair_count:
        temp = key.split(',')
        i = int(temp[0])
        j = int(temp[1])
        count = word_pair_count[key]
        word_freq_i = word_window_freq[i]
        word_freq_j = word_window_freq[j]
        pmi = math.log((1.0 * count / num_window) / (1.0 * word_freq_i * word_freq_j / (num_window * num_window)))
        if pmi <= 0:
            continue
        s_nodes.append(i)
        d_nodes.append(j)
        weight.append(pmi)



    # for stop_word in stop_words:
    #     stop_word_id = config.word_to_id[stop_word]
    #     stop_word_subscript =

    # g = dgl.graph((s_nodes, d_nodes), num_nodes=len(config.words))
    # g = dgl.add_self_loop(g)
    # g = dgl.to_simple(g)

    # g.edata['w'] = torch.tensor(weight, dtype=torch.float32)
    # if g.num_edges() != g.edata['w'].size():
    #     print('边的特征发生错误')

    # dgl.save_graphs(config.word_word_Pmi_dir, g)
    # (g,), _ = dgl.load_graphs(config.word_word_Pmi_dir)

    # print(g.edata['w'].shape)
    # print(g.number_of_edges())
    # print(g.number_of_nodes())
    #
    # feature = torch.randn(g.number_of_nodes(), 300)
    # SAGEConv = SAGEConv(300, 300, aggregator_type='mean')
    #
    # result = SAGEConv(g, feature, edge_weight=g.edata['w'])
    # print(result)


def construct_pmi_graph_remove_stopword_ours(config):
    word_documents_adjmatrix_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
    word_documents_adjmatrix_test, y_test, length_id_test = process_dataset_sequence(config, 'test')

    word_documents_adjmatrix_total = np.concatenate((word_documents_adjmatrix_train, word_documents_adjmatrix_test), axis=0)
    length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.int)

    # word co-occurence with context windows
    windows = []
    for i in range(word_documents_adjmatrix_total.shape[0]):
        words = word_documents_adjmatrix_total[i][0:length_id_total[i][0]]
        length = len(words)

        if length <= config.PMI_windows_size:
            windows.append(words)
        else:
            for j in range(length - config.PMI_windows_size + 1):
                window = words[j: j + config.PMI_windows_size]
                windows.append(window)

    freqs = Counter()
    for window in windows:
        freqs.update({w: 1 for w in window})

    co_occurrences = Counter()
    for window in windows:
        for i, w1 in enumerate(window):
            for j, w2 in enumerate(window):
                if i == j:
                    continue
                if w1 != w2:
                    co_occurrences[(w1, w2)] += 1

    doc_length = len(windows)

    pmi_matrix = np.zeros((len(config.words), len(config.words)), dtype=np.float32)

    for (w1, w2), co_occurrence_count in co_occurrences.items():
        if config.words[w1] not in stop_words and config.words[w2] not in stop_words:
            w1_count = freqs[w1]
            w2_count = freqs[w2]
            pmi = math.log2((co_occurrence_count / doc_length) / ((w1_count / doc_length) * (w2_count / doc_length)))
            if pmi > 0:
                pmi_matrix[w1, w2] = pmi
    return pmi_matrix


def get_top_n_indices_row(matrix, row_idx, n):
    row_values = matrix[row_idx]
    indices = np.arange(len(row_values))

    sorted_indices = np.argsort(row_values)[::-1]

    return indices[sorted_indices[:n]]


def get_top_n_nonzero_indices_row(matrix, row_idx, n):

    row_values = matrix[row_idx]
    indices = np.arange(len(row_values))

    nonzero_indices = indices[row_values.nonzero()]
    if len(nonzero_indices) < n:
        n = len(nonzero_indices)

    sorted_indices = np.argsort(row_values[nonzero_indices])[::-1][:n]

    return nonzero_indices[sorted_indices]


def construct_pmi_graph(config):
    s_nodes = []
    d_nodes = []
    pmi_matrix = construct_pmi_graph_remove_stopword_ours(config)

    for (i, word) in enumerate(config.words):
        if word not in stop_words:
            indices = get_top_n_nonzero_indices_row(pmi_matrix, i, config.each_word_out_degree_pmi)
            s_nodes.extend(indices)
            d_nodes.extend([i]*len(indices))

    g = dgl.graph((s_nodes, d_nodes), num_nodes=len(config.words))
    g = dgl.add_self_loop(g)
    g = dgl.to_simple(g)

    dgl.save_graphs(config.word_word_Pmi_dir, g)
