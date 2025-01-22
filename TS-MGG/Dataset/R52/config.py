import os
import dgl
import torch
import random
import numpy as np


class Config(object):

    def __init__(self):
        self.dataset = 'R52'

        root_dir = ''

        self.dataset_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.txt'
        self.dataset_text = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.raw.txt'
        self.dataset_clean = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '.clean.txt'

        self.vocab_path = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict.txt'
        self.vocab_embedding = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict_embeddings.txt'
        self.vocab_embedding_glove = root_dir + 'Dataset/' + self.dataset + '/' + 'vocab_dict_glove.txt'

        self.doc_embedding = root_dir + 'Dataset/' + self.dataset + '/' + 'doc_dict_embeddings.txt'

        self.is_raw_text = False

        self.device = 'cuda'

        self.batch_size = 128

        # self.words = [x.strip() for x in open(self.vocab_path, encoding='utf-8').readlines()]
        # self.embedding_pretrained = torch.tensor(np.loadtxt(self.vocab_embedding, dtype=np.float), dtype=torch.float32)
        self.words, self.embedding_pretrained = read_glove_embedding(self.vocab_embedding_glove)
        self.docs, self.embedding_pretrained_doc = read_glove_embedding(self.doc_embedding)

        # self.pading_tesor = torch.zeros([1, self.embedding_pretrained.size()[-1]], dtype=torch.float32)

        self.word_to_id = dict(zip(self.words, range(len(self.words))))
        self.n_vocab = len(self.words)
        print(f"Vocab size: {len(self.words)}")

        self.categories = sorted(list(set([x.strip().split('\t')[2] for x in open(self.dataset_dir, encoding='utf-8').readlines()])))
        self.num_classes = len(self.categories)  # 类别数
        self.cat_to_id = dict(zip(self.categories, range(len(self.categories))))
        print(self.categories)
        print("类别数量：", len(self.categories))

        self.max_length = 50

        self.each_word_out_degree_dis = 1000
        self.each_word_out_degree_pmi = 1000
        self.each_word_out_degree_top = 1000

        self.each_doc_out_degree_dis = 50
        self.each_doc_out_degree_pmi = 50
        self.each_doc_out_degree_top = 50

        self.PMI_windows_size = 60

        self.num_topics = 50
        self.lda_epoch = 500
        self.topic_model_random_state = 0
        self.topic_model_dir = root_dir + 'Dataset/' + self.dataset + '/LDA/' + self.dataset + '.lda_model.topic_' + str(self.num_topics) + '.gensim'

        self.word_word_Dis_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_Dis_Graph_' + str(self.each_word_out_degree_dis) + '.dgl'
        self.word_word_Pmi_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_PMI_Graph_' + str(self.each_word_out_degree_pmi) + '.dgl'
        self.word_word_Topic_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_word_to_word_LDA_Graph_' + str(self.each_word_out_degree_top) + '.dgl'

        self.doc_doc_Dis_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_doc_to_doc_Dis_Graph_' + str(
            self.each_doc_out_degree_dis) + '.dgl'
        self.doc_doc_Pmi_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_doc_to_doc_PMI_Graph_' + str(
            self.each_doc_out_degree_pmi) + '.dgl'
        self.doc_doc_Topic_dir = root_dir + 'Dataset/' + self.dataset + '/' + self.dataset + '_doc_to_doc_LDA_Graph_' + str(
            self.each_doc_out_degree_top) + '.dgl'

        if os.path.exists(self.word_word_Dis_dir):
            (self.g_word_word_Dis,), _ = dgl.load_graphs(self.word_word_Dis_dir)
            # self.g_word_word_Dis = self.g_word_word_Dis.to('cuda')
        if os.path.exists(self.word_word_Pmi_dir):
            (self.g_word_word_Pmi,), _ = dgl.load_graphs(self.word_word_Pmi_dir)
            # self.g_word_word_PMI = self.g_word_word_PMI.to('cuda')
        if os.path.exists(self.word_word_Topic_dir):
            (self.g_word_word_Top,), _ = dgl.load_graphs(self.word_word_Topic_dir)
            # self.g_word_word_Top = self.g_word_word_Top.to('cuda')
        if os.path.exists(self.doc_doc_Dis_dir):
            (self.g_doc_doc_Dis,), _ = dgl.load_graphs(self.doc_doc_Dis_dir)
            # self.g_doc_doc_Dis = self.g_doc_doc_Dis.to('cuda')
        if os.path.exists(self.doc_doc_Pmi_dir):
            (self.g_doc_doc_Pmi,), _ = dgl.load_graphs(self.doc_doc_Pmi_dir)
            # self.g_doc_doc_PMI = self.g_doc_doc_PMI.to('cuda')
        if os.path.exists(self.doc_doc_Topic_dir):
            (self.g_doc_doc_Top,), _ = dgl.load_graphs(self.doc_doc_Topic_dir)
            # self.g_doc_doc_Top = self.g_doc_doc_Top.to('cuda')

        self.GCN_hidden_in_feat_dim = 300
        self.GCN_hidden_out_feat_dim = 300

        self.LSTM_hidden_in_feat_dim = 300
        self.LSTM_hidden_out_feat_dim = 300
        self.layer_num = 2
        self.droup_out_lstm = 0.5

        self.classifer_in_feat_dim = 300


def read_glove_embedding(read_dir):
    data = [x.strip().split(' ') for x in open(read_dir, encoding='utf-8').readlines()]
    words = [x[0] for x in data]
    embedding = []
    for i in range(len(data)):
        e = [float(x) for x in data[i][1:]]
        embedding.append(e)
    embedding_pretrained = torch.tensor(embedding, dtype=torch.float32)
    return words, embedding_pretrained
