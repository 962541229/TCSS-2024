import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch.factory import KNNGraph
from multi_head_attention import Multi_Head_Attention, Scaled_Dot_Product_Attention

import dgl.function as fn
from wzj_GATconv import GATConv


class SAGEConv(nn.Module):
    def __init__(self, in_feat, out_feat):
        super(SAGEConv, self).__init__()
        # A linear submodule for projecting the input and neighbor feature to the output.
        self.linear = nn.Linear(in_feat * 2, out_feat)
        self.dropout = nn.Dropout(0.8)
        self.layer_norm = nn.LayerNorm(out_feat)

    def forward(self, block, h_s, h_d):
        with block.local_scope():
            block.srcdata['h'] = h_s
            block.dstdata['h'] = h_d
            block.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            out = self.linear(torch.cat([block.dstdata['h'], block.dstdata['h_neigh']], 1))
            out = self.dropout(out)
            # out = h_d + out
            return out


class Merge_Model(nn.Module):
    def __init__(self, config):
        super(Merge_Model, self).__init__()

        self.config = config
        self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        self.embedding_doc = nn.Embedding.from_pretrained(config.embedding_pretrained_doc, freeze=False)

        self.SAGEConv_Dis = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)
        self.SAGEConv_PMI = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)
        self.SAGEConv_Top = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)

        self.SAGEConv_Dis_doc = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)
        self.SAGEConv_PMI_doc = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)
        self.SAGEConv_Top_doc = SAGEConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim)

        # self.GATConv_Top = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)
        # self.GATConv_PMI = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)
        # self.GATConv_Top = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)
        # self.GATConv_Top_doc = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)
        # self.GATConv_PMI_doc = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)
        # self.GATConv_Top_doc = GATConv(config.GCN_hidden_in_feat_dim, config.GCN_hidden_out_feat_dim, num_heads=10)

        # self.attention = Multi_Head_Attention(300, 2)
        self.scaled_dot_product_attention = Scaled_Dot_Product_Attention()
        self.scaled_dot_product_attention_doc = Scaled_Dot_Product_Attention()

        # a wide Multi-Layer Perceptron network, the model uses only one wide hidden layer
        self.dense_layer = nn.Linear(300, 300)

        # lstm
        # self.lstm = nn.LSTM(300, 300, 2, batch_first=True, dropout=0.5)

        self.fc = nn.Linear(config.classifer_in_feat_dim, config.num_classes)

    def forward(self, blocks, blocks_doc, x_batch, length_batch, return_doc_representation):
        (block_Dis, block_Pmi, block_Top) = blocks
        (block_Dis_doc, block_Pmi_doc, block_Top_doc) = blocks_doc

        dst_nodes_old = block_Dis.dstdata[dgl.NID]
        graph_dst_features = self.embedding(dst_nodes_old)

        src_nodes_old_Dis = block_Dis.srcdata[dgl.NID]
        src_nodes_old_Pmi = block_Pmi.srcdata[dgl.NID]
        src_nodes_old_Top = block_Top.srcdata[dgl.NID]

        graph_src_features_Dis = self.embedding(src_nodes_old_Dis)
        graph_src_features_Pmi = self.embedding(src_nodes_old_Pmi)
        graph_src_features_Top = self.embedding(src_nodes_old_Top)

        word_Dis_h = self.SAGEConv_Dis(block_Dis, graph_src_features_Dis, graph_dst_features)
        word_Pmi_h = self.SAGEConv_PMI(block_Pmi, graph_src_features_Pmi, graph_dst_features)
        word_Top_h = self.SAGEConv_Top(block_Top, graph_src_features_Top, graph_dst_features)

        cat_tensor = torch.cat((word_Dis_h, word_Pmi_h, word_Top_h), dim=1).reshape(word_Dis_h.size()[0], 3, word_Dis_h.size()[-1])

        # doc_embedding = torch.mean(cat_tensor, dim=-2)
        doc_embedding = self.scaled_dot_product_attention(cat_tensor, cat_tensor, cat_tensor, scale=cat_tensor.size(-1) ** -0.5)
        # doc_embedding = self.attention(cat_tensor)

        graph_word_embeddings = torch.cat([torch.squeeze(doc_embedding) + graph_dst_features,  # word embeddings
                                           doc_embedding.new_zeros([1, doc_embedding.size()[-1]])], dim=0)

        doc_embedding_sequence = F.embedding(x_batch, graph_word_embeddings)

        # a wide Multi-Layer Perceptron network, the model uses only one wide hidden layer
        doc_out = self.dense_layer(doc_embedding_sequence)
        doc_out = torch.mean(doc_out, dim=1, keepdim=False)


        # lstm
        # output, (hidden, cell) = self.lstm(doc_embedding_sequence)
        # indexs = length_batch.unsqueeze(-1).expand(-1, -1, output.size()[-1])
        # doc_out = torch.gather(output, dim=1, index=indexs-1).squeeze()

        src_nodes_old_Dis_doc = block_Dis_doc.srcdata[dgl.NID]
        src_nodes_old_Pmi_doc = block_Pmi_doc.srcdata[dgl.NID]
        src_nodes_old_Top_doc = block_Top_doc.srcdata[dgl.NID]

        graph_src_features_Dis_doc = self.embedding_doc(src_nodes_old_Dis_doc)
        graph_src_features_Pmi_doc = self.embedding_doc(src_nodes_old_Pmi_doc)
        graph_src_features_Top_doc = self.embedding_doc(src_nodes_old_Top_doc)

        doc_Dis_h = self.SAGEConv_Dis_doc(block_Dis_doc, graph_src_features_Dis_doc, doc_out)
        doc_Pmi_h = self.SAGEConv_PMI_doc(block_Pmi_doc, graph_src_features_Pmi_doc, doc_out)
        doc_Top_h = self.SAGEConv_Top_doc(block_Top_doc, graph_src_features_Top_doc, doc_out)

        cat_tensor_doc = torch.cat((doc_Dis_h, doc_Pmi_h, doc_Top_h), dim=1).reshape(doc_Dis_h.size()[0], 3, doc_Dis_h.size()[-1])

        doc_out_gnn = self.scaled_dot_product_attention_doc(cat_tensor_doc, cat_tensor_doc, cat_tensor_doc, scale=cat_tensor_doc.size(-1) ** -0.5)
        doc_out_residual = doc_out_gnn + doc_out

        gcn_re = self.fc(doc_out_residual)

        if return_doc_representation:
            return doc_out_residual
        else:
            return gcn_re
