import os
import time
from datetime import timedelta

from build_word_word_Topic_graph import construct_topic_graph
from build_word_word_PMI_graph import construct_pmi_graph
from build_word_word_Dis_graph import construct_dis_graph_numpy

from Dataset.ng20.config import Config
# from Dataset.mr.config import Config
# from Dataset.ohsumed.config import Config
# from Dataset.R8.config import Config
# from Dataset.R52.config import Config
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


start_time = time.time()

each_word_out_degree = []
for i in range(len(each_word_out_degree)):

    config = Config()

    config.each_word_out_degree_dis = each_word_out_degree[i]
    config.each_word_out_degree_pmi = each_word_out_degree[i]
    config.each_word_out_degree_top = each_word_out_degree[i]

    config.word_word_Dis_dir = 'Dataset/' + config.dataset + '/' + config.dataset + '_word_to_word_Dis_Graph_' + str(
        config.each_word_out_degree_dis) + '.dgl'
    config.word_word_Pmi_dir = 'Dataset/' + config.dataset + '/' + config.dataset + '_word_to_word_PMI_Graph_' + str(
        config.each_word_out_degree_pmi) + '.dgl'
    config.word_word_Topic_dir = 'Dataset/' + config.dataset + '/' + config.dataset + '_word_to_word_LDA_Graph_' + str(
        config.each_word_out_degree_top) + '.dgl'

    print("each_word_out_degree:", each_word_out_degree[i])

    construct_dis_graph_numpy(config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print('dis_graph_over!')

    construct_topic_graph(config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print('topic_graph_over!')

    construct_pmi_graph(config)
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)
    print('pmi_graph_over!')

