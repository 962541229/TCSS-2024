import os
import dgl
import time
import torch
import random
import numpy as np
from sklearn import metrics
from model import Merge_Model

import torch.nn.functional as F
from dgl.nn.pytorch.factory import KNNGraph
from utils import process_dataset_sequence, split_train_val, split_train_val_own, get_time_dif

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def batch_iter(x, y, z, batch_size=128):
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]
    z_shuffle = z[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id], z_shuffle[start_id:end_id]


def evaluate(config, model, val_x, val_y, val_length, test=False):
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)

    batch_val = batch_iter(val_x, val_y, val_length)
    with torch.no_grad():
        for x_batch, y_batch, length_batch in batch_val:
            batch_nodes = torch.unique(x_batch)
            y_batch = y_batch.to(config.device)
            length_batch = length_batch.to(config.device)

            frontier_Dis = dgl.in_subgraph(config.g_word_word_Dis, batch_nodes[0:-1])
            block_Dis = dgl.to_block(frontier_Dis, batch_nodes[0:-1]).to(config.device)

            frontier_Pmi = dgl.in_subgraph(config.g_word_word_Pmi, batch_nodes[0:-1])
            block_Pmi = dgl.to_block(frontier_Pmi, batch_nodes[0:-1]).to(config.device)

            frontier_Top = dgl.in_subgraph(config.g_word_word_Top, batch_nodes[0:-1])
            block_Top = dgl.to_block(frontier_Top, batch_nodes[0:-1]).to(config.device)

            for i in range(batch_nodes.size()[0]):
                x_batch[x_batch == batch_nodes[i]] = i
            x_batch = x_batch.to(config.device)

            outputs = model((block_Dis, block_Pmi, block_Top), x_batch, length_batch)
            loss = F.cross_entropy(outputs, y_batch)
            loss_total += loss

            y_batch = y_batch.data.cpu().numpy()
            predict = torch.max(outputs.data, 1)[1].cpu().numpy()
            labels_all = np.append(labels_all, y_batch)
            predict_all = np.append(predict_all, predict)

    acc = metrics.accuracy_score(labels_all, predict_all)

    if test:
        report = metrics.classification_report(labels_all, predict_all, target_names=config.categories, digits=4)
        confusion = metrics.confusion_matrix(labels_all, predict_all)
        return acc, loss_total / len(val_x), report, confusion

    return acc, loss_total / len(val_x)


def train(config, model):
    word_documents_train, y_train, length_id_train = process_dataset_sequence(config, 'train')
    train_indexs = np.zeros(y_train.shape[0])

    word_documents_test, y_test, length_id_test = process_dataset_sequence(config, 'test')
    test_indexs = np.ones(y_test.shape[0])

    labels_total = np.concatenate((y_train, y_test)).astype(np.int)
    word_documents_total = np.concatenate((word_documents_train, word_documents_test), axis=0)
    length_id_total = np.concatenate((length_id_train, length_id_test)).astype(np.float)

    test_mask = np.concatenate((train_indexs, test_indexs)).astype(np.bool)
    train_mask = ~test_mask
    val_mask, real_train_mask = split_train_val_own(train_mask, labels_total, config.num_classes)

    test_mask = torch.tensor(test_mask, dtype=torch.bool)

    labels_total = torch.LongTensor(labels_total)
    word_documents_total = torch.LongTensor(word_documents_total)
    length_id_total = torch.LongTensor(length_id_total)

    real_train_x, val_x, test_x = word_documents_total[real_train_mask], word_documents_total[val_mask], word_documents_total[test_mask]
    real_train_y, val_y, test_y = labels_total[real_train_mask], labels_total[val_mask], labels_total[test_mask]
    real_train_length, val_length, test_length = length_id_total[real_train_mask], length_id_total[val_mask], length_id_total[test_mask]

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    total_batch = 0
    best_val_acc = 0
    best_test_acc = 0
    epoch_num = 50

    for e in range(epoch_num):
        print('Epoch [{}/{}]'.format(e + 1, epoch_num))
        batch_train = batch_iter(real_train_x, real_train_y, real_train_length, batch_size=config.batch_size)
        for x_batch, y_batch, length_batch in batch_train:
            model.train()

            batch_nodes = torch.unique(x_batch)

            frontier_Dis = dgl.in_subgraph(config.g_word_word_Dis, batch_nodes[0:-1])
            block_Dis = dgl.to_block(frontier_Dis, batch_nodes[0:-1]).to(config.device)

            frontier_Pmi = dgl.in_subgraph(config.g_word_word_Pmi, batch_nodes[0:-1])
            block_Pmi = dgl.to_block(frontier_Pmi, batch_nodes[0:-1]).to(config.device)

            frontier_Top = dgl.in_subgraph(config.g_word_word_Top, batch_nodes[0:-1])
            block_Top = dgl.to_block(frontier_Top, batch_nodes[0:-1]).to(config.device)

            for i in range(batch_nodes.size()[0]):
                x_batch[x_batch == batch_nodes[i]] = i
            x_batch = x_batch.to(config.device)

            y_batch = y_batch.to(config.device)
            length_batch = length_batch.to(config.device)

            model.zero_grad()
            outputs = model((block_Dis, block_Pmi, block_Top), x_batch, length_batch)

            loss = F.cross_entropy(outputs, y_batch)
            loss.backward()
            optimizer.step()

            if total_batch % 1 == 0:
                true = y_batch.data.cpu()
                predict = torch.max(outputs.data, 1)[1].cpu()
                train_acc = metrics.accuracy_score(true, predict)

                dev_acc, dev_loss = evaluate(config, model, val_x, val_y, val_length)
                test_acc, test_loss = evaluate(config, model, test_x, test_y, test_length)

                if dev_acc > best_val_acc:
                    best_val_acc = dev_acc
                    # torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},' \
                      '  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%}, Test Loss: {5:>5.2},  Test Acc: {6:>6.2%},  ({7:>6.2%})  Time: {8} {9}'
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, test_loss, test_acc,
                                 best_val_acc, time_dif, improve))
            total_batch += 1
        scheduler.step()

    return best_val_acc


if __name__ == '__main__':
    try:
        from Dataset.ng20.config import Config

        start_time = time.time()

        config = Config()

        model = Merge_Model(config).to(config.device)

        time_dif = get_time_dif(start_time)
        # print("Time usage:", time_dif)

        train(config, model)

    except KeyboardInterrupt as e:
        pass
