# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from data_helpers import *
from model import Seq2SeqModel
import math

if __name__ == '__main__':

    # 超参数
    rnn_size = 1024 #设置RNN中隐藏状态h的维度
    num_layers = 2 #设置RNN中的层数
    embedding_size = 1024 #设置embedding向量的维度
    batch_size = 128 #设置batch_size
    learning_rate = 0.0001 #设置学习率
    epochs = 5000 #设置所有数据经过多少次网络
    sources_txt = 'data/sources.txt' #设置sources文件的路径
    targets_txt = 'data/targets.txt' #设置targets文件的路径
    model_dir = 'model/' #设置网络模型参数的保存路径

    # 得到分词后的sources和targets，方法的具体操作请进入data_helpers.py中查看
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射为id，方法的具体操作请进入data_helpers.py中查看
    sources_data, targets_data, word_to_id, _ = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        #创建Seq2Seq模型，具体方法请进入model.py中查看
        model = Seq2SeqModel(rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode='train',
                             use_attention=True, beam_search=False, beam_size=5, cell_type='LSTM', max_gradient_norm=5.0)
        sess.run(tf.global_variables_initializer()) #初始化所有全局变量

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(sources_data, targets_data, batch_size)
            for nextBatch in batches:
                loss, summary = model.train(sess, nextBatch)
                perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
            model.saver.save(sess, model_dir + 'seq2seq.ckpt')