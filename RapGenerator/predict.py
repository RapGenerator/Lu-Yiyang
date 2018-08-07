# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
import numpy as np
from data_helpers import load_and_cut_data, create_dic_and_map, sentence2enco
from model import Seq2SeqModel
import sys


def predict_ids_to_seq(predict_ids, id2word, beam_szie):
    '''
    将beam_search返回的结果转化为字符串
    :param predict_ids: 列表，长度为batch_size，每个元素都是decode_len*beam_size的数组
    :param id2word: vocab字典
    :no return
    '''
    for single_predict in predict_ids: # 遍历batch中的每一句话的预测，本例中的输入只有一句话
        for i in range(beam_size): # 遍历beam_size句预测的句子
            # np.tolist()将对象转换成list
            # [:, i]是切片操作，将beam search中的单独一句预测切出来
            predict_list = np.ndarray.tolist(single_predict[:, i])
            # 将id形式的单句转换成word形式
            predict_seq = [id2word[idx] for idx in predict_list]
            # string.join()方法的功能是在第一个参数字符串序列中的每两个元素之间插入string
            # 本例中就是在每两个词之间插入空格
            print(" ".join(predict_seq))


# 如果这个.py文件是被直接执行的，那么__name__会被赋值为__main__
# 如果这个.py文件是被import的，那么__name__会被赋值为import它的.py文件名
# 也就是说，当这个predict.py文件被直接运行的时候会进入这个if，被import时不进入
if __name__ == '__main__':

    # 超参数
    rnn_size = 256 # 设置RNN中隐藏状态h的维度
    num_layers = 4 # 设置RNN中的层数
    embedding_size = 256 # 设置embedding向量的维度
    learning_rate = 0.001 # 设置学习率
    mode = 'predict' # 设置训练或是预测
    use_attention = True # 设置是否使用attention
    beam_search = True # 设置是否使用beam_search
    beam_size = 3 # 设置beam_search中的beam_size
    cell_type = 'LSTM' # 设置RNNcell的类型
    max_gradient_norm = 5.0 # 设置梯度截断的最大梯度值，若是梯度过大，将其截断到设置的最大梯度值
    teacher_forcing = True # 设置是否使用tercher_forcing
    teacher_forcing_probability = 0.5 # 设置teacher_forcing中使用target或是output的概率

    batch_size = 128 # 设置batch_size
    sources_txt = 'data/sources.txt' # 设置sources文件的路径
    targets_txt = 'data/targets.txt' # 设置targets文件的路径
    model_dir = 'model/' # 设置网络模型参数的保存路径

    # 得到分词后的sources和targets，方法的具体操作请进入data_helpers.py中查看
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射为id，方法的具体操作请进入data_helpers.py中查看
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        # 创建Seq2Seq模型，具体方法请进入model.py中查看
        model = Seq2SeqModel(
            sess=sess,
            rnn_size=rnn_size, # 设置RNN中隐向量h的维度
            num_layers=num_layers, # 设置RNN中的层数
            embedding_size=embedding_size, # 设置embedding向量的维度
            learning_rate=learning_rate, # 设置学习率
            word_to_id=word_to_id, # {词：id}词典
            mode=mode, # 值为train或是decode，运行的是train还是predict
            use_attention=use_attention, # 是否使用attention机制
            beam_search=beam_search, # 是否使用beam search
            beam_size=beam_size, # beam search中每层保留最大概率词语组合的条数
            cell_type=cell_type, # RNN的类型，LSTM或是GRU
            max_gradient_norm=max_gradient_norm, # 梯度截断，若是梯度过大，将其截断到设置的最大梯度值
            teacher_forcing=teacher_forcing, # 是否使用teacher_forcing
            teacher_forcing_probability=teacher_forcing_probability # 设置teacher_forcing中使用target或是output的概率
        )

        # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名
        ckpt = tf.train.get_checkpoint_state(model_dir)
        # 如果模型存在
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            # 使用saver.restore()方法恢复变量
            model.saver.restore(sess, ckpt.model_checkpoint_path)
        else: # 如果模型不存在
            raise ValueError('No such file:[{}]'.format(model_dir)) # 报错

        # 打印一'>'，提示用户输入句子
        # sys.stdout.write()的功能大概是不带回车'\n'的print()
        sys.stdout.write("> ")
        # sys.stdout带有缓冲区，使用sys.stdout.flush()使其立即输出
        sys.stdout.flush()
        # sys.stdin.readline()用于读取一行输入
        sentence = sys.stdin.readline()
        while sentence: # 只要还在输入，就持续运行
            # 将用户输入的句子切词、转换成id并放入一个batch中，具体细节进入data_helpers.py查看
            batch = sentence2enco(sentence, word_to_id)
            # 通过这一句，预测下一句
            predicted_ids = model.infer(batch)
            # 将beam_search返回的结果转化为字符串
            predict_ids_to_seq(predicted_ids, id_to_word, beam_size)
            # 再次输出'>'，提示用户输入下一句子
            print("> ")
            # 使用sys.stdout.flush()使其立即输出
            sys.stdout.flush()
            # sys.stdin.readline()再次读取一行输入
            sentence = sys.stdin.readline()