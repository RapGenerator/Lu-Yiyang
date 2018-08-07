# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from data_helpers import load_and_cut_data, create_dic_and_map, getBatches
from model import Seq2SeqModel
import math

# 如果这个.py文件是被直接执行的，那么__name__会被赋值为__main__
# 如果这个.py文件是被import的，那么__name__会被赋值为import它的.py文件名
# 也就是说，当这个train.py文件被直接运行的时候会进入这个if，被import时不进入
if __name__ == '__main__':

    # 超参数
    rnn_size = 256 # 设置RNN中隐藏状态h的维度
    num_layers = 2 # 设置RNN中的层数
    embedding_size = 256 # 设置embedding向量的维度
    learning_rate = 0.001 # 设置学习率
    mode = 'train' # 设置训练或是预测
    use_attention = True # 设置是否使用attention
    beam_search = False # 设置是否使用beam_search
    beam_size = 3 # 设置beam_search中的beam_size
    cell_type = 'LSTM' # 设置RNNcell的类型
    max_gradient_norm = 5.0 # 设置梯度截断的最大梯度值，若是梯度过大，将其截断到设置的最大梯度值
    teacher_forcing = False # 设置是否使用tercher_forcing
    teacher_forcing_probability = 0.5 # 设置teacher_forcing中使用target或是output的概率

    batch_size = 32  # 设置batch_size
    epochs = 40 # 设置所有数据经过多少次网络
    display = 100 # 设置训练多少个batch后打印一遍loss
    pretrained = False # 设置是否使用之前训练过的模型参数
    sources_txt = 'data/sources.txt' # 设置sources文件的路径
    targets_txt = 'data/targets.txt' # 设置targets文件的路径
    model_dir = 'model/' # 设置网络模型参数的保存路径

    # 得到分词后的sources和targets，方法的具体操作请进入data_helpers.py中查看
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射为id，方法的具体操作请进入data_helpers.py中查看
    sources_data, targets_data, word_to_id, _ = create_dic_and_map(sources, targets)

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

        if pretrained: # 如果需要加载之前训练的模型
            # tf.train.get_checkpoint_state函数通过checkpoint文件找到模型文件名
            ckpt = tf.train.get_checkpoint_state(model_dir)
            # 如果模型存在
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                # 使用saver.restore()方法恢复变量
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else: # 找不到模型
                raise ValueError('No such file:[{}]'.format(model_dir)) # 报错
        else: # 从零开始训练模型
            sess.run(tf.global_variables_initializer())

        for e in range(epochs): # 对于每一个epoch
            # 打印训练Epoch信息
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            # 将一个epoch中的数据制作成一个个batch，具体方法进入data_helpers.py查看
            batches = getBatches(sources_data, targets_data, batch_size)
            step = 0 # 记录训练了多少个batch
            for nextBatch in batches: # 遍历batches中的每个batch
                # 将这个batch的数据喂给网络进行训练
                loss, summary = model.train(nextBatch)
                if step % display == 0: # 每隔display个batch打印一次训练信息
                    # math.exp(x)返回e的x次方
                    # inf表示正无穷
                    # 计算困惑度
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    # 打印loss和困惑度
                    print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                step += 1 # 计数，记录训练了多少个batch
            # 保存当前网络模型参数
            model.saver.save(sess, model_dir + 'seq2seq.ckpt')