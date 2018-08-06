# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from data_helpers import load_and_cut_data, create_dic_and_map, getBatches
from model import Seq2SeqModel
import math


if __name__ == '__main__':

    # 超参数
    rnn_size = 256 #设置RNN中隐藏状态h的维度
    num_layers = 2 #设置RNN中的层数
    embedding_size = 256 #设置embedding向量的维度
    learning_rate = 0.001 #设置学习率
    mode = 'train' #设置训练或是预测
    use_attention = True #设置是否使用attention
    beam_search = False #设置是否使用beam_search
    beam_size = 3 #设置beam_search中的beam_size
    cell_type = 'LSTM' #设置RNNcell的类型
    max_gradient_norm = 5.0 #设置梯度截断的最大梯度值，若是梯度过大，将其截断到设置的最大梯度值
    teacher_forcing = True #设置是否使用tercher_forcing
    teacher_forcing_probability = 0.5 #设置teacher_forcing中使用target或是output的概率

    batch_size = 32  #设置batch_size
    epochs = 40 #设置所有数据经过多少次网络
    display = 100 #设置训练多少个batch后打印一遍loss
    pretrained = True #设置是否使用之前训练过的模型参数
    sources_txt = 'data/sources.txt' #设置sources文件的路径
    targets_txt = 'data/targets.txt' #设置targets文件的路径
    model_dir = 'model/' #设置网络模型参数的保存路径

    #得到分词后的sources和targets，方法的具体操作请进入data_helpers.py中查看
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    #根据sources和targets创建词典，并映射为id，方法的具体操作请进入data_helpers.py中查看
    sources_data, targets_data, word_to_id, _ = create_dic_and_map(sources, targets)

    with tf.Session() as sess:
        #创建Seq2Seq模型，具体方法请进入model.py中查看
        model = Seq2SeqModel(
            sess=sess,
            rnn_size=rnn_size,
            num_layers=num_layers,
            embedding_size=embedding_size,
            learning_rate=learning_rate,
            word_to_id=word_to_id,
            mode=mode,
            use_attention=use_attention,
            beam_search=beam_search,
            beam_size=beam_size,
            cell_type=cell_type,
            max_gradient_norm=max_gradient_norm,
            teacher_forcing=teacher_forcing,
            teacher_forcing_probability=teacher_forcing_probability
        )

        if pretrained:
            ckpt = tf.train.get_checkpoint_state(model_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                print('Reloading model parameters..')
                model.saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise ValueError('No such file:[{}]'.format(model_dir))
        else:
            sess.run(tf.global_variables_initializer())

        for e in range(epochs):
            print("----- Epoch {}/{} -----".format(e + 1, epochs))
            batches = getBatches(sources_data, targets_data, batch_size)
            step = 0
            for nextBatch in batches:
                loss, summary = model.train(nextBatch)
                if step % display == 0:
                    perplexity = math.exp(float(loss)) if loss < 300 else float('inf')
                    print("----- Loss %.2f -- Perplexity %.2f" % (loss, perplexity))
                step += 1
            model.saver.save(sess, model_dir + 'seq2seq.ckpt')