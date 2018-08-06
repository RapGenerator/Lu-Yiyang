# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.python.util import nest
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import ScheduledEmbeddingTrainingHelper, TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import LSTMCell, GRUCell


class Seq2SeqModel(object):
    # 类Seq2SeqModel的构造函数，定义Seq2SeqModel类时，会自动调用__init__
    def __init__(self, sess, rnn_size, num_layers, embedding_size, learning_rate, word_to_id, mode, use_attention,
                 beam_search, beam_size, cell_type, max_gradient_norm, teacher_forcing, teacher_forcing_probability):

        self.sess = sess
        self.learing_rate = learning_rate # 设置学习率
        self.embedding_size = embedding_size # 设置embedding向量的维度
        self.rnn_size = rnn_size # 设置RNN中隐向量h的维度
        self.num_layers = num_layers # 设置RNN中的层数
        self.word_to_id = word_to_id # {词：id}词典
        self.vocab_size = len(self.word_to_id) # 词库大小(含有多少个不同的词语)，len()是词典的size
        self.mode = mode # 值为train或是decode，运行的是train还是predict
        self.use_attention = use_attention # 是否使用attention机制
        self.beam_search = beam_search # 是否使用beam search
        self.beam_size = beam_size # beam search中每层保留最大概率词语组合的条数
        self.cell_type = cell_type # RNN的类型，LSTM或是GRU
        self.max_gradient_norm = max_gradient_norm # 梯度截断，若是梯度过大，将其截断到设置的最大梯度值
        self.teacher_forcing = teacher_forcing # 是否使用teacher_forcing
        self.teacher_forcing_probability = teacher_forcing_probability # 设置teacher_forcing中使用target或是output的概率

        self.build_graph() # 调用build_graph
        self.saver = tf.train.Saver() # 保存模型参数

    def build_graph(self):
        print('Building model...')

        # 调用buile_placeholder
        self.build_placeholder()
        # 调用build_encoder
        self.build_encoder()
        # 调用build_decoder
        self.build_decoder()

    def build_placeholder(self):
        print('Building placeholder...')

        # 定义编码器的输入数据的类型和维度，维度[None,None]即根据真实的输入的数据的维度确定
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        # 定义编码器输入每一条数据的长度。
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        # 定义解码器的输出数据的类型和维度，维度[None,None]即根据真实的ground truth的数据的维度确定
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        # 定义解码器的输出数据的长度，后续用来求targets句子的最大长度
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        # 设置batch size的类型和维度
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        # dropout中保留多少百分比的神经元
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        # tensorflow.reduce_max()函数根据tensor的维度计算最大值
        # 其中第一个参数为要求解的tensor，本身需要的第二个参数是维度，第三个参数是keep_dims，意为在一个维度上求最大值后是否保留此维度
        # 因为在本例中数据只有1维，所以省略了第二个和第三个参数
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        # tensorflowo.sequence_mask得到一个序列的mask
        # 第一个参数是前多少个Boolean设置为True，第二个参数是最大长度，第三个参数是输出张量的类型
        # 这里得到每一句target的前几个位置是词语，后几个位置是填充的，是词语的部分mask值为True，不是的为False
        self.mask = tf.sequence_mask(
            self.decoder_targets_length,
            self.max_target_sequence_length,
            dtype=tf.float32,
            name='masks'
        )

        # embedding矩阵,encoder和decoder共用该词向量矩阵
        # tensorflow.get_variable(name,  shape, initializer)定义一个变量
        # 第一个参数是变量的名字，第二个参数是变量的维度，第三个参数是初始化方式,实验得到的默认初始化是随机初始化
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

    def build_encoder(self):
        print('Building encoder...')

        # 使用名字空间encoder
        with tf.variable_scope('encoder'):
            encoder_cell = self.create_rnn_cell() # 创建RNN网络结构
            # tensorflow.nn.embedding_lookup()方法在张量中寻找索引对应的元素
            # 第一个参数是张量，第二个参数是索引
            # 将encoder_inputs中的词语id转换为embedding向量
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            # 用rnn处理变长文本时，使用dynamic_rnn可以跳过padding部分的计算，减少计算量。
            # 第一个参数是RNN的网络结构，第二个参数是输入x，sequence_length指定了句子的长度，dtype指定了数据类型
            self.encoder_outputs, self.encoder_state = tf.nn.dynamic_rnn(
                encoder_cell,
                encoder_inputs_embedded,
                sequence_length=self.encoder_inputs_length,
                dtype=tf.float32
            )

    def build_decoder(self):
        print('Building decoder...')

        # 使用名字空间decoder
        with tf.variable_scope('decoder'):
            # 创建decoder网络，具体方法请进入build_decoder_cell()中查看
            self.decoder_cell, self.decoder_initial_state = self.build_decoder_cell()
            # tensorflow.layers.Dense()定义一个全连接层
            # 第一个参数为input，指明该层的输入，kernel_initializer指明初始化方式
            # 本例中使用的tf.truncated_normal_initializer()生成截断正态分布的随机数
            self.output_layer = tf.layers.Dense(
                self.vocab_size,
                kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
            )

            if self.mode == 'train': # 如果是训练
                self.build_train_decoder() # 进入build_train_decoder()
            elif self.mode == 'predict': # 如果是预测
                self.build_predict_decoder() # 进入build_predict_decoder()
            else:
                raise RuntimeError

    def build_train_decoder(self):
        print('Building train decoder...')

        # tf.strided_slice(data,begin,end,stride)是一个跨步切片操作，切片区间左闭右开
        # 本例中data为decoder_targets，对真实的下一句子进行切片，end中的-1会得到那一维度的最后一个
        # 得到的ending为一个batch中的一行行target句子
        ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
        # tf.fill(dim,value)的功能是创建一个dim维度，值为value的tensor对象
        # tf.concat(values,axis)的功能是将values在axis维上进行拼接
        # 在本例中，是将每一个target句子的前面加上<GO>
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_id['<GO>']), ending], 1)
        # tensorflow.nn.embedding_lookup()方法在张量中寻找索引对应的元素
        # 第一个参数是张量，第二个参数是索引
        # 将decoder_inputs中的词语id转换为embedding向量
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

        # 定义一个Helper，是Decoder的一部分，决定Decoder的输入是什么。
        # 官网给出了下面几种Helper类：
        # "Helper"：最基本的抽象类
        # "TrainingHelper"：训练过程中最常使用的Helper，下一时刻输入就是上一时刻target的真实值
        # "GreedyEmbeddingHelper"：预测阶段最常使用的Helper，下一时刻输入是上一时刻概率最大的单词通过embedding之后的向量
        # "SampleEmbeddingHelper"：预测时helper，继承自GreedyEmbeddingHelper，下一时刻输入是上一时刻通过某种概率分布采样而来在经过embedding之后的向量
        # "CustomHelper"：最简单的helper，一般用户自定义helper时会基于此，需要用户自己定义如何根据输出得到下一时刻输入
        # "ScheduledEmbeddingTrainingHelper"：训练时Helper，继承自TrainingHelper，添加了广义伯努利分布，对id的embedding向量进行sampling
        # "ScheduledOutputTrainingHelper"：训练时Helper，继承自TrainingHelper，直接对输出进行采样
        # "InferenceHelper"：CustomHelper的特例，只用于预测的helper，也需要用户自定义如何得到下一时刻输入

        if self.teacher_forcing: # 如果使用teacher_forcing
            training_helper = ScheduledEmbeddingTrainingHelper( # 定义一个ScheduledEmbeddingTrainingHelper
                inputs=decoder_inputs_embedded, # decoder的输入
                sequence_length=self.decoder_targets_length, # 输入的长度
                embedding=self.embedding, # embedding矩阵
                sampling_probability=self.teacher_forcing_probability, # teacher_forcing中使用target或是output的概率
                # time_major表示是否时间序列为第一维，如果是True，则输入需要是T×B×E，否则，为B×T×E
                # 其中T代表时间序列的长度，B代表batch size。 E代表词向量的维度。
                time_major=False,
                name='teacher_forcing_training_helper'
            )
        else: # 如果不使用teacher_forcing
            training_helper = TrainingHelper( # 定义一个TrainingHelper
                inputs=decoder_inputs_embedded, # decoder的输入
                sequence_length=self.decoder_targets_length, # 输入的长度
                # time_major表示是否时间序列为第一维，如果是True，则输入需要是T×B×E，否则，为B×T×E
                # 其中T代表时间序列的长度，B代表batch size。 E代表词向量的维度。
                time_major=False,
                name='training_helper'
            )

        training_decoder = BasicDecoder( # 基础的取样解码器
            cell=self.decoder_cell, # 使用的RNN网络
            helper=training_helper, # 使用的helper
            initial_state=self.decoder_initial_state, # 使用的h0
            output_layer=self.output_layer # 使用的输出层
        )

        decoder_outputs, _, _ = dynamic_decode( # 动态解码器
            decoder=training_decoder, # decoder实例
            # impute_finished为真时会拷贝最后一个时刻的状态并将输出置零，程序运行更稳定，使最终状态和输出具有正确的值，
            # 在反向传播时忽略最后一个完成步。但是会降低程序运行速度 TODO:不懂实际用处
            impute_finished=True,
            maximum_iterations=self.max_target_sequence_length # 最大解码步数，这里就设置为最大的target长度
        )

        # 那这里就卖个萌吧，我也不知道为什么要用tf.identity TODO：为什么需要tf.identity()？
        self.decoder_logits_train = tf.identity(decoder_outputs.rnn_output)

        # 定义损失函数
        self.loss = sequence_loss( # 将损失函数定义为sequence_loss
            logits=self.decoder_logits_train, # 输出logits
            targets=self.decoder_targets, # 真实targets
            weights=self.mask # 即mask，滤去padding的loss计算，使loss计算更准确
        )

        # summary，用于可视化
        tf.summary.scalar('loss', self.loss)
        self.summary_op = tf.summary.merge_all()

        # 进入build_optimizer()
        self.build_optimizer()

    def build_predict_decoder(self):
        print('Building predict decoder...')

        start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_id['<GO>']
        end_token = self.word_to_id['<EOS>']

        if self.beam_search:
            inference_decoder = BeamSearchDecoder(
                cell=self.decoder_cell,
                embedding=self.embedding,
                start_tokens=start_tokens,
                end_token=end_token,
                initial_state=self.decoder_initial_state,
                beam_width=self.beam_size,
                output_layer=self.output_layer
            )

        else:
            decoding_helper = GreedyEmbeddingHelper(
                embedding=self.embedding,
                start_tokens=start_tokens,
                end_token=end_token
            )
            inference_decoder = BasicDecoder(
                cell=self.decoder_cell,
                helper=decoding_helper,
                initial_state=self.decoder_initial_state,
                output_layer=self.output_layer
            )

        decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=50)

        if self.beam_search:
            self.decoder_predict_decode = decoder_outputs.predicted_ids
        else:
            self.decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)

    def build_decoder_cell(self):
        encoder_inputs_length = self.encoder_inputs_length # 编码器输入长度
        if self.beam_search: # 是否使用beam search
            print("use beamsearch decoding..")
            # 如果使用beam_search，则需要将encoder的输出进行tile_batch
            # tile_batch的功能是将第一个参数的数据复制multiplier份，在此例中是beam_size份
            self.encoder_outputs = tile_batch(self.encoder_outputs, multiplier=self.beam_size)
            # lambda是一个表达式，在此处相当于是一个关于s的函数
            # nest.map_structure(func,structure)将func应用于每一个structure并返回值
            # 因为LSTM中有c和h两个structure，所以需要使用nest.map_structrue()
            self.encoder_state = nest.map_structure(lambda s: tile_batch(s, self.beam_size), self.encoder_state)
            encoder_inputs_length = tile_batch(encoder_inputs_length, multiplier=self.beam_size)

        # 定义要使用的attention机制。
        # 使用的attention机制是Bahdanau Attention,关于这种attention机制的细节，可以查看论文
        # Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
        # "Neural Machine Translation by Jointly Learning to Align and Translate."
        # ICLR 2015. https://arxiv.org/abs/1409.0473
        # 这种attention机制还有一种正则化的版本，如果需要在tensorflow中使用，加上参数normalize=True即可
        # 关于正则化的细节，可以查看论文
        # Tim Salimans, Diederik P. Kingma.
        # "Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks."
        # https://arxiv.org/abs/1602.07868
        attention_mechanism = BahdanauAttention(
            num_units=self.rnn_size, # 隐层的维度
            memory=self.encoder_outputs, # 通常情况下就是encoder的输出
            # memory的mask，超过长度数据不计入attention
            memory_sequence_length=encoder_inputs_length
        )

        # 定义decoder阶段要使用的RNNCell，然后为其封装attention wrapper
        decoder_cell = self.create_rnn_cell() # 定义decoder阶段要使用的RNNCell
        decoder_cell = AttentionWrapper( # AttentionWrapper()用于封装带attention机制的RNN网络
            cell=decoder_cell, # cell参数指明了需要封装的RNN网络
            attention_mechanism=attention_mechanism, # attention_mechanism指明了AttentionMechanism的实例
            attention_layer_size=self.rnn_size, # attention_layer_size TODO：是attention封装后的RNN状态维度？
            name='Attention_Wrapper' # name指明了AttentionWrapper的名字
        )

        # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size
        batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

        # AttentionWrapper.zero_state()的功能是将AttentionWrapper对象0初始化
        # AttentionWrapper对象0初始化后可以使用.clone()方法将参数中的状态赋值给AttentionWrapper对象
        # 本例中使用encoder阶段的最后一个隐层状态来赋值定义decoder阶段的初始化状态
        decoder_initial_state = decoder_cell.zero_state(
            batch_size=batch_size,
            dtype=tf.float32).clone(
            cell_state=self.encoder_state
        )

        return decoder_cell, decoder_initial_state

    def create_rnn_cell(self):
        '''
        return: RNN结构
        '''
        def single_rnn_cell():
            # 根据参数cell_type的值，创建单个LSTM或者GRU的细胞
            single_cell = GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
            # DropoutWrapper()用来设置dropout
            # 第一个参数是指定需要设置dropout的细胞，output_keep_prob设置保留输出，不进行dropout的概率
            basic_cell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basic_cell
        # 创建多层RNN，此例中共有num_layers层
        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def build_optimizer(self):
        print('Building optimizer...')

        # 定义优化器
        # tf.train.AdamOptimizer()定义一个Adam优化器，其中的第一个参数是初始的学习率
        optimizer = tf.train.AdamOptimizer(self.learing_rate)
        # tf.trainable_variables()返回所有当前计算图中在获取变量时未标记trainable=False的变量集合
        trainable_params = tf.trainable_variables()
        # tf.gradients(ys,xs)实现ys对xs求导，本例中即为self.loss对trainable_params求导
        gradients = tf.gradients(self.loss, trainable_params)
        # tf.clip_by_global_norm()进行梯度截断，本例中对超过self.max_gradient_norm的gradients都进行截断
        clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
        # zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个元组，然后返回由这些元组组成的列表
        # optimizer.apply_gradients(grads_and_vars)将梯度应用到变量
        # grads_and_vars需要是(梯度，变量)对的列表
        self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))

    def train(self, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 0.5,
                     self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary = self.sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss, summary = self.sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = self.sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict