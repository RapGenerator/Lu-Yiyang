# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import tensorflow as tf
from tensorflow.contrib.framework import nest
from tensorflow.contrib.seq2seq import BahdanauAttention, AttentionWrapper
from tensorflow.contrib.seq2seq import sequence_loss
from tensorflow.contrib.seq2seq import TrainingHelper, GreedyEmbeddingHelper
from tensorflow.contrib.seq2seq import BasicDecoder, dynamic_decode
from tensorflow.contrib.seq2seq import BeamSearchDecoder
from tensorflow.contrib.seq2seq import tile_batch
from tensorflow.contrib.rnn import DropoutWrapper, MultiRNNCell
from tensorflow.contrib.rnn import LSTMCell, GRUCell


class Seq2SeqModel(object):
    #类Seq2SeqModel的构造函数，定义Seq2SeqModel类时，会自动调用__init__
    #参数中含有=的，指，若没有传入此参数，该参数的默认值
    def __init__(self, rnn_size, num_layers, embedding_size, learning_rate, word_to_idx, mode, use_attention,
                 beam_search, beam_size, cell_type='LSTM', max_gradient_norm=5.0):
        self.learing_rate = learning_rate #设置学习率
        self.embedding_size = embedding_size #设置embedding向量的维度
        self.rnn_size = rnn_size #设置RNN中隐向量h的维度
        self.num_layers = num_layers #设置RNN中的层数
        self.word_to_idx = word_to_idx #{词：id}词典
        self.vocab_size = len(self.word_to_idx) #词库大小（含有多少个不同的词语），len()是词典的size
        self.mode = mode #值为train或是decode，运行的是train还是predict
        self.use_attention = use_attention #是否使用attention机制
        self.beam_search = beam_search #是否使用beam search
        self.beam_size = beam_size #beam search中每层保留最大概率词语组合的条数
        self.cell_type = cell_type #RNN的类型，LSTM或是GRU
        self.max_gradient_norm = max_gradient_norm #梯度截断，若是梯度过大，将其截断到设置的最大梯度值

        #placeholder相当于定义了一个位置，这个位置中的数据在程序运行时再指定。
        #placeholder的第一个参数是数据的类型，第二个参数是数据的维度，第三个参数是数据的名字
        #数据的名字在绘制计算图的时候会使用到

        #定义编码器的输入数据的类型和维度，维度[None,None]即根据真实的输入的数据的维度确定
        self.encoder_inputs = tf.placeholder(tf.int32, [None, None], name='encoder_inputs')
        #定义编码器输入每一条数据的长度。
        self.encoder_inputs_length = tf.placeholder(tf.int32, [None], name='encoder_inputs_length')
        #定义解码器的输出数据的类型和维度，维度[None,None]即根据真实的ground truth的数据的维度确定
        self.decoder_targets = tf.placeholder(tf.int32, [None, None], name='decoder_targets')
        #定义解码器的输出数据的长度，后续用来求targets句子的最大长度
        self.decoder_targets_length = tf.placeholder(tf.int32, [None], name='decoder_targets_length')
        #设置batch size的类型和维度
        self.batch_size = tf.placeholder(tf.int32, [], name='batch_size')
        #dropout中保留多少百分比的神经元
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        #tensorflow.reduce_max()函数根据tensor的维度计算最大值
        #其中第一个参数为要求解的tensor，本身需要的第二个参数是维度，第三个参数是keep_dims，意为在一个维度上求最大值后是否保留此维度
        #因为在本例中数据只有1维，所以省略了第二个和第三个参数
        self.max_target_sequence_length = tf.reduce_max(self.decoder_targets_length, name='max_target_len')
        #tensorflowo.sequence_mask得到一个序列的mask
        #第一个参数是前多少个Boolean设置为True，第二个参数是最大长度，第三个参数是输出张量的类型
        #这里得到每一句target的前几个位置是词语，后几个位置是填充的，是词语的部分mask值为True，不是的为False
        self.mask = tf.sequence_mask(self.decoder_targets_length, self.max_target_sequence_length, dtype=tf.float32,
                                     name='masks')

        # embedding矩阵,encoder和decoder共用该词向量矩阵
        #tensorflow.get_variable(name,  shape, initializer)定义一个变量
        #第一个参数是变量的名字，第二个参数是变量的维度，第三个参数是初始化方式,实验得到的默认初始化是随机初始化
        self.embedding = tf.get_variable('embedding', [self.vocab_size, self.embedding_size])

        self.__graph__()

        #保存模型参数
        self.saver = tf.train.Saver()

    def __graph__(self):

        # encoder
        encoder_outputs, encoder_state = self.encoder()

        # decoder
        with tf.variable_scope('decoder'): #使用名字空间decoder
            encoder_inputs_length = self.encoder_inputs_length #编码器输入长度
            if self.beam_search: #是否使用beam search
                print("use beamsearch decoding..")
                #如果使用beam_search，则需要将encoder的输出进行tile_batch
                #tile_batch的功能是将第一个参数的数据复制multiplier份，在此例中是beam_size份。
                encoder_outputs = tile_batch(encoder_outputs, self.beam_size)
                #lambda是一个表达式，在此处相当于是一个关于s的函数
                #nest.map_structure(func,structure)将func应用于每一个structure并返回值
                #因为LSTM中有c和h两个structure，所以需要使用nest.map_structrue()
                encoder_state = nest.map_structure(lambda s: tile_batch(s, self.beam_size), encoder_state)
                encoder_inputs_length = tile_batch(encoder_inputs_length, self.beam_size)

            #定义要使用的attention机制。
            #使用的attention机制是Bahdanau Attention,关于这种attention机制的细节，可以查看论文
            #Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio.
            #"Neural Machine Translation by Jointly Learning to Align and Translate."
            #ICLR 2015. https://arxiv.org/abs/1409.0473
            #这种attention机制还有一种正则化的版本，如果需要在tensorflow中使用，加上参数normalize=True即可
            #关于正则化的细节，可以查看论文
            #Tim Salimans, Diederik P. Kingma.
            #"Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks."
            #https://arxiv.org/abs/1602.07868
            attention_mechanism = BahdanauAttention(num_units=self.rnn_size, #隐层的维度
                                                    memory=encoder_outputs, #通常情况下就是encoder的输出
                                                    #memory的mask，超过长度数据不计入attention
                                                    memory_sequence_length=encoder_inputs_length)
            # 定义decoder阶段要使用的RNNCell
            decoder_cell = self.create_rnn_cell()
            #AttentionWrapper()用于封装带attention机制的RNN网络
            decoder_cell = AttentionWrapper(cell=decoder_cell, #cell参数指明了需要封装的RNN网络
                                            #attention_mechanism指明了AttentionMechanism的实例
                                            attention_mechanism=attention_mechanism,
                                            #attention_layer_size TODO：是attention封装后的RNN状态维度？
                                            attention_layer_size=self.rnn_size,
                                            #name指明了AttentionWrapper的名字
                                            name='Attention_Wrapper')
            # 如果使用beam_seach则batch_size = self.batch_size * self.beam_size
            batch_size = self.batch_size if not self.beam_search else self.batch_size * self.beam_size

            #AttentionWrapper.zero_state()的功能是将AttentionWrapper对象0初始化
            #AttentionWrapper对象0初始化后可以使用.clone()方法将参数中的状态赋值给AttentionWrapper对象
            #本例中使用encoder阶段的最后一个隐层状态来赋值定义decoder阶段的初始化状态
            decoder_initial_state = decoder_cell.zero_state(batch_size=batch_size,
                                                            dtype=tf.float32).clone(cell_state=encoder_state)

            #tensorflow.layers.Dense()定义一个全连接层
            #第一个参数为input，指明该层的输入，kernel_initializer指明初始化方式
            #本例中使用的tf.truncated_normal_initializer()生成截断正态分布的随机数
            output_layer = tf.layers.Dense(self.vocab_size, kernel_initializer=tf.truncated_normal_initializer(
                                                            mean=0.0, #指定正态分布的平均值
                                                            stddev=0.1)) #指定正态分布的标准差

            if self.mode == 'train': #如果运行的是训练，则运行以下部分
                self.decoder_outputs = self.decoder_train(decoder_cell, decoder_initial_state, output_layer)
                # loss
                self.loss = sequence_loss(logits=self.decoder_outputs, targets=self.decoder_targets, weights=self.mask)

                # summary
                tf.summary.scalar('loss', self.loss)
                self.summary_op = tf.summary.merge_all()

                # optimizer
                optimizer = tf.train.AdamOptimizer(self.learing_rate)
                trainable_params = tf.trainable_variables()
                gradients = tf.gradients(self.loss, trainable_params)
                clip_gradients, _ = tf.clip_by_global_norm(gradients, self.max_gradient_norm)
                self.train_op = optimizer.apply_gradients(zip(clip_gradients, trainable_params))
            elif self.mode == 'decode':
                self.decoder_predict_decode = self.decoder_decode(decoder_cell, decoder_initial_state, output_layer)

    def encoder(self):
        '''
        创建模型的encoder部分
        :return: encoder_outputs: 用于attention，batch_size*encoder_inputs_length*rnn_size
                 encoder_state: 用于decoder的初始化状态，batch_size*rnn_size
        '''
        #使用名字空间encoder TODO：在这里这里有什么用？
        with tf.variable_scope('encoder'):
            encoder_cell = self.create_rnn_cell() #创建RNN网络结构
            #tensorflow.nn.embedding_lookup()方法在张量中寻找索引对应的元素
            #第一个参数是张量，第二个参数是索引
            #将encoder_inputs中的词语id转换为embedding向量
            encoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, self.encoder_inputs)
            #用rnn处理变长文本时，使用dynamic_rnn可以跳过padding部分的计算，减少计算量。
            #第一个参数是RNN的网络结构，第二个参数是输入x，sequence_length指定了句子的长度，dtype指定了数据类型
            #TODO：encoder_outputs,encoder_state分别是什么？
            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(encoder_cell, encoder_inputs_embedded, sequence_length=
                                                               self.encoder_inputs_length, dtype=tf.float32)
            return encoder_outputs, encoder_state

    def decoder_train(self, decoder_cell, decoder_initial_state, output_layer):
        '''
        创建train的decoder部分
        :param encoder_outputs: encoder的输出
        :param encoder_state: encoder的state
        :return: decoder_logits_train: decoder的predict
        '''
        ending = tf.strided_slice(self.decoder_targets, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], self.word_to_idx['<GO>']), ending], 1)
        decoder_inputs_embedded = tf.nn.embedding_lookup(self.embedding, decoder_input)

        training_helper = TrainingHelper(inputs=decoder_inputs_embedded,
                                         sequence_length=self.decoder_targets_length,
                                         time_major=False, name='training_helper')
        training_decoder = BasicDecoder(cell=decoder_cell,
                                        helper=training_helper,
                                        initial_state=decoder_initial_state,
                                        output_layer=output_layer)
        decoder_outputs, _, _ = dynamic_decode(decoder=training_decoder,
                                               impute_finished=True,
                                               maximum_iterations=self.max_target_sequence_length)
        decoder_logits_train = tf.identity(decoder_outputs.rnn_output)
        return decoder_logits_train

    def decoder_decode(self, decoder_cell, decoder_initial_state, output_layer):
        start_tokens = tf.ones([self.batch_size, ], tf.int32) * self.word_to_idx['<GO>']
        end_token = self.word_to_idx['<EOS>']

        if self.beam_search:
            inference_decoder = BeamSearchDecoder(cell=decoder_cell,
                                                  embedding=self.embedding,
                                                  start_tokens=start_tokens,
                                                  end_token=end_token,
                                                  initial_state=decoder_initial_state,
                                                  beam_width=self.beam_size,
                                                  output_layer=output_layer)
        else:
            decoding_helper = GreedyEmbeddingHelper(embedding=self.embedding,
                                                    start_tokens=start_tokens,
                                                    end_token=end_token)
            inference_decoder = BasicDecoder(cell=decoder_cell,
                                             helper=decoding_helper,
                                             initial_state=decoder_initial_state,
                                             output_layer=output_layer)

        decoder_outputs, _, _ = dynamic_decode(decoder=inference_decoder, maximum_iterations=50)
        if self.beam_search:
            decoder_predict_decode = decoder_outputs.predicted_ids
        else:
            decoder_predict_decode = tf.expand_dims(decoder_outputs.sample_id, -1)
        return decoder_predict_decode

    def create_rnn_cell(self):
        '''
        :return: RNN结构
        '''
        def single_rnn_cell():
            #根据参数cell_type的值，创建单个LSTM或者GRU的细胞
            single_cell = GRUCell(self.rnn_size) if self.cell_type == 'GRU' else LSTMCell(self.rnn_size)
            #DropoutWrapper()用来设置dropout
            #第一个参数是指定需要设置dropout的细胞，output_keep_prob设置保留输出，不进行dropout的概率
            basiccell = DropoutWrapper(single_cell, output_keep_prob=self.keep_prob)
            return basiccell
        #创建多层RNN，此例中共有num_layers层
        cell = MultiRNNCell([single_rnn_cell() for _ in range(self.num_layers)])
        return cell

    def train(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 0.5,
                     self.batch_size: len(batch.encoder_inputs)}
        _, loss, summary = sess.run([self.train_op, self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def eval(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.decoder_targets: batch.decoder_targets,
                     self.decoder_targets_length: batch.decoder_targets_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        loss, summary = sess.run([self.loss, self.summary_op], feed_dict=feed_dict)
        return loss, summary

    def infer(self, sess, batch):
        feed_dict = {self.encoder_inputs: batch.encoder_inputs,
                     self.encoder_inputs_length: batch.encoder_inputs_length,
                     self.keep_prob: 1.0,
                     self.batch_size: len(batch.encoder_inputs)}
        predict = sess.run([self.decoder_predict_decode], feed_dict=feed_dict)
        return predict