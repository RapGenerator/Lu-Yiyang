# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch:
    def __init__(self):
        self.encoder_inputs = []
        self.encoder_inputs_length = []
        self.decoder_targets = []
        self.decoder_targets_length = []


def load_and_cut_data(filepath):
    '''
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    '''
    #filepath指定了需要读取的文件的路径，'r'表示以只读方式打开，'encoding'指明了文件的编码方式
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = [] #创建一个列表
        lines = f.readlines() #读取文件，将文件的每一行作为列表lines中的一个元素存储
        for line in lines: #遍历每一行
            #jieba是一个中文分词工具
            #jieba.cut的第一个参数是需要分词的字符串，cut_all指定是否需要全模式分词
            #全模式分词指是否每种可能的词都取出来，比如："我来到北京清华大学"
            #全模式分词结果：我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
            #默认模式分词结果：我/ 来到/ 北京/ 清华大学
            #string.strip()的功能是删除字符串开头和结尾的[空格][tab][回车][换行]
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list] #将分词完毕的一行词语列表保存
            data.append(cutted_line) #将分词完毕的词语列表加到data的最后
    return data


def create_dic_and_map(sources, targets):
    '''
    得到输入和输出的字符映射表
    :param sources:
           targets:
    :return: sources_data:
             targets_data:
             word_to_id: 字典，数字到数字的转换
             id_to_word: 字典，数字到汉字的转换
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

    '''
    #这里是需要将只出现过一次的词语不添加进词库时的方法，现在使用的将所有词语都添加到词库，不需要使用这段代码
    #得到每次词语的使用频率
    word_dic = {} #创建一个dict
    for line in (sources + targets): #将sources和targets连接在一起，遍历其中的所有行
        for character in line: #遍历一行中的所有词语
            #dict.get()方法返回对应键的值，第一个参数是需要找的键，第二个参数是找不到该键时的默认返回值
            #将word_dic词典中的对应词语的出现次数+1
            word_dic[character] = word_dic.get(character, 0) + 1
    
    # 去掉使用频率为1的词
    word_dic_new = [] #创建一个list
    #遍历词典word_dic中的每一组键值对，dict.items()方法返回词典中可遍历的(键,值)元组数组
    for key, value in word_dic.items():
        #if value > 1: #对于只出现过一次的词语，忽略之
        if value > 0: #这里尝试将只出现过一次的词语也加入词库
            word_dic_new.append(key) #将出现了两次及以上的词语添加到list word_dic_new的末尾
    
    '''
    
    #对sources和targets中的每一行中的每一个词语，添加到word_dic_new中，使用set来去重
    word_dic_new = list(set([character for line in (sources + targets) for character in line]))

    # 将字典中的汉字/英文单词映射为数字
    #enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    #枚举每一个特殊词和出现两次及以上的词语，创建一个id到词语的映射，储存在dict中
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    #反向创建一个词语到id的映射，存储在dict中
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    #将sources和targets中的每一行中的每一个汉字/英文单词映射为id
    #如果词库中不含有这个词语，就将id设置为UNK的id
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in targets]

    return sources_data, targets_data, word_to_id, id_to_word


def createBatch(sources, targets):
    batch = Batch()
    batch.encoder_inputs_length = [len(source) for source in sources]
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length)
    max_target_length = max(batch.decoder_targets_length)

    for source in sources:
        # 将source进行反序并PAD
        source = list(reversed(source))
        pad = [padToken] * (max_source_length - len(source))
        batch.encoder_inputs.append(pad + source)

    for target in targets:
        # 将target进行PAD，并添加EOS符号
        pad = [padToken] * (max_target_length - len(target) - 1)
        eos = [eosToken] * 1
        batch.decoder_targets.append(target + eos + pad)

    return batch


def getBatches(sources_data, targets_data, batch_size):

    data_len = len(sources_data)

    def genNextSamples():
        for i in range(0, len(sources_data), batch_size):
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size, data_len)]

    batches = []
    for sources, targets in genNextSamples():
        batch = createBatch(sources, targets)
        batches.append(batch)

    return batches


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '':
        return None
    # 分词
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list]

    # 将每个单词转化为id
    wordIds = []
    for word in cutted_line:
        wordIds.append(word2id.get(word, unknownToken))
    print(wordIds)
    # 调用createBatch构造batch
    batch = createBatch([wordIds], [[]])
    return batch


if __name__ == '__main__':

    sources_txt = 'data/sources.txt'
    targets_txt = 'data/targets.txt'
    keep_rate = 0.6
    batch_size = 128

    # 得到分词后的sources和targets
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)
    batches = getBatches(sources_data, targets_data, batch_size)

    temp = 0
    for nexBatch in batches:
        if temp == 0:
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1