# ！/usr/bin/env python
#  _*_ coding:utf-8 _*_

import jieba

# 设置 填充pad的Token为0，未知unknown的Token为1，每句话的开头go的Token为2，每行的结束eos的Token为3
padToken, unknownToken, goToken, eosToken = 0, 1, 2, 3


class Batch: # Batch类
    def __init__(self): # 构造函数
        self.encoder_inputs = [] # 一个batch中的编码器输入，即sources列表
        self.encoder_inputs_length = [] # 一个batch中的sources中的每句的长度
        self.decoder_targets = [] # 一个batch中的targets列表
        self.decoder_targets_length = [] # 一个batch中的targets中的每句的长度


def load_and_cut_data(filepath):
    '''
    加载数据并分词
    :param filepath: 路径
    :return: data: 分词后的数据
    '''
    # filepath指定了需要读取的文件的路径，'r'表示以只读方式打开，'encoding'指明了文件的编码方式
    with open(filepath, 'r', encoding='UTF-8') as f:
        data = [] # 创建一个列表
        lines = f.readlines() # 读取文件，将文件的每一行作为列表lines中的一个元素存储
        for line in lines: #遍历每一行
            # jieba是一个中文分词工具
            # jieba.cut的第一个参数是需要分词的字符串，cut_all指定是否需要全模式分词
            # 全模式分词指是否每种可能的词都取出来，比如："我来到北京清华大学"
            # 全模式分词结果：我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
            # 默认模式分词结果：我/ 来到/ 北京/ 清华大学
            # string.strip()的功能是删除字符串开头和结尾的[空格][tab][回车][换行]
            seg_list = jieba.cut(line.strip(), cut_all=False)
            cutted_line = [e for e in seg_list] # 将分词完毕的一行词语保存至列表
            data.append(cutted_line) # 将分词完毕的词语列表加到data的最后
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
    # 这里是需要将只出现过一次的词语不添加进词库时的方法，现在使用的将所有词语都添加到词库，不需要使用这段代码
    # 得到每次词语的使用频率
    word_dic = {} #创建一个dict
    for line in (sources + targets): # 将sources和targets连接在一起，遍历其中的所有行
        for character in line: # 遍历一行中的所有词语
            # dict.get()方法返回对应键的值，第一个参数是需要找的键，第二个参数是找不到该键时的默认返回值
            # 将word_dic词典中的对应词语的出现次数+1
            word_dic[character] = word_dic.get(character, 0) + 1
    
    # 去掉使用频率为1的词
    word_dic_new = [] #创建一个list
    # 遍历词典word_dic中的每一组键值对，dict.items()方法返回词典中可遍历的(键,值)元组数组
    for key, value in word_dic.items():
        #if value > 1: # 对于只出现过一次的词语，忽略之
        if value > 0: # 这里尝试将只出现过一次的词语也加入词库
            word_dic_new.append(key) # 将出现了两次及以上的词语添加到list word_dic_new的末尾
    
    '''
    
    # 对sources和targets中的每一行中的每一个词语，添加到word_dic_new中，使用set来去重
    word_dic_new = list(set([character for line in (sources + targets) for character in line]))

    # 将字典中的汉字/英文单词映射为数字
    # enumerate()函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
    # 枚举每一个特殊词和出现两次及以上的词语，创建一个id到词语的映射，储存在dict中
    id_to_word = {idx: word for idx, word in enumerate(special_words + word_dic_new)}
    # 反向创建一个词语到id的映射，存储在dict中
    word_to_id = {word: idx for idx, word in id_to_word.items()}

    # 将sources和targets中的每一行中的每一个汉字/英文单词映射为id
    # dict.get()方法返回对应键的值，第一个参数是需要找的键，第二个参数是找不到该键时的默认返回值
    # 如果词库中不含有这个词语，就将id设置为UNK的id
    sources_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in sources]
    targets_data = [[word_to_id.get(character, word_to_id['<UNK>']) for character in line] for line in targets]

    return sources_data, targets_data, word_to_id, id_to_word


def createBatch(sources, targets):
    batch = Batch() # 创建一个batch类
    # 计算得到这个batch中的sources中的每一行source句子的长度
    batch.encoder_inputs_length = [len(source) for source in sources]
    # 计算得到这个batch中的targets中的每一行target句子的长度，并+1
    # 这里的+1是因为需要在targets的每一行最后加上<EOS>
    batch.decoder_targets_length = [len(target) + 1 for target in targets]

    max_source_length = max(batch.encoder_inputs_length) # 得到这个batch中最长source的长度
    max_target_length = max(batch.decoder_targets_length) # 得到这个batch中最长target的长度

    for source in sources: # 对于这个batch中的sources中的每一行source
        # 将source进行反序并PAD
        # reverse()函数用于反向列表中元素
        source = list(reversed(source))
        # 计算出这句source需要填充padToken的个数
        pad = [padToken] * (max_source_length - len(source))
        # 将pad和source连接后加入到这个batch中的encoder_inputs中(pad在前的原因是source已经反向了)
        batch.encoder_inputs.append(pad + source)

    for target in targets: # 对于这个batch中的targets中的每一行target
        # 将target进行PAD，并添加EOS符号
        # 计算出这句target需要填充padToken的个数
        pad = [padToken] * (max_target_length - len(target) - 1)
        eos = [eosToken] * 1 # 每句添加一个<EOS>
        # 将target、eos、pad连接后加入到这个batch中的decoder_targets中
        batch.decoder_targets.append(target + eos + pad)

    return batch


def getBatches(sources_data, targets_data, batch_size):
    # len()方法返回对象（字符、列表、元组等）长度或项目个数
    # 得到sources_data的长度
    data_len = len(sources_data)

    def genNextSamples(): # 用于生成一个
        # range()函数可创建一个整数列表，一般用在for循环中,三个参数分别是开始，结束，步长
        for i in range(0, data_len, batch_size):
            # min()方法返回给定参数的最小值
            # yield对生成器对象返回一个返回值
            # 使用yield的原因是：若不使用，会将所有数据读到内存中，而使用，可以每次只读一小部分至内存中
            # 若不是特别理解，推荐阅读：http://pyzh.readthedocs.io/en/latest/the-python-yield-keyword-explained.html
            # 本例中返回sources和targets中的一个batch的数据
            yield sources_data[i:min(i + batch_size, data_len)], targets_data[i:min(i + batch_size, data_len)]

    batches = [] # 定义一个list
    for sources, targets in genNextSamples(): # 取一个batch的sources和targets数据
        batch = createBatch(sources, targets) # 生成一个batch的数据，具体方法进入createBatch()查看
        batches.append(batch) # 将生成的一个batch加入到batches列表中

    return batches


def sentence2enco(sentence, word2id):
    '''
    测试的时候将用户输入的句子转化为可以直接feed进模型的数据，现将句子转化成id，然后调用createBatch处理
    :param sentence: 用户输入的句子
    :param word2id: 单词与id之间的对应关系字典
    :return: 处理之后的数据，可直接feed进模型进行预测
    '''
    if sentence == '': # 如果输入的句子是空的
        return None # 直接返回None
    
    # jieba是一个中文分词工具
    # jieba.cut的第一个参数是需要分词的字符串，cut_all指定是否需要全模式分词
    # 全模式分词指是否每种可能的词都取出来，比如："我来到北京清华大学"
    # 全模式分词结果：我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
    # 默认模式分词结果：我/ 来到/ 北京/ 清华大学
    # string.strip()的功能是删除字符串开头和结尾的[空格][tab][回车][换行]
    seg_list = jieba.cut(sentence.strip(), cut_all=False)
    cutted_line = [e for e in seg_list] # 将分词完毕的一行词语保存至列表

    # 将每个单词转化为id
    wordIds = [] # 创建一个列表
    for word in cutted_line: # 遍历cutted_line列表
        # dict.get()方法返回对应键的值，第一个参数是需要找的键，第二个参数是找不到该键时的默认返回值
        # 将word转换成id，并加入到wordIds中
        wordIds.append(word2id.get(word, unknownToken))
    print(wordIds) # 打印wordIds
    # 调用createBatch构造batch
    # 创建一个batch，在预测时，只有source，没有target
    batch = createBatch([wordIds], [[]])
    return batch


# 如果这个.py文件是被直接执行的，那么__name__会被赋值为__main__
# 如果这个.py文件是被import的，那么__name__会被赋值为import它的.py文件名
# 也就是说，当这个data_helpers.py文件被直接运行的时候会进入这个if，被import时不进入
if __name__ == '__main__':

    sources_txt = 'data/sources.txt' # 设定sources文件路径
    targets_txt = 'data/targets.txt' # 设定targets文件路径
    batch_size = 128 # 设定batch_size

    # 得到分词后的sources和targets，具体方法进入load_and_cut_data查看
    sources = load_and_cut_data(sources_txt)
    targets = load_and_cut_data(targets_txt)

    # 根据sources和targets创建词典，并映射为id，具体方法进入create_dic_and_map查看
    sources_data, targets_data, word_to_id, id_to_word = create_dic_and_map(sources, targets)
    # getBatches()将数据制作成一个个batch，具体方法进入getBatches()查看
    batches = getBatches(sources_data, targets_data, batch_size)

    # 下面的代码应该是用于测试获取的batch是否正确的
    temp = 0 # 立一个flag
    for nexBatch in batches: # 遍历batches中的batch
        if temp == 0: # 对于第一个batch，打印它的属性
            print(len(nexBatch.encoder_inputs))
            print(len(nexBatch.encoder_inputs_length))
            print(nexBatch.decoder_targets)
            print(nexBatch.decoder_targets_length)
        temp += 1