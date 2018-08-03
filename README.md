所有运行环境均为：

    Python3.6
    Tensorflow1.9

目前包含：

    一个Char RNN demo
    一个seq2seq demo（已经准备删除）

每日进展：

    2018年08月03日：
        调试seq2seq+attention的代码（来自伊玮雯），失败。
        录制第一周视频。
        为seq2seq+attention代码注解，完成了一部分，已上传。
    
    2018年08月02日：
        学习seq2seq+attention的代码（来自庞云升）
        与王德瑞导师会面，明确当前问题与未来计划。
        就LSTM相关问题进行交流探讨，觉收获颇丰，填补了一些之前的知识盲点。

    2018年08月01日：
        学习seq2seq+attention的代码（来自庞云升）
        在讨论时，我们想到，如果给比如Char-RNN一个全是压一个韵的数据，
        是否Char-RNN就可以生成完全压同一个韵的歌词，
        实践证明，可以生成押韵歌词，生成结果可以在Char-RNN-Tensorflow-demo中看到，
        或许应该将输入数据按照所压的韵来划分？

    2018年07月31日：
        找到一个seq2seq+attention的诗歌生成demo，
        demo复现王德瑞导师发至群里的诗歌生成论文，
        跑了一下，效果不是很好，没有语义，句子不通顺，不押韵。

    2018年07月30日：
       学习了一个Char-RNN的demo，
       跑了一个Char-RNN的demo，代码已经上传至GitHub，
       其生成结果的语句，没有语义，句子也并不通顺，同时并不押韵。