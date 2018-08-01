# Char-RNN-Tensorflow-demo
 一个使用Char-RNN来生成rap歌词的demo

运行环境：
  python3.6
  tensorflow1.9
  
数据：
  目前data文件夹中使用的数据是老师上传的数据文件中的一个，我忘记是哪个了

训练（样例）：
  python train.py  \
  --input_file data/rap.txt \
  --num_steps 20 \
  --batch_size 32 \
  --name rap \
  --max_steps 5000 \
  --learning_rate 0.01 \
  --num_layers 3 \
  --use_embedding
  
生成(样例）：
  python sample.py --converter_path model/rap/converter.pkl \
  --checkpoint_path  model/rap  \
  --max_length 500  \
  --use_embedding \
  --num_layers 3 \
  --start_string 我（可以使用其他词语）

输出（样例）：
我的人都没有问题
这是一个你
你
我想你不是自己
我不是 我的兄弟
我的身力
我不想你 我们的你
我们想不要 这是不会在你
我不是意思在我的兄弟
在我的手里都有你
我不想要一切我都不能忘记
你不会忘记我的意思
这么多人都没有办何问题
我只想你你你的你的心气
不会我想你
这么是你
我不知道你是你
我的兄弟 不管我的兄弟
不管你不要的人的心里
这么是我都会有你
没有人可有的意义我不是 不会的意义你
想要不要 我在哪怕都没有问题
你不要的意义 是我们的一切都不能有意思
我的兄弟
不管我只是在这么你
 你是我的兄弟不再在你
我是我们在哪里都有你
这一个人一天 你的身边 我不想你你
不要对你你 不会再想要你
不想我们的你不是 不要你
我不会是一天我都没有的你
你的我的心里 我是你
我们的我的身力
你们都是不会在意
不要再的你
没为人的人不是你
我是你的身气
我不会要去自己 就是你
不会再你不想自己
我不想你
这个你 我们一切一次
不想要 你 不会再 不要的你
你只是我的兄弟
不想对你
我们的一切 我的兄弟
这是我不在意
我想你想你
不是我不想你 没人的意义
不想要再的意义
你想要的你情在的人的心里