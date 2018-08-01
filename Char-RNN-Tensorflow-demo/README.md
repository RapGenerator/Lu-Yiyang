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
我的我不是我
不是不再在我的身体
我说我的心脏
这么的人
就是你的心里
就不是我的人都不是一起
我的我们的时候 我是否
你是你 我的生活
不要有我们不会
我们不会再再不能再
我的我们不要再够
我不想在我的心里
我的我的人都会是你
你们不会再够
你的我的时间
我的时间
你的心情 在你心里都会有一天
我们都在这么
我们的爱我都没有我不会再去去
我的爱我不会让我们的心
你不会会让我一个
不想让你
我的生命
不要不想再去去
我的心里都在一起
你的心情
我们都是我
我说你的心义
我们是我的我的心里
不会在这些
我们的人
我们是你的心
我是不是你的爱
你的心在你的眼子
不要再不要你
我不想再让你
你不会是你一起的
不是我的我们
我不知道你
不会不会再够的人
我的爱
你我的爱 我的我的我们不能够
我不想让我
我的时候 我是我
不会是我一起的心
我的时候
我的我们的爱 我的心
你是我的我
我的心情
不想要不要你的
这个人都没是
我的生于我的时候
我的时间
这些我不知道我
不再再再不会让