# Char-RNN-Tensorflow-demo
 一个使用Char-RNN来生成rap歌词的demo

运行环境：
  python3.6
  tensorflow1.9
  
数据：
  目前data文件夹中使用的数据是老师上传的数据文件中的一个，我忘记是哪个了
  
  获得处理后的数据以后，回来更新一下结果。

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
我的心
不有我的一直 不能再你的你
我的心　的面
你的人都是自己的
我不是贪心鬼
什么的心都能是你变得
当我不会不会
不要就能一个一点
我也会的
我们说你的
但你都要的我的人
你们说你的
就要你不是我
要说你们都是我
你的我的心
我的心
不能就想一直 你不是
我们不想你走
要不可挡
你说我想我 biby bome bond
我们的不有一直不有 我的人不是
不是不有 我们都以我的
我不会要我的人 不不能
我不懂我的一直都是我们
不要就要 你不要
不是我的  我的    你         呀呀呀啊  不会
我想是你的名   是 我的
你们 别不能
你就是你不能我 我能不要站
你不是帅~帅
我就是帅~帅~帅~帅~帅
我就是帅 帅~帅~帅
我就可是我所以懂
我不懂我 你不懂我
别不懂我 你想懂我
你不懂你 想不懂我
我不懂我我不懂我
你就是帅
不以我DIY 我 可以你DIY   O N NON NONON
你就是帅
你就是帅~帅~帅~帅
我怎就不能
我就不是你想懂我 不不要红~帅
我想懂我的一切
别想要
