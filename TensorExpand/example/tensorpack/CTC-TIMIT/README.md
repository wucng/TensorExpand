[LSTM + CTC应用于TIMIT语音识别数据集](https://github.com/tensorpack/tensorpack/tree/master/examples/CTC-TIMIT)

# 安装依赖项：

- python binding for lmdb
`pip install --user lmdb`

- 用于MFCC提取的bob.ap包
	- 安装[blitz](https://github.com/blitzpp/blitz)和openblas作为bob.ap的依赖项
	- pip install --user bob.extension bob.blitz bob.core bob.sp bob.ap

# Prepare Data:
我们假设以下文件结构：

```
TRAIN/
  DR1/
    FCJF0/
      *.WAV     # NIST WAV file
      *.TXT
      *.PHN
  ...
```
将NIST wav格式转换为RIFF wav格式：

```
cd /PATH/TO/TIMIT
find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'
```
提取MFCC特征和音标签，并将所有内容保存到LMDB数据库。 预处理遵循以下设置：

- 连接主义时间分类：用RNN标记未分段的序列数据 - Alex Graves

```
./create-lmdb.py build --dataset /PATH/TO/TIMIT/TRAIN --db train.mdb
./create-lmdb.py build --dataset /PATH/TO/TIMIT/TEST --db test.mdb
```
- 计算训练集的均值/标准值（默认保存为stats.data）：

```
./create-lmdb.py stat --db train.mdb
```
# Train:

```
./train-timit.py --train train.mdb --test test.mdb --stat stats.data
```
# Results:
在大约40个epochs之后获得0.28 LER（normalized edit distance）。
