# -*- coding: utf-8 -*-

import os
import random
import numpy as np
 
conv_path = 'dgk_shooter_min.conv'
 
if not os.path.exists(conv_path):
	print('数据集不存在')
	exit()
 
# 数据集格式
"""
E
M 畹/华/吾/侄/
M 你/接/到/这/封/信/的/时/候/
M 不/知/道/大/伯/还/在/不/在/人/世/了/
E
M 咱/们/梅/家/从/你/爷/爷/起/
M 就/一/直/小/心/翼/翼/地/唱/戏/
M 侍/奉/宫/廷/侍/奉/百/姓/
M 从/来/不/曾/遭/此/大/祸/
M 太/后/的/万/寿/节/谁/敢/不/穿/红/
M 就/你/胆/儿/大/
M 唉/这/我/舅/母/出/殡/
M 我/不/敢/穿/红/啊/
M 唉/呦/唉/呦/爷/
M 您/打/得/好/我/该/打/
M 就/因/为/没/穿/红/让/人/赏/咱/一/纸/枷/锁/
M 爷/您/别/给/我/戴/这/纸/枷/锁/呀/
E
M 您/多/打/我/几/下/不/就/得/了/吗/
M 走/
M 这/是/哪/一/出/啊/…/ / /这/是/
M 撕/破/一/点/就/弄/死/你/
M 唉/
M 记/着/唱/戏/的/再/红/
M 还/是/让/人/瞧/不/起/
M 大/伯/不/想/让/你/挨/了/打/
M 还/得/跟/人/家/说/打/得/好/
M 大/伯/不/想/让/你/再/戴/上/那/纸/枷/锁/
M 畹/华/开/开/门/哪/
E
...
"""
 
# 我首先使用文本编辑器sublime把dgk_shooter_min.conv文件编码转为UTF-8，一下子省了不少麻烦
convs = []  # 对话集合
with open(conv_path, encoding = "utf8") as f:
	one_conv = []        # 一次完整对话
	for line in f:
		line = line.strip('\n').replace('/', '')
		if line == '':
			continue
		if line[0] == 'E':
			if one_conv:
				convs.append(one_conv)
			one_conv = []
		elif line[0] == 'M':
			one_conv.append(line.split(' ')[1])
"""
print(convs[:3])  # 个人感觉对白数据集有点不给力啊
[ ['畹华吾侄', '你接到这封信的时候', '不知道大伯还在不在人世了'], 
  ['咱们梅家从你爷爷起', '就一直小心翼翼地唱戏', '侍奉宫廷侍奉百姓', '从来不曾遭此大祸', '太后的万寿节谁敢不穿红', '就你胆儿大', '唉这我舅母出殡', '我不敢穿红啊', '唉呦唉呦爷', '您打得好我该打', '就因为没穿红让人赏咱一纸枷锁', '爷您别给我戴这纸枷锁呀'], 
  ['您多打我几下不就得了吗', '走', '这是哪一出啊 ', '撕破一点就弄死你', '唉', '记着唱戏的再红', '还是让人瞧不起', '大伯不想让你挨了打', '还得跟人家说打得好', '大伯不想让你再戴上那纸枷锁', '畹华开开门哪'], ....]
"""
 
# 把对话分成问与答
ask = []        # 问
response = []   # 答
for conv_ in convs:
    if len(conv_) == 1:
        continue
    if len(conv_) % 2 != 0:  # 奇数对话数, 转为偶数对话
        conv_ = conv_[:-1]
    for i in range(len(conv_)):
        if i % 2 == 0:
            ask.append(conv_[i])
        else:
            response.append(conv_[i])
 
"""
print(len(ask), len(response))
print(ask[:3])
print(response[:3])
['畹华吾侄', '咱们梅家从你爷爷起', '侍奉宫廷侍奉百姓']
['你接到这封信的时候', '就一直小心翼翼地唱戏', '从来不曾遭此大祸']
"""
 
def convert_seq2seq_files(questions, answers, TESTSET_SIZE = 8000):
    # 创建文件
    train_enc = open('train.enc','w')  # 问
    train_dec = open('train.dec','w')  # 答
    test_enc  = open('test.enc', 'w')  # 问
    test_dec  = open('test.dec', 'w')  # 答

    # 选择20000数据作为测试数据
    test_index = random.sample([i for i in range(len(questions))],TESTSET_SIZE) # 随机选取TESTSET_SIZE 个数字 ，返回list
 
    for i in range(len(questions)):
        if i in test_index:
            test_enc.write(questions[i]+'\n')
            test_dec.write(answers[i]+ '\n' )
        else:
            train_enc.write(questions[i]+'\n')
            train_dec.write(answers[i]+ '\n' )
        if i % 1000 == 0:
            print(len(range(len(questions))), '处理进度:', i)
    '''
 	# -----------------------------
	# 数据打乱
	np.random.seed(100)
	test_index=np.arange(0,len(questions),dtype=np.int32)
	np.random.shuffle(test_index)
	questions=np.array(questions)[test_index]
	answers = np.array(answers)[test_index]
	# ------------------------------
	'''

    train_enc.close()
    train_dec.close()
    test_enc.close()
    test_dec.close()
 
convert_seq2seq_files(ask, response)
# 生成的*.enc文件保存了问题
# 生成的*.dec文件保存了回答


# 前一步生成的问答文件路径
train_encode_file = 'train.enc'
train_decode_file = 'train.dec'
test_encode_file = 'test.enc'
test_decode_file = 'test.dec'

print('开始创建词汇表...')
# 特殊标记，用来填充标记对话
PAD = "__PAD__"
GO = "__GO__"
EOS = "__EOS__"  # 对话结束
UNK = "__UNK__"  # 标记未出现在词汇表中的字符
START_VOCABULART = [PAD, GO, EOS, UNK]
PAD_ID = 0
GO_ID = 1
EOS_ID = 2
UNK_ID = 3
# 参看tensorflow.models.rnn.translate.data_utils

vocabulary_size = 5000


# 生成词汇表文件
def gen_vocabulary_file(input_file, output_file):
	vocabulary = {}
	with open(input_file) as f:
		counter = 0
		for line in f:
			counter += 1
			tokens = [word for word in line.strip()]
			for word in tokens:
				if word in vocabulary:
					vocabulary[word] += 1
				else:
					vocabulary[word] = 1
		vocabulary_list = START_VOCABULART + sorted(vocabulary, key=vocabulary.get, reverse=True)
		# 取前5000个常用汉字, 应该差不多够用了(额, 好多无用字符, 最好整理一下. 我就不整理了)
		if len(vocabulary_list) > 5000:
			vocabulary_list = vocabulary_list[:5000]
		print(input_file + " 词汇表大小:", len(vocabulary_list))
		with open(output_file, "w") as ff:
			for word in vocabulary_list:
				ff.write(word + "\n")


gen_vocabulary_file(train_encode_file, "train_encode_vocabulary")
gen_vocabulary_file(train_decode_file, "train_decode_vocabulary")

train_encode_vocabulary_file = 'train_encode_vocabulary'
train_decode_vocabulary_file = 'train_decode_vocabulary'

print("对话转向量...")

''' append与extend区别
a=[1,2]
c=[]
c.append(a) # [[1, 2]]
d=[]
d.extend(a) # [1, 2]
'''
# 把对话字符串转为向量形式
def convert_to_vector(input_file, vocabulary_file, output_file):
	tmp_vocab = []
	with open(vocabulary_file, "r") as f:
		tmp_vocab.extend(f.readlines()) # f.readlines() 按行读取所有行（包含'\n'） extend
	tmp_vocab = [line.strip() for line in tmp_vocab] # 得到词汇表（不包含重复的词） strip()去除换行符'\n'
	vocab = dict([(x, y) for (y, x) in enumerate(tmp_vocab)]) # 词与索引号对应 得到词汇字典
	# {'硕': 3142, 'v': 577, 'Ｉ': 4789, '\ue796': 4515, '拖': 1333, '疤': 2201 ...}
	output_f = open(output_file, 'w')
	with open(input_file, 'r') as f:
		for line in f:
			line_vec = []
			for words in line.strip(): # line.strip() 去除换行符 '\n'
				line_vec.append(vocab.get(words, UNK_ID)) # vocab.get(words, UNK_ID) 如果words不在字典中使用 UNK_ID作为它的值
			output_f.write(" ".join([str(num) for num in line_vec]) + "\n") # 转成字符串 写入文本  " ".join(map(str,line_vec))
	output_f.close()


convert_to_vector(train_encode_file, train_encode_vocabulary_file, 'train_encode.vec')
convert_to_vector(train_decode_file, train_decode_vocabulary_file, 'train_decode.vec')

convert_to_vector(test_encode_file, train_encode_vocabulary_file, 'test_encode.vec')
convert_to_vector(test_decode_file, train_decode_vocabulary_file, 'test_decode.vec')
