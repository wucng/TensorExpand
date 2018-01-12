#! /usr/bin/python
# -*- coding: utf8 -*-

'''
验证码生成器

'''

from captcha.image import ImageCaptcha  # pip install captcha
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random

# 验证码中的字符, 就不用汉字了
number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
alphabet = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']


# 验证码一般都无视大小写；验证码长度4个字符
def random_captcha_text(char_set=number + alphabet + ALPHABET, captcha_size=4):
    '''
    随机生成4个字符
    :param char_set: 验证码包含的所有字符（数字+大小写字母）
    :param captcha_size: 验证码尺寸 默认为4个
    :return: 
    '''
    captcha_text = []
    for i in range(captcha_size):
        c = random.choice(char_set) # 随机选取字符
        captcha_text.append(c)
    return captcha_text


# 生成字符对应的验证码
def gen_captcha_text_and_image():
    '''
    根据随机选取的字符，生成对应字符的验证码（图像）
    :return: 
    '''
    image = ImageCaptcha()

    captcha_text = random_captcha_text()
    captcha_text = ''.join(captcha_text) # 列表转字符串
    # 如：x=['1','a','2','c'] ; s=''.join(x) ; s ==> '1a2c'

    captcha = image.generate(captcha_text) # 生成验证码（图）
    # image.write(captcha_text, captcha_text + '.jpg')  # 写到文件

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image) # 转成numpy
    return captcha_text, captcha_image # captcha_text 为验证码字符文本（可以做标签），captcha_image验证码图像


if __name__ == '__main__':
    # 测试
    text, image = gen_captcha_text_and_image()

    f = plt.figure()
    ax = f.add_subplot(111)
    ax.text(0.1, 0.9, text, ha='center', va='center', transform=ax.transAxes)
    plt.imshow(image)

    plt.show()
