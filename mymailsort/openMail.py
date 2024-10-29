import jieba
import nltk
import re
import chardet
import os
import time
import pickle
import random
import pandas as pd
#from numpy.random import random
import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from idlelib.iomenu import encoding
from langdetect import detect
from nltk import word_tokenize
from nltk.corpus import stopwords
from email import message_from_string
from email.header import decode_header
from nltk.stem import WordNetLemmatizer


# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000个字节进行编码检测
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    return encoding

#提取文本
def get_text(path):
    encoding=detect_encoding(path)
    mail_index = open(path, 'r' , encoding=encoding, errors='ignore')
    TextList = [text for text in mail_index]
    try:
        empty_line_index = TextList.index('\n')  # 空行的索引
        # 提取邮件正文部分（空行之后的内容）
        text = ''.join(TextList[empty_line_index + 1:])
    except ValueError:
        # 如果未找到空行，则假设整个内容为正文
        text = ''.join(TextList)

    text = re.sub( "\u3000", "", re.sub("\n", "", text))  # 去空格分隔符及一些特殊字符
    return text

#判断语言
def detect_language_langdetect(text):
    try:
        lang = detect(text)
        if lang == 'zh-cn' or lang == 'zh-tw':
            return "Chinese"
        elif lang == 'en':
            return "English"
        else:
            return "Other"
    except:
        return "Unknown"


def get_data_label():
    mail_index = open(r"D:\code\pythoncode\mailsort\trec06c\full\index", "r", encoding="gb2312", errors='ignore')
    index_list = [t for t in mail_index]
    index_split = [x.split() for x in index_list if len(x.split()) == 2]  # 分割了标记和路径
    path_list = [y[1].replace('..', '../mailsort/trec06c') for y in index_split]
    label_list = [1 if y[0] == "spam" else 0 for y in index_split]  # 1：垃圾邮件；0：正常邮件
    return path_list, label_list

with open(r'D:\code\pythoncode\mailsort\stopwords.txt', encoding='utf8') as file:
    file_str = file.read()
    stopword_list = file_str.split('\n')

stopword_list.extend(['--', '..', ',', '─', '☆', ' '])

test_folder_path = r"D:\code\pythoncode\mailsort\test_100"  # 测试集文件夹路径
test_file_paths = [os.path.join(test_folder_path, fname) for fname in os.listdir(test_folder_path)]
test_list = []
for filename in os.listdir(test_folder_path):
    #print(filename)
    file_path = os.path.join(test_folder_path, filename)
    content = get_text(file_path)
    test_list.append(content)
    print(filename,detect_language_langdetect(content),':',content)

lemmatizer = WordNetLemmatizer()
test_cutWords_list = []
interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%','--']   #定义标点符号列表
startTime = time.time()
i=0
for mail in test_list:
    i=i+1
    print("cutting test: %.4f"%(i/len(test_list)))
    if detect_language_langdetect(mail) == "Chinese":
        cutWords = [k for k in jieba.lcut(mail) if k not in set(stopword_list)]
    else:
        cutWords = [lemmatizer.lemmatize(k) for k in word_tokenize(mail) if k not in interpunctuations and k.casefold() not in stopwords.words('english')]
    test_cutWords_list.append(cutWords)
# with open(r'D:\code\pythoncode\mailsort\test_cutWords_list.pkl', 'wb') as f:
#     pickle.dump(test_cutWords_list, f)
# with open(r'D:\code\pythoncode\mailsort\trec06c\test_cutWords_list.pkl', 'wb') as f:
#      pickle.dump(test_cutWords_list, f)

print('分词用时%.2f秒' % (time.time() - startTime))

