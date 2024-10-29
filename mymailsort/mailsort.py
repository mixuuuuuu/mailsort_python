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
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle
from idlelib.iomenu import encoding
from langdetect import detect
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from email import message_from_string

# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('punkt_tab')
# nltk.download('stopwords')

#检测编码
def detect_encoding(file_path):
    with open(file_path, 'rb') as f:
        raw_data = f.read(10000)  # 读取前10000个字节进行编码检测
        result = chardet.detect(raw_data)
        encoding = result['encoding']
    return encoding
# def detect_encoding(file_path):
#     with open(file_path, 'rb') as f:
#         raw_data = f.read()
#         result = chardet.detect(raw_data)
#         encoding = result['encoding'] if result['confidence'] > 0.5 else 'utf-8'
#
#     mail_index = open(file_path, 'r', encoding=encoding, errors='ignore')
#     msg = message_from_string(mail_index.read())
#     # 获取邮件内容编码
#     charset = msg.get_content_charset()  # 例如 'GB2312' 或 None
#     return charset

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

    text = re.sub( "\u3000", "", re.sub("\n", "", text))  # 去分隔符及一些特殊字符
    return text
# def get_text(path):
#     encoding=detect_encoding(path)
#     # 解析邮件内容
#     if encoding=='gb2312':
#         mail_index = open(path, 'r', encoding='GB2312', errors='ignore')
#     else:
#         mail_index = open(path, 'r', encoding='utf8', errors='ignore')
#
#     TextList = [text for text in mail_index]
#     try:
#         empty_line_index = TextList.index('\n')  # 空行的索引
#         # 提取邮件正文部分（空行之后的内容）
#         text = ''.join(TextList[empty_line_index + 1:])
#     except ValueError:
#         # 如果未找到空行，则假设整个内容为正文
#         text = ''.join(TextList)
#
#     text = re.sub('\\s+', '', re.sub("\u3000", "", re.sub("\n", "", text)))  # 去空格分隔符及一些特殊字符
#     return text

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
# test=get_text(r'D:\code\pythoncode\mailsort\trec06c\data\000\000')
# print(test)
# print(detect_language_langdetect(test))

# 训练集文件夹路径
cn_ham_folder_path = r'D:\code\pythoncode\mailsort\trec06c\ham'
cn_spam_folder_path = r'D:\code\pythoncode\mailsort\trec06c\spam'
en_ham_folder_path = r'D:\code\pythoncode\mailsort\train2024\trainham'
en_spam_folder_path = r'D:\code\pythoncode\mailsort\train2024\trainspam'

# 遍历文件夹中的所有文件并打开
def get_data_list(folder_path,output_path=None):
    data_list=[]
    i=0
    for filename in os.listdir(folder_path):
        i=i+1
        print('getting',filename,i/len(os.listdir(folder_path)))
        file_path = os.path.join(folder_path, filename)
        content = get_text(file_path)
        data_list.append(content)
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(data_list, f)
    return data_list
#中文训练集
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_ham_list.pkl', 'rb') as f:
#     cn_ham_list = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_spam_list.pkl', 'rb') as f:
#     cn_spam_list = pickle.load(f)
with open(r'D:\code\pythoncode\mailsort\trec06c\cn_data_list.pkl', 'rb') as f:
    cn_data_list = pickle.load(f)

# 获取path、label列表
cn_data_path_list, cn_data_label_list = get_data_label()

# 获取所有文本
# cn_data_list = [get_text(Path) for Path in cn_data_path_list]
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_data_list.pkl', 'wb') as f:
#     pickle.dump(cn_data_list, f)
#全量数据
#cn_ham_list=get_data_list(cn_ham_folder_path,r'D:\code\pythoncode\mailsort\trec06c\cn_ham_list.pkl')
#cn_spam_list=get_data_list(cn_spam_folder_path,r'D:\code\pythoncode\mailsort\trec06c\cn_spam_list.pkl')


#抽样数据
# random.seed(42)
# ham_range=range(0,21766)
# ham_random=random.sample(ham_range, 2082)
# spam_range=range(0,42854)
# spam_random=random.sample(spam_range, 935)
# cn_ham_list_range = [cn_ham_list[i] for i in ham_random]
# cn_spam_list_range = [cn_spam_list[i] for i in spam_random]

#中文分词
def get_cutWords_list_cn(data_list,output_path=None):
    cutWords_list=[]
    i=0
    for mail in data_list:
        i=i+1
        print("cutting: %.4f%%"%((i/len(data_list))*100))
        cutWords = [k for k in jieba.lcut(mail) if k not in set(stopword_list)]
        cutWords_list.append(cutWords)
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(cutWords_list, f)
    return cutWords_list
#jieba分词
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_ham_cutWords_list_range.pkl', 'rb') as f:
#     cn_ham_cutWords_list_range = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_spam_cutWords_list_range.pkl', 'rb') as f:
#     cn_spam_cutWords_list_range = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_ham_cutWords_list.pkl', 'rb') as f:
#     cn_ham_cutWords_list = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\trec06c\cn_spam_cutWords_list.pkl', 'rb') as f:
#     cn_spam_cutWords_list = pickle.load(f)

with open(r'D:\code\pythoncode\mailsort\trec06c\cn_data_cutWords_list.pkl', 'rb') as f:
    cn_data_cutWords_list = pickle.load(f)

# startTime = time.time()
# cn_ham_cutWords_list_range = get_cutWords_list_cn(cn_ham_list_range,r'D:\code\pythoncode\mailsort\trec06c\cn_ham_cutWords_list_range.pkl')
# cn_spam_cutWords_list_range = get_cutWords_list_cn(cn_spam_list_range,r'D:\code\pythoncode\mailsort\trec06c\cn_spam_cutWords_list_range.pkl')
# cn_ham_cutWords_list=get_cutWords_list_cn(cn_ham_list,r'D:\code\pythoncode\mailsort\trec06c\cn_ham_cutWords_list.pkl')
# cn_spam_cutWords_list=get_cutWords_list_cn(cn_spam_list,r'D:\code\pythoncode\mailsort\trec06c\cn_spam_cutWords_list.pkl')
# cn_data_cutWords_list=get_cutWords_list_cn(cn_data_list,r'D:\code\pythoncode\mailsort\trec06c\cn_data_cutWords_list.pkl')
# print('jieba分词用时%.2f秒' % (time.time() - startTime))

# 英文训练集
# with open(r'D:\code\pythoncode\mailsort\train2024\en_ham_list.pkl', 'rb') as f:
#     en_ham_list = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\train2024\en_spam_list.pkl', 'rb') as f:
#     en_spam_list = pickle.load(f)
with open(r'D:\code\pythoncode\mailsort\train2024\en_word_list.pkl', 'rb') as f:
    en_word_list = pickle.load(f)
with open(r'D:\code\pythoncode\mailsort\train2024\en_word_label_list.pkl', 'rb') as f:
    en_word_label_list = pickle.load(f)

# en_ham_list = get_data_list(en_ham_folder_path,r'D:\code\pythoncode\mailsort\train2024\en_ham_list.pkl')
# en_spam_list = get_data_list(en_spam_folder_path,r'D:\code\pythoncode\mailsort\train2024\en_spam_list.pkl')

# # 合并en_ham_list和en_spam_list
# en_word_list=en_spam_list+en_ham_list
# # 创建标签列表：spam=1，ham=0
# en_word_label_list = [1] * len(en_spam_list)+ [0] * len(en_ham_list)
#
# # 检查是否一致
# assert len(en_word_list) == len(en_word_label_list), "样本数不一致"
#
# # 打乱训练数据
# path_list, label_list = shuffle(en_word_list, en_word_label_list, random_state=0)
# combined = list(zip(en_word_list, en_word_label_list))
# random.shuffle(combined)
# en_word_list[:], en_word_label_list[:] = zip(*combined)
# en_word_list=list(en_word_list)
# en_word_label_list=list(en_word_label_list)
#
# with open(r'D:\code\pythoncode\mailsort\train2024\en_word_list.pkl', 'wb') as f:
#     pickle.dump(en_word_list, f)
# with open(r'D:\code\pythoncode\mailsort\train2024\en_word_label_list.pkl', 'wb') as f:
#     pickle.dump(en_word_label_list, f)


#英文分词
# with open(r'D:\code\pythoncode\mailsort\train2024\en_ham_cutWords_list.pkl', 'rb') as f:
#     en_ham_cutWords_list = pickle.load(f)
# with open(r'D:\code\pythoncode\mailsort\train2024\en_spam_cutWords_list.pkl', 'rb') as f:
#     en_spam_cutWords_list = pickle.load(f)
with open(r'D:\code\pythoncode\mailsort\train2024\en_cutWords_list.pkl', 'rb') as f:
    en_cutWords_list = pickle.load(f)

def get_cutWords_list_en(data_list,output_path=None):
    cutWords_list=[]
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
    lemmatizer = WordNetLemmatizer()
    i=0
    for mail in data_list:
        i=i+1
        print("cutting: %.4f%%"%((i/len(data_list))*100))
        cutWords = [lemmatizer.lemmatize(k) for k in word_tokenize(mail) if k not in interpunctuations and k.casefold() not in stopwords.words('english')]
        cutWords_list.append(cutWords)
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(cutWords_list, f)
    return cutWords_list

# startTime = time.time()
# en_ham_cutWords_list = get_cutWords_list_en(en_ham_list,r'D:\code\pythoncode\mailsort\train2024\en_ham_cutWords_list.pkl')
# en_spam_cutWords_list = get_cutWords_list_en(en_spam_list,r'D:\code\pythoncode\mailsort\train2024\en_spam_cutWords_list.pkl')
# en_cutWords_list = get_cutWords_list_en(en_word_list,r'D:\code\pythoncode\mailsort\train2024\en_cutWords_list.pkl')
# print('nltk分词用时%.2f秒' % (time.time() - startTime))


#将分词结果的每个列表合并成空格分隔的字符串，以便用于向量化处理
# ham_content_text = [' '.join(text) for text in cn_ham_cutWords_list_range + en_ham_cutWords_list]
# spam_content_text = [' '.join(text) for text in cn_spam_cutWords_list_range + en_spam_cutWords_list]
# ham_content_text = [' '.join(text) for text in cn_ham_cutWords_list]
# spam_content_text = [' '.join(text) for text in cn_spam_cutWords_list]

cn_data_content_text=[' '.join(text) for text in cn_data_cutWords_list]
en_data_content_text=[' '.join(text) for text in en_cutWords_list]



# 初始化 CountVectorizer，设置参数用于特征选择
#cv = CountVectorizer(max_features=5000, max_df=0.6, min_df=5)
cv_cn=TfidfVectorizer(
    max_features=5000,
    max_df=0.5,
    min_df=3,
    use_idf=True,
    sublinear_tf=True
)
cv_en=TfidfVectorizer(
    max_features=5000,
    max_df=0.5,
    min_df=3,
    use_idf=True,
    sublinear_tf=True,
    ngram_range=(1, 2)
)

# 将文本转换为词频矩阵
# ham_counts = cv.fit_transform(ham_content_text)
# spam_counts= cv.fit_transform(spam_content_text)
cn_data_counts=cv_cn.fit_transform(cn_data_content_text)
en_data_counts=cv_en.fit_transform(en_data_content_text)

# print(ham_counts)
# print(spam_counts)

tfidf = TfidfTransformer()
# ham_tfidf_matrix = tfidf.fit_transform(ham_counts)
# spam_tfidf_matrix = tfidf.fit_transform(spam_counts)
cn_data_tfidf_matrix=tfidf.fit_transform(cn_data_counts)
en_data_tfidf_matrix=tfidf.fit_transform(en_data_counts)
# print(ham_tfidf_matrix)
# print(spam_tfidf_matrix)


# 将 spam 和 ham 词频矩阵合并
# tfidf_matrix = np.vstack([spam_tfidf_matrix.toarray(), ham_tfidf_matrix.toarray()])
#
# # 创建标签列表：spam=1，ham=0
# label_list = [1] * spam_tfidf_matrix.shape[0] + [0] * ham_tfidf_matrix.shape[0]
# # 验证合并结果
# print("tfidf_matrix shape:", tfidf_matrix.shape)
# print("Label list length:", len(label_list))

# 检查是否一致
# assert tfidf_matrix.shape[0] == len(label_list), "样本数不一致"

# 打乱训练数据
#path_list, label_list = shuffle(tfidf_matrix, label_list, random_state=0)
#combined = list(zip(tfidf_matrix, label_list))
#random.shuffle(combined)
#tfidf_matrix[:], label_list[:] = zip(*combined)
#tfidf_matrix=list(tfidf_matrix)
#label_list=list(label_list)

# # 定义参数网格
# param_grid = {
#     'tfidf__max_df': [0.5, 0.75, 1.0],
#     'tfidf__min_df': [1, 3, 5],
#     'tfidf__max_features': [None, 5000, 10000],
#     'tfidf__ngram_range': [(1, 1), (1, 2)]
# }
#
# # 创建管道
# pipeline = Pipeline([
#     ('tfidf', TfidfVectorizer()),
#     ('clf', MultinomialNB())
# ])
#
# # 执行网格搜索
# grid_search = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
# grid_search.fit(en_data_tfidf_matrix, en_word_label_list)
#
# # 输出最佳参数和最佳得分
# print("Best parameters set found on development set:")
# print(grid_search.best_params_)
# print("Grid scores on development set:")
# means = grid_search.cv_results_['mean_test_score']
# stds = grid_search.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
#     print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


#中文贝叶斯分类器
cn_train_X, cn_test_X, cn_train_y, cn_test_y = train_test_split(cn_data_tfidf_matrix, cn_data_label_list, test_size=0.2, random_state=0)

mnb_cn = MultinomialNB()
startTime = time.time()
mnb_cn.fit(cn_train_X, cn_train_y)  # 训练过程
print('中文贝叶斯分类器训练用时%.2f秒' % (time.time() - startTime))

sc1 = mnb_cn.score(cn_test_X, cn_test_y)  # 在测试集上计算得分
print('准确率为:', sc1)

y_pred1 = mnb_cn.predict(cn_test_X)
print('召回率为:', recall_score(cn_test_y, y_pred1))

precision = precision_score(cn_test_y, y_pred1)
print('精确率为:', precision)

#英文贝叶斯分类器
en_train_X, en_test_X, en_train_y, en_test_y = train_test_split(en_data_tfidf_matrix, en_word_label_list, test_size=0.2, random_state=0)

mnb_en = MultinomialNB()
startTime = time.time()
mnb_en.fit(en_train_X, en_train_y)  # 训练过程
print('英文贝叶斯分类器训练用时%.2f秒' % (time.time() - startTime))

sc1 = mnb_en.score(en_test_X, en_test_y)  # 在测试集上计算得分
print('准确率为:', sc1)

y_pred1 = mnb_en.predict(en_test_X)
print('召回率为:', recall_score(en_test_y, y_pred1))

precision = precision_score(en_test_y, y_pred1)
print('精确率为:', precision)

#中文逻辑回归分类器
lr_cn = LogisticRegressionCV(max_iter=10000)
startTime = time.time()
lr_cn.fit(cn_train_X, cn_train_y)
print('逻辑回归分类器训练用时%.2f秒' % (time.time() - startTime))

sc2 = lr_cn.score(cn_test_X, cn_test_y)
print('准确率为:', sc2)

y_pred2 = lr_cn.predict(cn_test_X)
print('召回率为:', recall_score(cn_test_y, y_pred2))

precision = precision_score(cn_test_y, y_pred2)
print('精确率为:', precision)

#英文逻辑回归分类器
lr_en = LogisticRegressionCV(max_iter=10000)
startTime = time.time()
lr_en.fit(en_train_X, en_train_y)
print('逻辑回归分类器训练用时%.2f秒' % (time.time() - startTime))

sc2 = lr_en.score(en_test_X, en_test_y)
print('准确率为:', sc2)

y_pred2 = lr_en.predict(en_test_X)
print('召回率为:', recall_score(en_test_y, y_pred2))

precision = precision_score(en_test_y, y_pred2)
print('精确率为:', precision)




# 1. 遍历测试集文件夹，读取文件内容
with open(r'D:\code\pythoncode\mailsort\test_list.pkl', 'rb') as f:
    test_list = pickle.load(f)

test_folder_path = r"D:\code\pythoncode\mailsort\test2024"  # 测试集文件夹路径
# test_folder_path=r"D:\code\pythoncode\mailsort\trec06c\test"
test_file_paths = [os.path.join(test_folder_path, fname) for fname in os.listdir(test_folder_path)]

# test_ham_folder_path = r"D:\code\pythoncode\mailsort\train2024\trainham"  # 测试集文件夹路径
# test_spam_folder_path = r"D:\code\pythoncode\mailsort\train2024\trainspam"

# test_ham_file_paths = [os.path.join(test_ham_folder_path, fname) for fname in os.listdir(test_ham_folder_path)]
# test_spam_file_paths = [os.path.join(test_spam_folder_path, fname) for fname in os.listdir(test_spam_folder_path)]

# test_file_paths=test_spam_file_paths+test_ham_file_paths

# 2. 对每个文件内容进行分词并向量化
# with open(r'D:\code\pythoncode\mailsort\trec06c\test_cutWords_list.pkl', 'rb') as f:
#     test_cutWords_list = pickle.load(f)
with open(r'D:\code\pythoncode\mailsort\test_cutWords_list.pkl', 'rb') as f:
    test_cutWords_list = pickle.load(f)

# test_list = get_data_list(test_folder_path,r'D:\code\pythoncode\mailsort\test_list.pkl')
# test_ham_list = get_data_list(test_ham_folder_path,r'D:\code\pythoncode\mailsort\train2024\test_ham_list.pkl')
# test_spam_list = get_data_list(test_spam_folder_path,r'D:\code\pythoncode\mailsort\train2024\test_spam_list.pkl')

# # 合并test_ham_list和test_spam_list
# test_word_list=test_spam_list+test_ham_list
# # 创建标签列表：spam=1，ham=0
# test_word_label_list = [1] * len(test_spam_list)+ [0] * len(test_ham_list)
#
# # 检查是否一致
# assert len(test_word_list) == len(test_word_label_list), "label样本数不一致"
# assert len(test_word_list) == len(test_file_paths), "paths样本数不一致"
#
# # 打乱训练数据
# path_list, label_list = shuffle(test_word_list, test_word_label_list, random_state=0)
# combined = list(zip(test_word_list, test_word_label_list, test_file_paths))
# random.shuffle(combined)
# test_word_list[:], test_word_label_list[:], test_file_paths[:] = zip(*combined)
# test_word_list=list(test_word_list)
# test_word_label_list=list(test_word_label_list)
# test_file_paths=list(test_file_paths)

# with open(r'D:\code\pythoncode\mailsort\train2024\en_word_list.pkl', 'wb') as f:
#     pickle.dump(en_word_list, f)
# with open(r'D:\code\pythoncode\mailsort\train2024\en_word_label_list.pkl', 'wb') as f:
#     pickle.dump(en_word_label_list, f)

#分词
def get_cutWords_list_test(data_list,output_path=None):
    cutWords_list=[]
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']  # 定义标点符号列表
    lemmatizer = WordNetLemmatizer()
    i=0
    for mail in data_list:
        i = i + 1
        print("cutting test: %.4f" % (i / len(data_list)))
        if detect_language_langdetect(mail) == "Chinese":
            cutWords = [k for k in jieba.lcut(mail) if k not in set(stopword_list)]
        else:
            cutWords = [lemmatizer.lemmatize(k) for k in word_tokenize(mail) if
                        k not in interpunctuations and k.casefold() not in stopwords.words('english')]
        cutWords_list.append(cutWords)
    if output_path:
        with open(output_path, 'wb') as f:
            pickle.dump(cutWords_list, f)
    return cutWords

# startTime = time.time()
# test_cutWords_list = get_cutWords_list_test(test_list,r'D:\code\pythoncode\mailsort\test_cutWords_list.pkl')
# print('分词用时%.2f秒' % (time.time() - startTime))


# 将测试集文件内容转化为向量表示

# 将中文和英文分开
test_content_text_cn = []
test_content_text_en = []
indices_cn = []
indices_en = []
empty_indices = []

# 用于判断文档是否为空或者是否全是停用词
def is_empty_or_stopwords(doc, vectorizer):
    return len(vectorizer.build_analyzer()(doc)) == 0

for idx, (text, mail) in enumerate(zip(test_cutWords_list, test_list)):
    joined_text = ' '.join(text)
    if detect_language_langdetect(mail) == "Chinese":
        if is_empty_or_stopwords(joined_text, cv_cn):
            empty_indices.append(idx)
        else:
            test_content_text_cn.append(joined_text)
            indices_cn.append(idx)
    else:
        if is_empty_or_stopwords(joined_text, cv_en):
            empty_indices.append(idx)
        else:
            test_content_text_en.append(joined_text)
            indices_en.append(idx)



# 将中文和英文分别转化为词频矩阵
test_counts_cn = cv_cn.fit_transform(test_content_text_cn)
test_tfidf_matrix_cn = tfidf.fit_transform(test_counts_cn)

test_counts_en = cv_en.fit_transform(test_content_text_en)
test_tfidf_matrix_en = tfidf.fit_transform(test_counts_en)

# 使用训练好的模型对测试集进行预测
# predicted_labels = [None] * len(test_list)  # 初始化预测标签列表
#
# for idx, mail in zip(indices_cn, test_tfidf_matrix_cn):
#     predicted_labels[idx] = mnb_cn.predict(mail)[0]
#
# for idx, mail in zip(indices_en, test_tfidf_matrix_en):
#     predicted_labels[idx] = mnb_en.predict(mail)[0]

predicted_labels = [None] * len(test_list)  # 初始化预测标签列表

for idx, mail in zip(indices_cn, test_tfidf_matrix_cn):
    predicted_labels[idx] = lr_cn.predict(mail)[0]

for idx, mail in zip(indices_en, test_tfidf_matrix_en):
    predicted_labels[idx] = lr_en.predict(mail)[0]


# 对于空文档或者全是停用词的文档，将其预测标签设为默认值
default_label = 0  # 默认值设为 ham
for idx in empty_indices:
    predicted_labels[idx] = default_label

# #将分词结果的每个列表合并成空格分隔的字符串，以便用于向量化处理
# test_content_text = [' '.join(text) for text in test_cutWords_list]
#
# # 将文本转换为词频矩阵
# test_counts = cv_cn.fit_transform(test_content_text)
# test_tfidf_matrix = tfidf.fit_transform(test_counts)
#
# # 3. 使用训练好的模型对新测试集进行预测
# #predicted_labels = mnb_cn.predict(test_tfidf_matrix)
# predicted_labels = mnb_en.predict(test_tfidf_matrix)

# 4. 将分类结果和文件路径保存为每行 "分类结果 文件名" 格式的 txt 文件
# with open("D:\code\pythoncode\mailsort\classification_results.txt", "w", encoding="utf-8") as result_file:
#     for label, file_path in zip(predicted_labels, test_file_paths):
#         file_name=os.path.basename(file_path)
#         print("正在写入分类结果：",file_name)
#         label_str="spam" if label == 1 else "ham"
#         result_file.write(f"{label_str} {file_name}\n")

with open(r"D:\code\pythoncode\mailsort\classification_results.txt", "w", encoding="utf-8") as result_file:
    for label, file_path in zip(predicted_labels, test_file_paths):
        file_name = os.path.basename(file_path)
        label_str = "spam" if label == 1 else "ham"
        result_file.write(f"{label_str} {file_name}\n")


print("分类结果已保存到 classification_results.txt 文件中。")
#
# def load_true_labels(label_file_path):
#     true_labels = []
#     with open(label_file_path, "r", encoding="utf-8") as f:
#         for line in f:
#             label, file_name = line.strip().split()  # 读取标签和文件名
#             true_labels.append(1 if label == "spam" else 0)  # spam = 1, ham = 0
#     return true_labels


# 加载真实标签
#label_file_path = r"D:\code\pythoncode\mailsort\trec06c\test_index"  # 测试集标签文件路径
#true_labels = load_true_labels(label_file_path)
# true_labels = test_word_label_list

# 计算并输出准确率、召回率和精确率
# accuracy = accuracy_score(true_labels, predicted_labels)
# recall = recall_score(true_labels, predicted_labels)
# precision = precision_score(true_labels, predicted_labels)
#
# print("准确率:", accuracy)
# print("召回率:", recall)
# print("精确率:", precision)