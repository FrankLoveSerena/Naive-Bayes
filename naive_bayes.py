#!/usr/bin/env python 
# -*- coding:utf-8 -*-
# __author__ = 'Frank'
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import jieba
import os

# 设置根路径
PATH = 'D:\\data analysis\\PycharmProjects\\datalearning\\Naive Bayes\\text classification'
# 加载停用词
with open(os.path.join(PATH, 'stop', 'stopword.txt'), mode = 'rb') as f:
    STOP_WORDS = [line.strip() for line in f.readlines()]


# 创建数据加载函数
def load_data(root_path):
    # root_path 基础路径
    document_data = []
    label_data = []
    for path, dirs, files in os.walk(root_path):
        for file in files:
            label = path.split("\\")[-1]
            label_data.append(label)
            with open(os.path.join(path, file), 'rb') as f:
                document = f.read()
                document = jieba.cut(document)
                document_data.append(' '.join(document))
    # 返回分词列表和标签列表
    return document_data, label_data


# 创建模型并测试
def multinb_fun(train_doc, train_label, test_doc, test_label):
    # train_doc 训练集数据
    # train_label 训练集标签
    # test_doc 测试集数据
    # test_label 测试集标签
    tfidf = TfidfVectorizer(stop_words = STOP_WORDS, max_df = 0.5, token_pattern = r"\b\w+\b")
    train_feature = tfidf.fit_transform(train_doc)
    model = MultinomialNB(alpha = 0.001)
    model.fit(train_feature, train_label)
    test_tfidf = TfidfVectorizer(stop_words = STOP_WORDS, token_pattern = r"\b\w+\b",
                                 max_df = 0.5, vocabulary = tfidf.vocabulary_)
    test_feature = test_tfidf.fit_transform(test_doc)
    predict_label = model.predict(test_feature)
    x = accuracy_score(test_label, predict_label)
    # 返回准确率
    return x


# 导入数据并运行程序
if __name__ == '__main__':
    training_doc, training_label = load_data(os.path.join(PATH, 'train'))
    testing_doc, testing_label = load_data(os.path.join(PATH, 'test'))
    acscore = multinb_fun(training_doc, training_label, testing_doc, testing_label)
    print('本多项式朴素贝叶斯模型的准确率为：', acscore)

