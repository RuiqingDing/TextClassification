import numpy as np
import pandas as pd
import jieba
import jieba.analyse
import pkuseg
import codecs
import sys
import json
import re
import random
from sklearn.model_selection import KFold

# 目标 转化成
"""
JSON example:
{
    "doc_label": ["Computer--MachineLearning--DeepLearning", "Neuro--ComputationalNeuro"],
    "doc_token": ["I", "love", "deep", "learning"],
    "doc_keyword": ["deep learning"],
    "doc_topic": ["AI", "Machine learning"]
}
"""

# 停用词
def stop_words(path):
    with open(path, "r", encoding="utf-8") as f:
        return set([l for l in f])

def transfer_data(df, output_path):
    with open(output_path, "w+", encoding='utf-8') as f:
        for index in df.index:
            dict1 = {}
            doc_label = df[["category_A","category_B","category_C","category_D","category_E","category_F"]].loc[index].values

            if np.sum(doc_label) == 0:
                dict1['doc_label'] = ["G"]
            else:
                dict1['doc_label'] = []
                for i, l in enumerate(["A", "B", "C", "D", "E", "F"]):
                    if doc_label[i] == 1:
                        dict1['doc_label'].append(l)
            doc_token = df[["Question Sentence"]].loc[index].values[0]
            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            # for word in ["病情描述", "曾经治疗情况和效果", "想得到怎样的帮助"]:
            #     doc_token = doc_token.replace(word, "")

            # print(doc_token)
            # 中文分词
            seg_list = jieba.cut(doc_token, cut_all=False)
            # # 去除停用词
            # content = [x for x in seg_list if x not in stop_words('data/stopwords.txt')]
            content = [x for x in seg_list]
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            # 组合成字典
            if (index + 1) % 500 == 0:
                print("have processed {} questions".format(index + 1))
                print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)
    print("Finish transfering dataframe to json")

def transfer_data2(df, output_path):
    ## using 2-gram + jieba to generate x
    with open(output_path, "w+", encoding='utf-8') as f:
        for index in df.index:
            dict1 = {}
            doc_label = df[["category_A","category_B","category_C","category_D","category_E","category_F"]].loc[index].values

            if np.sum(doc_label) == 0:
                dict1['doc_label'] = ["G"]
            else:
                dict1['doc_label'] = []
                for i, l in enumerate(["A", "B", "C", "D", "E", "F"]):
                    if doc_label[i] == 1:
                        dict1['doc_label'].append(l)
            doc_token = df[["Question Sentence"]].loc[index].values[0]
            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            # print(doc_token)
            content = []
            ngram = 2
            for i in range(len(doc_token) - ngram + 1):
                content.append(doc_token[i:(i + ngram)])

            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            # 组合成字典
            if (index + 1) % 500 == 0:
                print("have processed {} questions".format(index + 1))
                print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)
    print("Finish transfering dataframe to json")


def transfer_test(df, output_path):
    with open(output_path, "w+", encoding='utf-8') as f:
        for index in df.index:
            dict1 = {}
            dict1['doc_label'] = []
            doc_token = df[["Question Sentence"]].loc[index].values[0]
            # 去除html标签
            doc_token = re.sub(r'<[^>]+>', "", doc_token)

            # doc_token = doc_token.replace("......", "")
            # for word in ["病情描述：", "曾经治疗情况和效果：", "想得到怎样的帮助：", "健康咨询描述：", "病情描述（发病时间、主要症状、症状变化等）：", "病情描述(发病时间、主要症状等)："]:
            #     doc_token = doc_token.replace(word, "")
            # check_duplicates = doc_token.split("...")
            # if len(check_duplicates) > 1:
            #     for i in range(len(check_duplicates)-1):
            #         if check_duplicates[i] in check_duplicates[i+1]:
            #             doc_token = doc_token[len(check_duplicates[i])+3 : ]
            #         elif check_duplicates[i+1][0:5] in check_duplicates[i]:
            #             doc_token = doc_token[len(check_duplicates[i]) + 3:]
            # print(doc_token)

            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            # 中文分词
            seg_list = jieba.cut(doc_token, cut_all=False)
            # # 去除停用词
            # content = [x for x in seg_list if x not in stop_words('data/stopwords.txt')]
            content = [x for x in seg_list]
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            # 组合成字典
            if (index + 1) % 500 == 0:
                print("have processed {} questions".format(index + 1))
                print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)
    print("Finish transfering dataframe to json")

def transfer_test2(df, output_path):
    with open(output_path, "w+", encoding='utf-8') as f:
        for index in df.index:
            dict1 = {}
            dict1['doc_label'] = []
            doc_token = df[["Question Sentence"]].loc[index].values[0]
            # 去除html标签
            doc_token = re.sub(r'<[^>]+>', "", doc_token)
            # 只保留中文、大小写字母和阿拉伯数字
            reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
            doc_token = re.sub(reg, '', doc_token)
            content = []
            ngram = 2
            for i in range(len(doc_token) - ngram + 1):
                content.append(doc_token[i:(i + ngram)])
            dict1['doc_token'] = content
            dict1['doc_keyword'] = []
            dict1['doc_topic'] = []
            # 组合成字典
            if (index + 1) % 500 == 0:
                print("have processed {} questions".format(index + 1))
                print(dict1)
            # 将字典转化成字符串
            json_str = json.dumps(dict1, ensure_ascii=False)
            f.write('%s\n' % json_str)
    print("Finish transfering dataframe to json")

if __name__ == '__main__':
    # 读入数据
    input_path = "../data/"
    file_name = input_path + "train.csv"
    df1 = pd.read_csv(file_name, encoding="utf-8")
    print(df1.shape)

    # 删除无标签的数据
    # df1["sum"] = df1["category_A"]+df1["category_B"]+df1["category_C"]+df1["category_D"]+df1["category_E"]+df1["category_F"]
    # print("len(df1) = ", len(df1))
    # df1 = df1[df1["sum"] > 0].reset_index()
    # print("After deleting no label's samples, len(df1) = ", len(df1))
    # stat_label_num(df1)

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    k = 0
    # for train_index, test_index in kf.split(df1):
    #     k += 1
    #     train_df = df1.loc[train_index].reset_index()
    #     valid_df = df1.loc[test_index].reset_index()
    #
    #     output_path_train = '../data/rcnn/train_2gram_{}.json'.format(k)
    #     transfer_data2(train_df, output_path_train)
    #
    #     output_path_valid = '../data/rcnn/valid_2gram_{}.json'.format(k)
    #     transfer_data2(valid_df, output_path_valid)
    #
    #     output_path_train = '../data/rcnn/train_seg_{}.json'.format(k)
    #     transfer_data2(train_df, output_path_train)
    #
    #     output_path_valid = '../data/rcnn/valid_seg_{}.json'.format(k)
    #     transfer_data2(valid_df, output_path_valid)
    #
    #     valid_df = valid_df[["ID", "category_A", "category_B", "category_C", "category_D", "category_E", "category_F"]]
    #     valid_df.to_csv("../data/rcnn/valid_true_label_{}.txt".format(k), index=False, encoding="utf-8")

    test_df = pd.read_csv('../data/nlp_test.csv')
    output_path_test = '../data/rcnn/test_seg.json'
    transfer_test(test_df, output_path_test)

    output_path_test = '../data/rcnn/test_2gram.json'
    transfer_test2(test_df, output_path_test)