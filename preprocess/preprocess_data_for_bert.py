import pandas as pd
import numpy as np
import random
import re
from sklearn.model_selection import KFold

def preprocess_data(df):
    for index in df.index:
        QS = df.loc[index, "Question Sentence"]
        # 去除html标签
        doc_token = re.sub(r'<[^>]+>', "", QS)
        # doc_token = doc_token.replace("......", "")
        # for word in ["病情描述：", "曾经治疗情况和效果：", "想得到怎样的帮助：", "健康咨询描述：", "病情描述（发病时间、主要症状、症状变化等）：",
        #              "病情描述(发病时间、主要症状等)："]:
        #     doc_token = doc_token.replace(word, "")
        # check_duplicates = doc_token.split("...")
        # if len(check_duplicates) > 1:
        #     for i in range(len(check_duplicates) - 1):
        #         if check_duplicates[i] in check_duplicates[i + 1]:
        #             doc_token = doc_token[len(check_duplicates[i]) + 3:]
        #         elif check_duplicates[i + 1][0:5] in check_duplicates[i]:
        #             doc_token = doc_token[len(check_duplicates[i]) + 3:]
        # print(doc_token)
        # 只保留中文、大小写字母和阿拉伯数字
        reg = "[^0-9A-Za-z\u4e00-\u9fa5]"
        doc_token = re.sub(reg, '', doc_token)
        df.loc[index, "Question Sentence"] = doc_token
    print("finish preprocess!")

def write_text(X, y, output_path):
    X = list(X)
    y = list(y)
    with open(output_path, "w+", encoding='utf-8') as f:
        for i in range(len(X)):
            QS = X[i]
            label = y[i]
            f.write(QS + "\t" + str(label)+"\n")
    print("finish write {}".format(output_path))

if __name__ == '__main__':
    # 读入数据
    file_name = "../data/train.csv"
    df1 = pd.read_csv(file_name, encoding="utf-8")
    print(df1.shape)
    preprocess_data(df1)

    test_df = pd.read_csv("../data/nlp_test.csv")
    preprocess_data(test_df)
    X = test_df["Question Sentence"].tolist()

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    k = 0
    for train_index, test_index in kf.split(df1):
        k += 1
        train_df = df1.loc[train_index].reset_index()
        valid_df = df1.loc[test_index].reset_index()
        for c in ["category_A", "category_B", "category_D", "category_E", "category_F"]:
            ll = c[len(c)-1:]
            sub_df = df1[["Question Sentence", c]]
            X = sub_df["Question Sentence"].values
            y = sub_df[c].values
            print(c, len(train_index), len(test_index))
            X_train, X_test = X[train_index], X[test_index]  # 训练集对应的值
            y_train, y_test = y[train_index], y[test_index]  # 类别集对应的值

            write_text(X_train, y_train, "../data/class2/"+ll+"/train{}.txt".format(k))
            write_text(X_test, y_test, "../data/class2/"+ll+"/dev{}.txt".format(k))

    with open("../data/class2/test.txt", "w+", encoding='utf-8') as f:
        for i in X:
            m = random.sample([0,1], 1)[0]
            f.write(i + "\t" + str(m) + "\n")