#coding=utf-8

import seaborn as sns
import pandas as pd
#import warning

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num'
]


def main():
    df = pd.read_csv("income_data/audlt.data.csv", names=_CSV_COLUMNS, skiprows=0, encoding="ISO-8859-1")
    df.loc[df['income_bracket']=='<=50K', 'income_bracket'] = 0  #字段转化
    df.loc[df['income_bracket']=='>50K', 'income_bracket'] = 1   #字段转化
    df1 = df.dropna(how='all', axis=1)      #数据清洗：将空值去掉
    sns.pairplot(df1)        #生成交叉表


if __name__ == "__main__":
    main()


