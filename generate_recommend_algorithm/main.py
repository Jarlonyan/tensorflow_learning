#coding=utf-8
#https://github.com/jxyyjm/tensorflow_test/blob/master/src/deep_and_cross.py

import tensorflow as tf
import argparse
import sys

from utils import tools

_CSV_COLUMNS = [
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship',
    'race', 'gender', 'capital_gain', 'capital_loss', 'hours_per_week', 'native_country', 'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [[0], [''], [0], [''], [0], [''], [''], [''], [''], [''], [0], [0], [0], [''], ['']]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281,
}

# 1. 最基本的特征：
# Continuous columns. Wide和Deep组件都会用到。
def build_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    # 离散特征
    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education',
        ['Bachelors', 'HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', \
         'Prof-school', '5th-6th', '10th', '1st-4th', 'Preschool', '12th']
    )

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status',
        ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Married-AF-spouse', 'Widowed']
    )

    relationship = tf.feature_column.categorical_column_with_vocabulary_list(
        'relationship',
        ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative']
    )

    workclass = tf.feature_column.categorical_column_with_vocabulary_list(
        'workclass',
        ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov', 'Local-gov', '?', 'Self-emp-inc', 'Without-pay', 'Never-worked']
    )

    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

    # Transformations
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

    # 2. The Wide Model: Linear Model with CrossedFeatureColumns
    """
    The wide model is a linear model with a wide set of *sparse and crossed feature* columns
    Wide部分用了一个规范化后的连续特征age_buckets，其他的连续特征没有使用
    """
    base_columns = [  # 全是离散特征
        education, marital_status, relationship, workclass, occupation, age_buckets,
    ]

    crossed_columns = [
        tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=1000)
    ]
    wide_columns = base_columns + crossed_columns

    # 3. The Deep Model: Neural Network with Embeddings
    """
    1. Sparse Features -> Embedding vector -> 串联(Embedding vector, 连续特征) -> 输入到Hidden Layer
    2. Embedding Values随机初始化
    3. 另外一种处理离散特征的方法是：one-hot or multi-hot representation. 但是仅仅适用于维度较低的，embedding是更加通用的做法
    4. embedding_column(embedding);indicator_column(multi-hot);
    """
    columns = [
        age,
        education_num,
        capital_gain,
        capital_loss,
        hours_per_week,
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),

        # To show an example of embedding
        tf.feature_column.embedding_column(occupation, dimension=8)
    ]

    return columns

def cross_variable_create(column_num):
    w = tf.Variable(tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
    b = tf.Variable(tf.random_normal((column_num, 1), mean=0.0, stddev=0.5), dtype=tf.float32)
    return w,b

def cross_op(x0, x, w, b):
    x0 = tf.expand_dims(x0, axis=2)
    x = tf.expand_dims(x, axis=2)

    multiple = w.get_shape().as_list()[0]
    x_broad_vertical = tf.



def train_model():
    wide_columns, deep_columns = build_columns()
    training_inputs = tf.feature_column.input_layer(wide_columns, deep_columns) 

if __name__ == '__main__':
    train_model()

