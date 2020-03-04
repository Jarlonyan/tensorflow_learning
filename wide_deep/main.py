#coding=utf-8

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
    deep_columns = [
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

    return  wide_columns, deep_columns


def build_model(model_dir):
    wide_columns, deep_columns = build_columns()

    # 4. Combine Wide & Deep：wide基础上组合Deep
    model = tf.estimator.DNNLinearCombinedClassifier(
        model_dir = model_dir,
        linear_feature_columns = wide_columns,
        dnn_feature_columns = deep_columns,
        dnn_hidden_units = [128, 64, 32]
    )
    return model
    
def input_fn(data_file, epochs, shuffle, batch_size):
    """为Estimator创建一个input function"""
    def parse_csv(line):
        print("Parsing", data_file)
        # tf.decode_csv会把csv文件转换成很a list of Tensor,一列一个。record_defaults用于指明每一列的缺失值用什么填充
        columns = tf.io.decode_csv(line, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K') # tf.equal(x, y) 返回一个bool类型Tensor， 表示x == y, element-wise

    dataset = tf.data.TextLineDataset(data_file).map(parse_csv, num_parallel_calls=5)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'] + _NUM_EXAMPLES['validation'])
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    #iterator = dataset.make_one_shot_iterator()
    #batch_features, batch_labels = iterator.get_next()
    #return batch_features, batch_labels
    return dataset.prefetch(buffer_size=6)

def train_model(argv):
    parser = tools.DNNArgParser()
    flags = parser.parse_args(args=argv[1:])
    print ("args=",flags)

    # build model
    model = build_model(flags.model_dir)
    #start to train model
    for n in range(flags.epochs // flags.epochs_per_eval):
        model.train(input_fn=lambda: input_fn(flags.train_file, flags.epochs_per_eval, True, flags.batch_size))
        results = model.evaluate(input_fn=lambda: input_fn(flags.test_file, 1, False, flags.batch_size))

        # Display Eval results
        print("Results at epoch {0}".format((n+1) * flags.epochs_per_eval))
        print('-'*30)

        for key in sorted(results):
            print("{0:20}: {1:.4f}".format(key, results[key]))

    #save model
    wide_columns, deep_columns = build_columns()
    feature_spec = tf.feature_column.make_parse_example_spec(wide_columns+deep_columns)
    example_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_saved_model(flags.model_dir, example_input_fn)


if __name__ == '__main__':
    train_model(argv=sys.argv)

