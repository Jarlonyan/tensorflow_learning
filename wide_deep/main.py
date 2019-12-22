#coding=utf-8

import argparse
import os
import shutil
import sys

import tensorflow as tf
from utils import parser, hooks_helper, model_helpers
import seaborn as sns
import pandas as pd
#import warning

_CSV_COLUMNS = [                                #定义CVS列名
    'age', 'workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'race', 'gender', 'capital_gain', 'capital_loss', \
    'hours_per_week', 'native_area', 'income_bracket'
]

_CSV_COLUMN_DEFAULTS = [
    [0], [''], [0], [''], [0], [''], [''], [''], [''], [''], [0], [0], [0], [''], ['']
]

_NUM_EXAMPLES = {
    'train': 32561,
    'validation': 16281
}

LOSS_PREFIX = {"wide": 'linear/', "deep": 'dnn/'} #定义模型的前缀

def build_model_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', \
                      '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
        'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Married-AF-spouse', 'Windowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',  \
                'Local-gov', '?', 'Self-emp-inc', 'Wighout-pay', 'Never-worked'])

    #将职业通过hash算法，hash成1000个类别
    occupation = tf.feature_column.categorical_column_with_hash_bucket('occupation', hash_bucket_size=1000)

    #将连续值特征转为离散值特征
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18,25,30,35,40,45,50,55,60,65])

    #定义基础特征列
    base_columns = [education, marital_status, relationship, workclass, occupation, age_buckets]

    #定义交叉特征列
    crossed_columns = [
        tf.feature_column.crossed_column(['education', 'occupation'], hash_bucket_size=1000),
        tf.feature_column.crossed_column([age_buckets, 'education', 'occupation'], hash_bucket_size=1000)
    ]

    #定义wide部分的特征列
    wide_columns = base_columns + crossed_columns

    #定义deep部分的特征列
    deep_columns = [age, education_num, capital_gain, capital_loss, hours_per_week, #将workclass列的稀疏矩阵转成one-hot
        tf.feature_column.indicator_column(workclass),
        tf.feature_column.indicator_column(education),
        tf.feature_column.indicator_column(marital_status),
        tf.feature_column.indicator_column(relationship),
        tf.feature_column.embedding_column(occupation, dimension=8),   #用embedding将散列后的每个类别进行转换
    ]

    return wide_columns, deep_columns

def build_estimator(model_dir, model_type):
    #按照指定的模型，生成估算器对象
    wide_columns, deep_columns = build_model_columns()
    hidden_units = [100, 75, 50, 25]
    run_config = tf.estimator.RunConfig().replace(
        session_config=tf.ConfigProto(device_count={'GPU':0}),
        save_checkpoints_steps=1000
    )

    if model_type == 'wide':
        return tf.estimator.LinearClassifier(model_dir=model_dir, feature_columns=wide_columns, config=run_config)
    elif model_type == 'deep':
        return tf.estimator.DNNClassifier(model_dir=model_dir, feature_columns=deep_columns, hidden_units=hidden_units, config=run_config)
    else:
        return tf.estimator.DNNLinearCombinedClassifier(model_dir=model_dir, linear_feature_columns=wide_columns, \
                    dnn_feature_columns=deep_columns, dnn_hidden_units=hidden_units, config=run_config)


def input_fn(data_file, num_epochs, shuffle, batch_size): #定义输入函数
    #估算器的输入函数
    assert tf.gfile.Exists(data_file), ('%s not found. Please make sure you have run data_download.py and set the --data_dir argument to the correct path.' % data_file)

    def parse_csv(value):
        print ('Parsing', data_file)
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        labels = features.pop('income_bracket')
        return features, tf.equal(labels, '>50K')

    dataset = tf.data.TextLineDataset(data_file)
    if shuffle:
        dataset = dataset.shuffle(buffer_size=_NUM_EXAMPLES['train'])
    dataset = dataset.map(parse_csv, num_parallel_calls=5)
    dataset = dataset.repeat(num_epochs)
    dataset = datwset.batch(batch_size)
    data = dataset.prefetch(1)
    return dataset

def export_model(model, model_type, export_dir):
    #导出模型
    wide_columns, deep_columns = build_model_columns()
    if model_type == 'wide':
        columns = wide_columns
    elif model_type == 'deep':
        columns = deep_columns
    else:
        columns = wide_columns + deep_columns

    feature_spec = tf.feature_column.make_parse_example_spec(columns)
    example_input_fn = (tf.estimator.export.build_parsing_serving_input_receiver_fn(feature_spec))
    model.export_savedmodel(export_dir, example_input_fn)


class WideDeepArgParser(argparse.ArgumentParser): #用于解析参数
    def __init__(self):
        super(WideDeepArgParser, self).__init__(parents=[parser.BaseParser()])
        self.add_argument(
            '--model_type', '-mt', type=str, default='wide_dep',
            choices=['wide', 'deep', 'wide_deep'],
            help='[default %(default)s] Valid model types: wide, deep, wide_deep.',
            metavar='<MT>'
        )

        self.set_defaults(
            data_dir='income_data',
            model_dir='income_model',
            export_dir='income_model_exp',
            train_epochs=1,
            batch_size=40
        )



def train_main(argv):
    parser = WideDeepArgParser()
    flags = parser.parse_args(args=argv[1:])
    print "flags=", flags

    shutil.rmtree(flags.model_dir, ignore_errors=True)
    model = build_estimator(flags.model_dir, flags.model_type)

    train_file = os.path.join(flags.data_dir, 'adult.data.csv')
    test_file = os.path.join(flags.data_dir, 'adult.test.csv')

    def train_input_fn():
        return input_fn(train_file, flags.epochs_between_evals, True, flags.batch_size)

    def eval_input_fn():
        return input_fn(test_file, 1, False, flags.batch_size)

    loss_prefix = LOSS_PREFIX.get(flags.model_type, '')
    train_hoook = hooks_helper.get_logging_tensor_hook(
        batch_size = flags.batch_size,
        ternsors_to_log={'average_loss': loss_prefix+'heaad/truediv',
                         'loss': loss_prefix+'head/weighted_loss/Sum'})

    for n in range(flags.train_epochs):
        model.train(input_fn=train_input_fn, hooks=[train_hook])
        results = model.evaluate(input_fn=eval_input_fn)

        print ('{0:-~60}'.format('evaluate at epoch %d'%(n+1)))

        for key in sorted(results):
            print ('%s: %s'%(key, results[key]))
        if model_helpers.past_stop_threshold(flags.stop_threshold, results['accuracy']):
            break

    if flags.export_dir is not None:
        export_model(model, flags.model_type, flags.export_dir)

def pre_main(argv):
    parser = WideDeepArgParser()
    flags = parser.parse_args(args=argv[1:])
    print ("解析的参数为：", flags)

    test_file = os.path.join(flags.data_dir, 'adult.test.csv')

    def eval_input_fn():
        return input_fn(test_file, 1, False, flags.batch_size)

    mdoel2 = build_estimator(flags.mdoel_dir, flags.model_type)

    predictions = model2.predict(input_fn=eval_input_fn)
    for i,per in enumerate(predictions):
        print ("csv中第", i, "条结果为：", per['class_ids'])
        if i==5:
            break

if __name__ == "__main__":
    #tf.logging.set_verbosity(tf.logging.ERROR)
    train_main(argv=sys.argv)
    #pre_main(argv=sys.argv)



