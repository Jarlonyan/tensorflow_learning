#coding=utf-8

import argsparse
import os
import shutil
import sys

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

def build_mdoel_columns():
    age = tf.feature_column.numeric_column('age')
    education_num = tf.feature_column.numeric_column('education_num')
    capital_gain = tf.feature_column.numeric_column('capital_gain')
    capital_loss = tf.feature_column.numeric_column('capital_loss')
    hours_per_week = tf.feature_column.numeric_column('hours_per_week')

    education = tf.feature_column.categorical_column_with_vocabulary_list(
        'education', ['HS-grad', '11th', 'Masters', '9th', 'Some-college', 'Assoc-acdm', 'Assoc-voc', '7th-8th', 'Doctorate', 'Prof-school', '5th-6th', \
                      '10th', '1st-4th', 'Preschool', '12th'])

    marital_status = tf.feture_column.categorical_column_with_vocabulary_list(
        'marital_status', ['Married-civ-spouse', 'Divorced', 'Married-spouse-absent', 'Never-married', 'Separated', 'Married-AF-spouse', 'Windowed'])

    relationship = tf.feature_column.categorical_column_with_vocabulary_list('relationship', ['Husband', 'Not-in-family', 'Wife', 'Own-child', 'Unmarried', 'Other-relative'])

    workclass = tf.feature_column.categorical_column_with_vocabulary_list('workclass', ['Self-emp-not-inc', 'Private', 'State-gov', 'Federal-gov',  \
                'Local-gov', '?', 'Self-emp-inc', 'Wighout-pay', 'Never-worked'])

    #将职业通过hash算法，hash成1000个类别
    occupation = tf.feature_column.categorical_column_weith_hash_bucket('occupation', hash_bucket_size=1000)

    #将连续值特征转为离散值特征
    age_buckets = tf.feature_column.bucketized_column(age, boundaries=[18,25,30,35,40,45,50,55,60,65])

    #定义基础特征列
    base_columns = [education, marital_status, relationship, workclass, occupation, age_bukets]

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

class WideDeepArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(WideDeepArgParser, self).__init__(parents=[parsers.BaseParser()])
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
            train_epochs=5,
            batch_size=40
        )



def train_main(argv):
    parser = WideDeepArgParser()
    flags = parser.parse_args(args(args=argv[1:])
    print "flags=", flags

    shutil.rmtree(flags.model_dir, ignore_errors=True)
    model f= build_estimator(flags.model_dir, flags.model_type)

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


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.ERROR)
    train_main(argv=sys.argv)
    pre_main(argv=sys.argv)



