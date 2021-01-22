#coding=utf-8
#https://github.com/jxyyjm/tensorflow_test/blob/master/src/deep_and_cross.py
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import tensorflow as tf
import argparse
import sys
import shutil
import argparse
import sys
reload(sys)
sys.setdefaultencoding('utf-8')

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='model_dir', help='Base directory for the model.')
parser.add_argument('--model_type', type=str, default='deep_cross', help="wide, deep, wide_deep, deep_cross.")
parser.add_argument('--train_epochs', type=int, default=40, help='Number of training epochs.')
parser.add_argument('--epochs_per_eval', type=int, default=2, help='The number of training epochs to run between evaluations.')
parser.add_argument('--batch_size', type=int, default=40, help='Number of examples per batch.')
parser.add_argument('--train_data', type=str, default='data/adult.data', help='Path to the training data.')
parser.add_argument('--test_data', type=str, default='data/adult.test', help='Path to the test data.')

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

#x_l+1 = x_0*x_l^T*w_l + b_l + w_l
def cross_op(x0, x, w, b):
    x0 = tf.expand_dims(x0, axis=2)
    x = tf.expand_dims(x, axis=2)
    multiple = w.get_shape().as_list()[0]

    x0_broad_horizon = tf.tile(x0, [1, 1, multiple])
    x_broad_vertical = tf.transpose(tf.tile(x, [1,1,multiple]), [0,2,1])
    w_broad_horizon = tf.tile(w, [1,multiple])
    mid_res = tf.multiply(tf.multiply(x0_broad_horizon, x_broad_vertical), w)
    res = tf.reduce_sum(mid_res, axis=2)
    res = res + tf.transpose(b)
    return res

def cross_op2(x0, x, w, b):
    batch_num = x0.get_shape().as_list()[0]
    res = []
    for i in range(batch_num):
        dd = tf.matmul(x0[i,:,:], tf.transpose(x[i,:,:]))
        dc = tf.matmul(dd, w) + b
        res[i] = dc
    return res + x0

def build_deep_cross_model(features, labels, mode, params):
    columns = build_columns()
    input_layer = tf.feature_column.input_layer(features=features, feature_columns=columns)
    print ('features.shape=', features)

    column_num = input_layer.get_shape().as_list()[1]
    print ('column_num, before cross_variable_create:', column_num)

    c_w_1, c_b_1 = cross_variable_create(column_num)
    c_w_2, c_b_2 = cross_variable_create(column_num)
    c_layer_1 = cross_op(input_layer, input_layer, c_w_1, c_b_1) + input_layer
    c_layer_2 = cross_op(input_layer, c_layer_1, c_w_2, c_b_2) + c_layer_1
    c_layer_5 = c_layer_2

    h_layer_1 = tf.layers.dense(inputs=input_layer, units=50, activation=tf.nn.relu, use_bias=True)
    bn_layer_1 = tf.layers.batch_normalization(inputs=h_layer_1, axis=-1, \
                                            momentum=0.99, epsilon=0.001, \
                                            center=True, scale=True)

    h_layer_2 = tf.layers.dense(inputs=bn_layer_1, units=40, activation=tf.nn.relu, use_bias=True)

    m_layer = tf.concat([h_layer_2, c_layer_5], 1)
    o_layer = tf.layers.dense(inputs=m_layer, units=1, activation=None, use_bias=True)
    o_prob = tf.nn.sigmoid(o_layer)

    predictions = tf.cast((o_prob>0.5), tf.float32)
    labels = tf.cast(labels, tf.float32)

    prediction_dict = {'income_bracket': predictions}
    loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=o_layer))
    accuracy = tf.metrics.accuracy(labels, predictions)
    tf.summary.scalar('accuracy', accuracy[1])

    optimizer = tf.train.AdamOptimizer(learning_rate=0.0004, beta1=0.9, beta2=0.999)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    if mode == tf.estimator.ModeKeys.TRAIN:
        return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)
    elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops={'accuracy':accuracy})
    elif mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

def build_estimator(model_dir, model_type):
    run_config = tf.estimator.RunConfig().replace(session_config=tf.ConfigProto(device_count={'GPU': 0}))
    if model_type == 'deep_cross':
        return tf.estimator.Estimator(model_fn = build_deep_cross_model, model_dir=model_dir, config=run_config)
    else: 
        print ('error')

def input_fn(data_file, num_epochs, shuffle, batch_size):
    assert tf.gfile.Exists(data_file), ('no file named:'+str(data_file))

    def process_list_column(list_column):
        sparse_strings = tf.string_split(list_column, delimiter="|")
        return sparse_strings.values
    def parse_csv(value):
        columns = tf.decode_csv(value, record_defaults=_CSV_COLUMN_DEFAULTS)
        features = dict(zip(_CSV_COLUMNS, columns))
        features['workclass'] = process_list_column([features['workclass']])
        labels = tf.equal(features.pop('income_bracket'), '>50K')
        labels = tf.reshape(labels, [-1])
        return features, labels

    dataset = tf.contrib.data.TextLineDataset(data_file)
    if shuffle: 
        dataset = dataset.shuffle(buffer_size=5000)
    dataset = dataset.map(parse_csv, num_threads=5)
    dataset = dataset.repeat(num_epochs)
    dataset = dataset.batch(batch_size)
    iterator  = dataset.make_one_shot_iterator()
    features, labels = iterator.get_next()
    return features, labels

def main():
    # Clean up the model directory if present
    shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)

    # Train and evaluate the model every `FLAGS.epochs_per_eval` epochs.
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        model.train(input_fn=lambda: input_fn( FLAGS.train_data, FLAGS.epochs_per_eval, True, FLAGS.batch_size))

        results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, False, FLAGS.batch_size))

        # Display evaluation metrics
        print('Results at epoch', (n + 1) * FLAGS.epochs_per_eval)
        print('-' * 60)

        for key in sorted(results):
            print('%s: %s' % (key, results[key]))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    #tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
    main()

