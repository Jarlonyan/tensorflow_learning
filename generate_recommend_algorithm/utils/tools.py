#coding=utf-8

import argparse

class BaseParser(argparse.ArgumentParser):
    def __init__(self, add_help=False, data_dir=True, model_dir=True,
                 epochs=True, ecochs_between_evals=True,
                 stop_threshold=True, batch_size=10, multi_gpu=True,
                 hooks=True, export_dir=True):
        super(BaseParser, self).__init__(add_help=add_help)
        if data_dir:
            self.add_argument(
                "--data_dir", "-dd", default="/tmp",
                help="[default:%(default)s] The location of the input data.",
                metavar="<DD>",
            )
        if model_dir:
            self.add_argument(
                "--model_dir", "-md", default="/tmp",
                help="[default:%(default)s] The location of the model checkpoint files.",
                metavar="<MD>"
            )

class DNNArgParser(argparse.ArgumentParser):
    def __init__(self):
        super(DNNArgParser, self).__init__(parents=[BaseParser()])
        self.add_argument(
            '--model_type', '-mt', type=str, default='wide_deep',
            choices=['wide', 'deep', 'wide_deep'],
            help='[default %(default)s] valid model types: wide, deep, wide_deep'
        )
        self.set_defaults(
            data_dir = './data',
            epochs = 5,
            epochs_per_eval = 2,
            batch_size = 40,
            train_file = './data/adult.data',
            test_file  = './data/adult.test',
            model_dir = './model/wide_deep',
        )

