#coding=utf-8

import argparse
import tensorflow as tf

class BaseParser(argparse.ArgumentParser):
    def __init__(self, add_help=False, data_dir=True, model_dir=True,
                 train_epochs=True, ecochs_between_evals=True,
                 stop_threshold=True, batch_size=True, multi_gpu=True,
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



