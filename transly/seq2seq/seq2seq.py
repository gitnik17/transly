import os

from transly.seq2seq.config import SConfig
from transly.seq2seq.version1 import Seq2Seq

"""
word boundary example
"""

filepath = os.path.dirname(os.path.abspath(__file__))


def train(
    model_path="./trained_model/",
    model_file_name="model.h5",
    training_data_path="./train.csv",
):
    """
    trains and saves a word boundary model
    """
    config = SConfig(training_data_path=training_data_path)
    s2s = Seq2Seq(config)
    s2s.fit()
    s2s.save_model(path_to_model=model_path, model_file_name=model_file_name)


def load_model(
    model_path=filepath + "/trained_models/word_boundary/", model_file_name="model.h5"
):
    """
    loads a pre-trained word boundary model
    :return: Seq2Seq object
    """
    config = SConfig(configuration_file=model_path + "config.pkl")
    s2s = Seq2Seq(config)
    s2s.load_model(path_to_model=model_path, model_file_name=model_file_name)
    return s2s


if __name__ == "__main__":
    s = load_model()
    s.infer("CRO CIN")
