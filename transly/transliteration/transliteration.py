import os
from transly.seq2seq.config import SConfig
from transly.seq2seq.version0 import Seq2Seq

"""
transliteration example
"""

TRAIN = False
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
    model_path=filepath + "/trained_models/hi2en/", model_file_name="model.h5"
):
    """
    loads a pre-trained word boundary model
    :return: Seq2Seq object
    """
    model_path = (
        filepath + "/trained_models/{}/".format(model_path)
        if model_path in ["en2hi", "hi2en"]
        else model_path
    )
    config = SConfig(configuration_file=model_path + "config.pkl")
    s2s = Seq2Seq(config)
    s2s.load_model(path_to_model=model_path, model_file_name=model_file_name)
    return s2s


if __name__ == "__main__":
    if TRAIN:
        train(
            model_path="./", model_file_name="model.h5", training_data_path="hi2en.csv"
        )

    QUERY = "नहीं"
    a = load_model(model_path="hi2en", model_file_name="model.h5")
    a.infer(QUERY)
