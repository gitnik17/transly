import os
from transly.seq2seq.config import SConfig
from transly.seq2seq.version0 import Seq2Seq

"""
pronunciation example
"""

TRAIN = False
filepath = os.path.dirname(os.path.abspath(__file__))


def train(
    model_path="./trained_model/",
    model_file_name="model.h5",
    training_data_path="./train.csv",
    input_mode="character_level",
    output_mode="word_level",
):
    """
    trains and saves a word boundary model
    """
    config = SConfig(
        training_data_path=training_data_path,
        input_mode=input_mode,
        output_mode=output_mode,
    )
    s2s = Seq2Seq(config)
    s2s.fit()
    s2s.save_model(path_to_model=model_path, model_file_name=model_file_name)


def load_model(
    model_path=filepath + "/trained_models/cmu/", model_file_name="model.h5"
):
    """
    loads a pre-trained word boundary model
    :return: Seq2Seq object
    """
    model_path = (
        filepath + "/trained_models/{}/".format(model_path)
        if model_path in ["cmu"]
        else model_path
    )
    config = SConfig(configuration_file=model_path + "config.pkl")
    s2s = Seq2Seq(config)
    s2s.load_model(path_to_model=model_path, model_file_name=model_file_name)
    return s2s


if __name__ == "__main__":
    if TRAIN:
        train(
            model_path="./",
            model_file_name="model.h5",
            training_data_path="cmu.csv",
            input_mode="character_level",
            output_mode="word_level",
        )

    QUERY = "MAKAAN"
    a = load_model(model_path="cmu", model_file_name="model.h5")
    a.infer(QUERY)
