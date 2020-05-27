import glob
import os
from abc import abstractmethod


class Config:
    """
    Configuration base class
    """

    def __init__(
        self,
        configuration_file=None,
        training_data_path="transly/seq2seq/train.data.csv",
        testing_data_path="transly/data/test.data.csv",
        static_config=None,
    ):
        """
        Initialise configuration
        :param configuration_file: path to configuration file of a pre-trained model, defaults to None
        :type configuration_file: str, optional
        :param training_data_path: path to training data, defaults to 'seq2seq/train.data.csv'
        :type training_data_path: str, optional
        :param testing_data_path: path to testing data, defaults to 'data/test.data.csv'
        :type testing_data_path: str, optional
        :param static_config: defaults to {'number_of_units': 64, 'batch_size': 1500, 'epochs': 100, 'PAD_INDEX': 0, 'GO_INDEX': 1}
        :type static_config: dict, optional
        """
        if static_config is None:
            static_config = {
                "number_of_units": 64,
                "batch_size": 1500,
                "epochs": 100,
                "PAD_INDEX": 0,
                "GO_INDEX": 1,
            }

        # existing configuration
        if configuration_file in ["word_boundary", "spell_correction", "ner"]:
            self.configuration_file = [
                f
                for f in glob.glob(
                    os.path.dirname(os.path.abspath(__file__))
                    + "/trained_models/"
                    + configuration_file
                    + "**/*.pkl",
                    recursive=True,
                )
            ][0]
        else:
            self.configuration_file = configuration_file

        # static configuration
        self.config = static_config

        # environment configuration
        self.training_data_path = training_data_path
        self.testing_data_path = testing_data_path

    @abstractmethod
    def get_config(self):
        """
        Computes the entire configuration, including that at the time of initialisation
        :return: full configuration
        :rtype: dict
        """
        pass
