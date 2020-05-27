import pickle
import os
from keras.models import load_model
import glob
from abc import abstractmethod


class KModel:
    def __init__(self):
        self.config = None
        self.model = None

    @abstractmethod
    def fit(self):
        pass

    def save_model(
        self, path_to_model="/transly/trained_models/model/", model_file_name="model.h5"
    ):
        """
        Saves trained model to directory
        :param path_to_model: path to save model, defaults to './trained_models/'
        :type path_to_model: str, optional
        :param model_file_name: model file name, defaults to 'word_boundary.h5'
        :type model_file_name: str, optional
        """
        if not os.path.exists(path_to_model):
            os.makedirs(path_to_model)

        self.model.save(path_to_model + model_file_name)

        with open(path_to_model + "/config.pkl", "wb") as f:
            pickle.dump(self.config, f, protocol=2)

    def load_model(
        self,
        path_to_model="transly/base/trained_models/model/",
        model_file_name="model.h5",
    ):
        """
        Loads trained model
        :param path_to_model: if it is 'word_boundary' then loads one of default models, else loads model from given path
        :type path_to_model: str
        :param model_file_name: model file name
        :type model_file_name: str
        """

        # if path in ['word_boundary', 'spell_correction', 'ner']:
        #     path_to_model = [f for f in glob.glob(os.path.dirname(os.path.abspath(__file__)) + '/trained_models/' +\
        #                                           path + "**/*.h5", recursive=True)][0]
        # else:
        #     path_to_model = path

        self.model = load_model(path_to_model + model_file_name)

    @abstractmethod
    def infer(self, text):
        pass
