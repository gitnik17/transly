import pickle

import pandas as pd

from transly.base.config import Config


class SConfig(Config):
    """
    Configuration for encoder decoder model
    """

    def __init__(
        self,
        configuration_file=None,
        training_data_path="khoj/seq2seq/train.data.csv",
        testing_data_path="data/test.data.csv",
        static_config=None,
        input_mode="character_level",
        output_mode="character_level",
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
        Config.__init__(
            self,
            configuration_file,
            training_data_path,
            testing_data_path,
            static_config,
        )
        self.input_mode = input_mode
        self.output_mode = output_mode

    def get_config(self):
        """
        Computes the entire configuration, including that at the time of initialisation
        :return: full configuration
        :rtype: dict
        """
        if self.configuration_file:
            self.config = pickle.load(open(self.configuration_file, "rb"))
            return self.config

        # derived configuration
        print("fetching training file")
        train_data = pd.read_csv(self.training_data_path)

        train_data = train_data.apply(
            lambda x: x.astype(str)
            .str.upper()
            .str.replace(r"[^A-Z0-9.%/\s]", " ")
            .str.replace(r" +", " ")
        )

        train_data = train_data[
            [
                True if max([len(str(v[0])), len(str(v[1]))]) < 20 else False
                for v in train_data.values
            ]
        ]
        train_input, train_output = train_data.values[:, 0], train_data.values[:, 1]

        self.config["train_input"], self.config["train_output"] = (
            train_input,
            train_output,
        )
        self.config["max_length_input"], self.config["max_length_output"] = (
            max([len(str(v)) for v in train_input]),
            max([len(str(v)) for v in train_output]),
        )

        if self.input_mode == "character_level":
            (
                self.config["input_char2ix"],
                self.config["input_ix2char"],
                self.config["input_dict_len"],
            ) = self.__char2index2char__(train_input)

        elif self.input_mode == "word_level":
            (
                self.config["input_char2ix"],
                self.config["input_ix2char"],
                self.config["input_dict_len"],
            ) = self.__word2index2word__(train_input)

        else:
            raise Exception(
                "input mode needs to be either 'character_level' or 'word_level'. Received {} instead.".format(
                    self.input_mode
                )
            )

        if self.output_mode == "character_level":
            (
                self.config["output_char2ix"],
                self.config["output_ix2char"],
                self.config["output_dict_len"],
            ) = self.__char2index2char__(train_output)

        elif self.output_mode == "word_level":
            (
                self.config["output_char2ix"],
                self.config["output_ix2char"],
                self.config["output_dict_len"],
            ) = self.__word2index2word__(train_output)

        else:
            raise Exception(
                "output mode needs to be either 'character_level' or 'word_level'. Received {} instead.".format(
                    self.output_mode
                )
            )

        self.config["input_mode"], self.config["output_mode"] = (
            self.input_mode,
            self.output_mode,
        )

        return self.config

    def __char2index2char__(self, words):
        """
        Computes character to indices dict (encoding) as well as indices to character dict (decoding)
        :param words: list of words
        :type words: list
        :return: character encoding dict, decoding dict, length of dict
        :rtype: dict, dict, int
        """
        char2ix = {
            w: i + 2 for i, w in enumerate(set([wc for w in words for wc in str(w)]))
        }
        char2ix["PAD"], char2ix["GO"] = (
            self.config["PAD_INDEX"],
            self.config["GO_INDEX"],
        )
        ix2char = {v: k for k, v in char2ix.items()}
        return char2ix, ix2char, len(ix2char)

    def __word2index2word__(self, sentences):
        """
        Computes character to indices dict (encoding) as well as indices to character dict (decoding)
        :param sentences: list of sentences
        :type sentences: list
        :return: word encoding dict, decoding dict, length of dict
        :rtype: dict, dict, int
        """
        word2ix = {
            w: i + 2
            for i, w in enumerate(set([w for s in sentences for w in str(s).split()]))
        }
        word2ix["PAD"], word2ix["GO"] = (
            self.config["PAD_INDEX"],
            self.config["GO_INDEX"],
        )
        ix2word = {v: k for k, v in word2ix.items()}
        return word2ix, ix2word, len(ix2word)
