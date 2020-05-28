import numpy as np
from keras import optimizers
from keras.layers import Activation, dot, concatenate
from keras.layers import Bidirectional
from keras.layers import Input, Embedding, LSTM, TimeDistributed, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from transly.base.model import KModel
from transly.seq2seq.config import SConfig


class Seq2Seq(KModel):
    """
    This model is a sequence to sequence / encoder-decoder model
    This model does a prediction at all time steps
    It converges fast but gives 4-5 ms slower inference than the one which predicts only at the last time step (obviously)
    """

    def __init__(self, configuration=None):
        """
        Loads configuration and initialises Seq2Seq model
        :param configuration: configuration class object, default to None
        :type configuration: object, optional
        """
        KModel.__init__(self)

        if configuration is None:
            configuration = SConfig()
        print("fetching configuration")
        self.config = configuration.get_config()

        self.go_index = self.config["GO_INDEX"]
        self.pad_index = self.config["PAD_INDEX"]
        self.input_char2ix = self.config["input_char2ix"]
        self.output_ix2char = self.config["output_ix2char"]
        self.max_length_input = self.config["max_length_input"]
        self.max_length_output = self.config["max_length_output"]

        input_dict_len = self.config["input_dict_len"]
        output_dict_len = self.config["output_dict_len"]

        encoder_input = Input(shape=(self.max_length_input,))
        decoder_input = Input(shape=(self.max_length_output,))

        lstm_units = self.config["number_of_units"]
        encoder = Embedding(
            input_dict_len,
            lstm_units,
            input_length=self.max_length_input,
            mask_zero=True,
        )(encoder_input)

        encoder = Bidirectional(
            LSTM(lstm_units, return_state=True, return_sequences=True, unroll=True),
            merge_mode="concat",
        )(encoder)

        (
            encoder_output,
            forward_hidden_state,
            forward_cell_state,
            backward_hidden_state,
            backward_cell_state,
        ) = encoder
        encoder_hidden_state = concatenate(
            [forward_hidden_state, backward_hidden_state]
        )
        encoder_cell_state = concatenate([forward_cell_state, backward_cell_state])

        decoder = Embedding(
            output_dict_len,
            lstm_units * 2,
            input_length=self.max_length_output,
            mask_zero=True,
        )(decoder_input)
        decoder = LSTM(lstm_units * 2, return_sequences=True, unroll=True)(
            decoder, initial_state=[encoder_hidden_state, encoder_cell_state]
        )

        attention = dot([decoder, encoder_output], axes=[2, 2])
        attention = Activation("softmax", name="attention")(attention)

        context = dot([attention, encoder_output], axes=[2, 1])
        decoder_combined_context = concatenate([context, decoder])

        output = TimeDistributed(Dense(lstm_units, activation="tanh"))(
            decoder_combined_context
        )
        output = TimeDistributed(Dense(output_dict_len, activation="softmax"))(output)

        self.model = Model(inputs=[encoder_input, decoder_input], outputs=[output])

    def encode(self, char2ix, words, vector_size, mode="character_level"):
        """
        Encode/index a vector
        :param mode: mode in which you want to deal in. characters (word_embedding) or words (sentence_embedding)
        :type mode: str
        :param char2ix: character to index mapping
        :type char2ix: dict
        :param words: words to be encoded/indexed
        :type words: list
        :param vector_size: size of the indexed vector
        :type vector_size: int
        :return: encoded vector padded to vector_size
        :rtype: numpy array
        """
        sequence = (
            [[char2ix[c] for c in str(w)] for w in words]
            if mode == "character_level"
            else [[char2ix[c] for c in str(w).split()] for w in words]
        )
        return pad_sequences(
            maxlen=vector_size,
            sequences=sequence,
            value=self.config["PAD_INDEX"],
            padding="post",
            truncating="post",
        )

    def decode(self, ix2char, vector, separator=""):
        """
        Decode an encoded/indexed vector
        :param separator: separator while decoding
        :param ix2char: index to character mapping
        :type ix2char: dict
        :param vector: encoded/indexed vector to be decoded
        :type vector: numpy array or list
        :return: decoded vector: decoded string
        :rtype decoded vector: str
        """
        return separator.join(
            [ix2char[value] for value in vector if value != self.config["PAD_INDEX"]]
        )

    def fit(self, learning_rate=0.001):
        print("encoding training input")
        encoded_input = input_encoder = self.encode(
            self.config["input_char2ix"],
            self.config["train_input"],
            vector_size=self.config["max_length_input"],
            mode=self.config["input_mode"],
        )

        print("encoding training output")
        encoded_output = self.encode(
            self.config["output_char2ix"],
            self.config["train_output"],
            vector_size=self.config["max_length_output"],
            mode=self.config["output_mode"],
        )

        output_decoder = np.eye(self.config["output_dict_len"])[
            encoded_output.astype("int")
        ]
        input_decoder = np.array(
            [np.insert(eo[:-1], 0, self.config["GO_INDEX"]) for eo in encoded_output]
        )

        optimizer = optimizers.adam(lr=learning_rate)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )

        self.model.fit(
            x=[input_encoder, input_decoder],
            y=[output_decoder],
            validation_split=0.1,
            batch_size=self.config["batch_size"],
            epochs=1000,
        )

    def infer(self, text, separator=""):
        """
        Inference of new text
        :param separator: separator for the decoded output
        :param text: strings as input to model
        :type text: str
        :return: predicted output
        :rtype: str
        """
        encoder_input = [
            [self.input_char2ix[c] for c in text]
            + [self.pad_index] * (self.max_length_input - len(text))
        ]
        decoder_input = [
            [self.go_index] + [self.pad_index] * (self.max_length_output - 1)
        ]

        for i in range(2, self.max_length_output):
            output = self.model.predict([encoder_input, decoder_input])
            decoder_input[0][i] = np.argmax(output[0][i], axis=-1)

            if decoder_input[0][i] == self.pad_index:
                break
        return self.decode(
            ix2char=self.output_ix2char,
            vector=decoder_input[0][1:],
            separator=separator,
        )
