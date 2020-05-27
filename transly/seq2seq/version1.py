import numpy as np
from keras import optimizers
from keras.layers import Activation, concatenate, multiply
from keras.layers import Bidirectional
from keras.layers import Input, Embedding, LSTM, Dense
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

from transly.base.model import KModel
from transly.seq2seq.config import SConfig


class Seq2Seq(KModel):
    """
    This model is a sequence to sequence / encoder-decoder model
    This model predicts only at the last time step
    It converges slow but obviously gives 4-5 ms faster inference than the one which predicts at all time steps
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

        encoder = Bidirectional(LSTM(lstm_units, return_state=True, unroll=True))(
            encoder
        )

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
        decoder = LSTM(lstm_units * 2, unroll=True)(
            decoder, initial_state=[encoder_hidden_state, encoder_cell_state]
        )

        attention = multiply([decoder, encoder_output])
        attention = Activation("softmax", name="attention")(attention)

        context = multiply([attention, encoder_output])
        decoder_combined_context = concatenate([context, decoder])

        output = Dense(lstm_units, activation="tanh")(decoder_combined_context)
        output = Dense(output_dict_len, activation="softmax")(output)

        self.model = Model(inputs=[encoder_input, decoder_input], outputs=[output])
        optimizer = optimizers.adam(lr=0.001)
        self.model.compile(
            optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"]
        )
        print(self.model.summary())

    def encode(self, char2ix, words, vector_size):
        """
        Encode/index a vector
        :param char2ix: character to index mapping
        :type char2ix: dict
        :param words: words to be encoded/indexed
        :type words: list
        :param vector_size: size of the indexed vector
        :type vector_size: int
        :return: encoded vector padded to vector_size
        :rtype: numpy array
        """
        return pad_sequences(
            maxlen=vector_size,
            sequences=[[char2ix[c] for c in str(w)] for w in words],
            value=self.config["PAD_INDEX"],
            padding="post",
            truncating="post",
        )

    def decode(self, ix2char, vector):
        """
        Decode an encoded/indexed vector
        :param ix2char: index to character mapping
        :type ix2char: dict
        :param vector: encoded/indexed vector to be decoded
        :type vector: numpy array or list
        :return: decoded vector: decoded string
        :rtype decoded vector: str
        """
        return "".join(
            [ix2char[value] for value in vector if value != self.config["PAD_INDEX"]]
        )

    def fit(self):
        print("encoding training input")

        z = np.array(
            [
                [str(v)[:i], str(u), str(v)[i]]
                if i < len(str(v))
                else [str(v)[:i], str(u), "PAD"]
                for u, v in zip(self.config["train_input"], self.config["train_output"])
                for i in range(1, len(str(v)) + 1)
            ]
        )
        input_decoder, input_encoder, output_decoder = z[:, 0], z[:, 1], z[:, 2]

        input_encoder = self.encode(
            self.config["input_char2ix"],
            input_encoder,
            vector_size=self.config["max_length_input"],
        )

        encoded_output = self.encode(
            self.config["output_char2ix"],
            input_decoder,
            vector_size=self.config["max_length_output"],
        )

        input_decoder = np.array(
            [np.insert(eo[:-1], 0, self.config["GO_INDEX"]) for eo in encoded_output]
        )

        output_decoder = np.eye(self.config["output_dict_len"])[
            np.array([self.config["output_char2ix"][i] for i in output_decoder]).astype(
                "int"
            )
        ]

        self.model.fit(
            x=[input_encoder, input_decoder],
            y=[output_decoder],
            validation_split=0.1,
            batch_size=self.config["batch_size"],
            epochs=1000,
        )

    def infer(self, text):
        """
        Inference of new text
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
            [self.go_index]
            + [encoder_input[0][0]]
            + [self.pad_index] * (self.max_length_output - 2)
        ]
        breaker, c, c_ = self.input_char2ix[text[-1]], text.count(text[-1]), 0

        for i in range(2, self.max_length_output):
            decoder_input[0][i] = self.model.predict(
                [encoder_input, decoder_input]
            ).argmax()
            if decoder_input[0][i] == breaker:
                c_ += 1
                if c_ == c:
                    break
        return self.decode(ix2char=self.output_ix2char, vector=decoder_input[0][1:])
