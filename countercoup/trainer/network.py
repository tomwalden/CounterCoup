from countercoup.shared.infoset import Infoset
from countercoup.trainer.memory import Memory
from keras.models import Model
from keras.layers import Dense, LSTM, Concatenate, Input


class Network:
    """Base class for the neural networks used in Deep CFR"""

    outputs = None
    final_activation = 'relu'
    model = None

    def __init__(self):
        self.__define_structure()

    def get_output(self, infoset: Infoset, filt: iter = None) -> dict:
        """
        Return the predicted output from the neural network
        :param infoset: the Infoset object that forms the input
        :param filt: the outputs that we want to potentially restrict on
        :return: a dict of possible outputs and output values
        """

        result = self.model.predict([infoset.fixed_vector] + infoset.history_vectors)
        output = {}

        for num, action in enumerate(self.outputs):
            if filt is None or action in filt:
                output[action] = result[0][num]

        return output

    def train(self, memory: Memory, epochs: int = 10, validation_split: float = 0.1):
        self.model.train(x=memory.data, epochs=epochs, validation_split=validation_split)

    def __define_structure(self):
        """
        Define the basic structure of the NN
        """

        fixed_input = Input(shape=(39,))
        history_curr_play_input = Input(shape=(None, 12))
        history_play_1_input = Input(shape=(None, 12))
        history_play_2_input = Input(shape=(None, 12))
        history_play_3_input = Input(shape=(None, 12))

        hist_lstm_curr = LSTM(10)(history_curr_play_input)
        hist_lstm_play_1 = LSTM(10)(history_play_1_input)
        hist_lstm_play_2 = LSTM(10)(history_play_2_input)
        hist_lstm_play_3 = LSTM(10)(history_play_3_input)

        concat = Concatenate(axis=1)(
            [fixed_input, hist_lstm_curr, hist_lstm_play_1, hist_lstm_play_2, hist_lstm_play_3])

        dense_1 = Dense(100, activation='relu')(concat)
        dense_2 = Dense(100, activation='relu')(dense_1)
        dense_3 = Dense(100, activation='relu')(dense_2)
        dense_4 = Dense(100, activation='relu')(dense_3)
        dense_5 = Dense(100, activation='relu')(dense_4)
        dense_6 = Dense(100, activation='relu')(dense_5)

        output = Dense(len(self.outputs), activation=self.final_activation)(dense_6)

        self.model = Model(
            [fixed_input, history_curr_play_input, history_play_1_input, history_play_2_input, history_play_3_input],
            output)
        self.model.compile(loss='categorical_crossentropy', optimizer='adam')

    @classmethod
    def create_train_data(cls, iput: Infoset, output: dict, iteration: int) -> tuple:
        """
        Turn the output dict into a tuple that can go into a NN
        :param iput: the Infoset input
        :param output: the dict output
        :param iteration: the iteration. Used to weigh when training.
        :return: a list output
        """

        new_input = iput.fixed_vector + iput.history_vectors

        new_output = []

        for x in cls.outputs:
            if x in output:
                new_output.append(output[x])
            else:
                new_output.append(0)

        return new_input, new_output, iteration