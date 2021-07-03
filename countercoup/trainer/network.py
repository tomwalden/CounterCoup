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

    def get_output(self, infoset: Infoset, filt: [] = None) -> dict:
        input = [infoset.fixed_vector] + infoset.history_vectors
        result = self.model.predict(input)
        output = {}

        for num, action in enumerate(self.outputs):
            if filt is None or action in filt:
                output[action] = result[num]

        return output

    def train(self, memory: Memory):
        pass

    def __define_structure(self):
        """
        Define the basic structure of the NN
        """

        fixed_input = Input(shape=(39,))
        history_curr_play_input = Input(shape=(10, 12))
        history_play_1_input = Input(shape=(10, 12))
        history_play_2_input = Input(shape=(10, 12))
        history_play_3_input = Input(shape=(10, 12))

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

    @classmethod
    def create_output(cls, output: dict) -> list:
        """
        Turn the output dict into a list that can go into a NN
        :param output: the dict outpt
        :return: a list output
        """

        new_output = []

        for x in cls.outputs:
            if x in output:
                new_output.append(output[x])
            else:
                new_output.append(0)

        return new_output
