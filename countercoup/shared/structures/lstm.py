from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Concatenate, Input
from countercoup.shared.structure import Structure


class LSTMNet(Structure):

    @staticmethod
    def define_structure(outputs: []) -> Model:
        fixed_input = Input(shape=(46,))
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

        dense_1 = Dense(100, activation='linear')(concat)
        dense_2 = Dense(100, activation='linear')(dense_1)
        dense_3 = Dense(100, activation='linear')(dense_2)
        dense_4 = Dense(100, activation='linear')(dense_3)
        dense_5 = Dense(100, activation='linear')(dense_4)
        dense_6 = Dense(100, activation='linear')(dense_5)

        output = Dense(len(outputs), activation='linear')(dense_6)

        model = Model(
            [fixed_input, history_curr_play_input, history_play_1_input, history_play_2_input, history_play_3_input],
            output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model
