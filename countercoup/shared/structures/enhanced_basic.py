from keras.models import Model, load_model
from keras.layers import Dense, LSTM, Concatenate, Input, Dropout
from countercoup.shared.structure import Structure


class EnhancedBasic(Structure):
    """Basic enhanced structure, same as Enhanced structure but without LSTM cells"""

    @staticmethod
    def define_structure(outputs: []) -> Model:
        fixed_input = Input(shape=(46,))
        history_curr_play_input = Input(shape=(None, 12))
        history_play_1_input = Input(shape=(None, 12))
        history_play_2_input = Input(shape=(None, 12))
        history_play_3_input = Input(shape=(None, 12))

        dense_1 = Dense(100, activation='linear')(fixed_input)
        dense_2 = Dense(100, activation='linear')(dense_1)
        dense_3 = Dense(100, activation='linear')(dense_2)
        dense_4 = Dense(100, activation='linear')(dense_3)
        dense_5 = Dense(100, activation='linear')(dense_4)
        dropout_1 = Dropout(0.1)(dense_5)
        dense_6 = Dense(100, activation='linear')(dropout_1)
        dense_7 = Dense(100, activation='linear')(dense_6)
        dense_8 = Dense(100, activation='linear')(dense_7)
        dense_9 = Dense(100, activation='linear')(dense_8)
        dense_10 = Dense(100, activation='linear')(dense_9)

        output = Dense(len(outputs), activation='linear')(dense_10)

        model = Model(
            [fixed_input, history_curr_play_input, history_play_1_input, history_play_2_input, history_play_3_input],
            output)
        model.compile(loss='mean_squared_error', optimizer='adam')

        return model
