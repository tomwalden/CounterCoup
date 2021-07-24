from countercoup.shared.infoset import Infoset
from countercoup.shared.memory import Memory
from countercoup.shared.batch_memory import BatchMemory
from countercoup.shared.structure import Structure
from countercoup.shared.structures.lstm import LSTMNet
from keras.models import Model, load_model
from numpy import array


class Network:
    """Base class for the neural networks used in Deep CFR"""

    outputs = None
    final_activation = 'linear'
    model = None

    def __init__(self, file_path: str = None, structure: Structure = None):

        if file_path is not None:
            self.load(file_path)
        elif structure is None:
            self.model = LSTMNet.define_structure(self.outputs)
        else:
            self.model = structure.define_structure(self.outputs)

    def get_output(self, infoset: Infoset, filt: [] = None) -> dict:
        """
        Return the predicted output from the neural network
        :param infoset: the Infoset object that forms the input
        :param filt: the outputs that we want to potentially restrict on
        :return: a dict of possible outputs and output values
        """

        result = self.model([infoset.fixed_vector] + infoset.history_vectors).numpy()
        output = {}

        for num, action in enumerate(self.outputs):
            if filt is None or action in filt:
                output[action] = result[0][num]

        return output

    def train(self, memory: Memory, epochs: int = 10, batch_size: int = None):
        """
        Train the network
        :param memory: a Memory of data to train on
        :param epochs: number of epochs to train on
        :param batch_size: size of each epoch
        """

        if batch_size is None:
            batch_size = round(len(memory) / epochs)

        bm = BatchMemory(memory, batch_size)
        self.model.fit(x=bm, epochs=epochs)

    def save(self, file_path: str):
        """
        Save the network to disk
        :param file_path: the path to save the net to
        """
        self.model.save(file_path)

    def load(self, file_path: str):
        """
        Load a network
        :param file_path: the path to load the net from
        """
        self.model = load_model(file_path)

    @classmethod
    def create_train_data(cls, iput: Infoset, output: dict, iteration: int) -> tuple:
        """
        Turn the output dict into a tuple that can go into a NN
        :param iput: the Infoset input
        :param output: the dict output
        :param iteration: the iteration. Used to weigh when training.
        :return: a list output
        """

        new_input = [iput.fixed_vector] + iput.history_vectors

        new_output = []

        for x in cls.outputs:
            if x in output:
                new_output.append(output[x])
            else:
                new_output.append(0)

        return new_input, array([new_output]), array([[iteration]])
