from countercoup.shared.memory import Memory
from keras.utils.data_utils import Sequence
from random import shuffle


class BatchMemory(Sequence):
    """
    Sequence based object that can batch up the data for each epoch when training
    """

    def __init__(self, memory: Memory, batch_size: int):
        self.memory = memory
        self.batch_size = batch_size

        # Give the data a good shufflin'
        shuffle(self.memory.data)

        # Set the batch_begin at 0
        self.batch_begin = 0

    def __len__(self):
        return self.batch_size

    def __getitem__(self, idx):
        if idx > self.batch_size - 1:
            raise IndexError("index larger than batch size")

        return self.memory[(self.batch_begin + idx) % len(self.memory)]

    def on_epoch_end(self):
        self.batch_begin = (self.batch_begin + self.batch_size) % len(self.memory)
