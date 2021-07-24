from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.structure import Structure
from zipfile import ZipFile
from os import remove


class NetworkGroup:
    """
    Combined group of networks used to define CounterCoup
    """

    def __init__(self, file_path: str = None, structure: Structure = None):

        if file_path is not None:
            self.load(file_path)
        else:
            self.action = ActionNet(structure=structure)
            self.block = BlockCounteractNet(structure=structure)
            self.counteract = BlockCounteractNet(structure=structure)
            self.lose = LoseNet(structure=structure)

    def train_networks(self, action_mem, block_mem, counteract_mem, lose_mem):
        """
        Train the strategy networks, at the end
        """
        self.action.train(action_mem)
        self.block.train(block_mem)
        self.counteract.train(counteract_mem)
        self.lose.train(lose_mem)

    def load(self, file_path: str):

        with ZipFile(file_path, 'r') as zf:
            zf.extract('action.h5')
            self.action = ActionNet('action.h5')
            remove('action.h5')

            zf.extract('block.h5')
            self.block = BlockCounteractNet('block.h5')
            remove('block.h5')

            zf.extract('counteract.h5')
            self.counteract = BlockCounteractNet('counteract.h5')
            remove('counteract.h5')

            zf.extract('lose.h5')
            self.lose = LoseNet('lose.h5')
            remove('lose.h5')

    def save(self, file_path: str):

        with ZipFile(file_path, 'a') as zf:
            self.action.save('action.h5')
            zf.write('action.h5')
            remove('action.h5')

            self.block.save('block.h5')
            zf.write('block.h5')
            remove('block.h5')

            self.counteract.save('counteract.h5')
            zf.write('counteract.h5')
            remove('counteract.h5')

            self.lose.save('lose.h5')
            zf.write('lose.h5')
            remove('lose.h5')









