from countercoup.trainer.trainer_stats import TrainerStats
from countercoup.model.game import Game
from countercoup.shared.network import Network
from countercoup.shared.infoset import Infoset


class Traverser:
    """Base class for traversers"""

    def __init__(self, action_nets: [], block_nets: [], counteract_nets: [], lose_nets: [], iteration: int):
        self.action_nets = action_nets
        self.block_nets = block_nets
        self.counteract_nets = counteract_nets
        self.lose_nets = lose_nets

        self.iteration = iteration

        self.action_mem = [[] for _ in action_nets]
        self.block_mem = [[] for _ in block_nets]
        self.counteract_mem = [[] for _ in counteract_nets]
        self.lose_mem = [[] for _ in lose_nets]

        self.action_strategy_mem = []
        self.block_strategy_mem = []
        self.counteract_strategy_mem = []
        self.lose_strategy_mem = []

        self.stats = TrainerStats()

    def get_regret_strategy(self, network: Network, infoset: Infoset, filt: [] = None):
        """
        Get the strategy calculated from the advantage networks
        :param network: the network to calculate the advantages
        :param infoset: the infoset for the game state
        :param filt: the outputs that we're allowed to output
        :return: a dict of available actions and the strategy
        """

        # If we're on the first iteration, don't bother using the NNs. Speeds up this iteration, and
        # resolves issues where the networks don't zero correctly.
        if self.iteration == 1:
            output = {x: 0 for x in (filt if filt is not None else network.outputs)}
        else:
            output = network.get_output(infoset, filt)

        total = sum(filter(lambda x: x > 0, output.values()))

        if total == 0:
            return {x: 1 / len(output) for x in output}
        else:
            return {x: (output[x] if output[x] > 0 else 0) / total for x in output}

    def calculate_regrets(self, values: {}, strategy: {}, memory: [], infoset: Infoset, output_formatter):
        """
        Calculate the regret values (and insert them into memory)
        :param values: the advantage values
        :param strategy: the calculated strategy
        :param memory: the memory to insert the calculated regrets into
        :param infoset: the infoset for the game state
        :param output_formatter: a function that formats the regret data before being inserted into the memory
        :return: the total instr_regret
        """

        instr_regret = 0

        for x in values:
            instr_regret += strategy[x] * values[x]

        # Calculate the scale factor - for robust sampling, it is the inverse of the fraction of actions selected
        scale_factor = len(strategy) / len(values)

        # Scale the instantaneous regret by the scale factor
        instr_regret *= scale_factor

        new_regrets = {}
        for x in strategy:
            if x in values:
                new_regrets[x] = (values[x] * scale_factor) - instr_regret
            else:
                new_regrets[x] = 0 - instr_regret

        memory.append(output_formatter(infoset, new_regrets, self.iteration))

        return instr_regret

    def traverse(self, game: Game, curr_play: int) -> int:
        pass
