from countercoup.model.game import Game
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.net_group import NetworkGroup
from countercoup.shared.memory import Memory
from countercoup.shared.structure import Structure
from countercoup.trainer.traverser import Traverser
from multiprocessing import Queue, Process
from time import sleep
from logging import getLogger

class Trainer:
    """
    Overarching class for generating the neural networks needed for Deep CFR in CounterCoup
    """

    _log = getLogger('trainer')

    def __init__(self, num_of_traversals, advantage_memory_size, strategy_memory_size, structure: Structure = None):
        """
        Set up our Trainer
        :param num_of_traversals: number of tree traversals per player
        :param advantage_memory_size: size of memory used to train advantage networks
        :param strategy_memory_size: size of memory used to train strategy networks
        """

        self.num_of_player = 4
        self.num_of_traversals = num_of_traversals
        self.advantage_memory_size = advantage_memory_size
        self.strategy_memory_size = strategy_memory_size
        self.iteration = 0

        self.action_nets = None
        self.action_mem = [Memory(self.advantage_memory_size) for _ in range(self.num_of_player)]
        self.action_strategy_mem = Memory(self.strategy_memory_size)

        self.block_nets = None
        self.block_mem = [Memory(self.advantage_memory_size) for _ in range(self.num_of_player)]
        self.block_strategy_mem = Memory(self.strategy_memory_size)

        self.counteract_nets = None
        self.counteract_mem = [Memory(self.advantage_memory_size) for _ in range(self.num_of_player)]
        self.counteract_strategy_mem = Memory(self.strategy_memory_size)

        self.lose_nets = None
        self.lose_mem = [Memory(self.advantage_memory_size) for _ in range(self.num_of_player)]
        self.lose_strategy_mem = Memory(self.strategy_memory_size)

        self.strategy_nets = None
        self.net_structure = structure

        self.init_advantage_nets()

    def init_advantage_nets(self):
        """
        Set up empty advantage networks
        :return:
        """
        self._log.info('Setting up advantage networks')
        self.action_nets = [ActionNet(structure=self.net_structure) for _ in range(self.num_of_player)]
        self.block_nets = [BlockCounteractNet(structure=self.net_structure) for _ in range(self.num_of_player)]
        self.counteract_nets = [BlockCounteractNet(structure=self.net_structure) for _ in range(self.num_of_player)]
        self.lose_nets = [LoseNet(structure=self.net_structure) for _ in range(self.num_of_player)]

    def train_advantage_nets(self):
        """
        Train the advantage networks
        """
        self._log.info('Training advantage networks')
        for x in range(self.num_of_player):
            self.action_nets[x].train(self.action_mem[x])
            self.block_nets[x].train(self.block_mem[x])
            self.counteract_nets[x].train(self.counteract_mem[x])
            self.lose_nets[x].train(self.lose_mem[x])

    def perform_run(self, num_of_processes, num_of_traversals):
        """
        Perform a set number of traversals, and train the strategy networks
        :param num_of_processes: number of threads to run the traversals on
        :param num_of_traversals: number of traversals to
        """
        for t in range(num_of_traversals):
            self._log.info('Performing iteration {num}'.format(num=t))
            self.perform_iteration(num_of_processes)

        self.train_strategy_nets()

    def train_strategy_nets(self):
        """
        Train the strategy networks
        """

        self.strategy_nets = NetworkGroup(structure=self.net_structure)
        self.strategy_nets.train_networks(self.action_strategy_mem
                                          , self.block_strategy_mem
                                          , self.counteract_strategy_mem
                                          , self.lose_strategy_mem)

    def perform_iteration(self, num_of_processes: int = 2):
        """
        Perform one iteration of the Deep CFR algorithm
        :param num_of_processes: number of threads to run the traversals on
        """
        self.iteration += 1

        input_queue = Queue()
        output_queue = Queue()

        # We need one game for each player in each traversal
        for k in range(self.num_of_traversals):
            for p in range(self.num_of_player):
                input_queue.put(p)

        processes = []

        # Spin up our processes
        for p in range(num_of_processes):
            traverser = Traverser(self.action_nets
                                  , self.block_nets
                                  , self.counteract_nets
                                  , self.lose_nets
                                  , self.iteration)
            processes.append(Process(target=self.run_process, args=(input_queue
                                                                    , output_queue
                                                                    , traverser
                                                                    , self.num_of_player)))
            processes[p].start()

        # Wait for each process to finish. When the results from a thread come through, add them to
        # the memories
        counter = 0
        while counter < num_of_processes:
            if output_queue.empty():
                self._log.info('Queue size: {size}'.format(size=input_queue.qsize()))
                sleep(30)
            else:
                x = output_queue.get()

                for p in range(self.num_of_player):
                    self.action_mem[p].add_bulk(x[0][p])
                    self.block_mem[p].add_bulk(x[1][p])
                    self.counteract_mem[p].add_bulk(x[2][p])
                    self.lose_mem[p].add_bulk(x[3][p])

                self.action_strategy_mem.add_bulk(x[4])
                self.block_strategy_mem.add_bulk(x[5])
                self.counteract_strategy_mem.add_bulk(x[6])
                self.lose_strategy_mem.add_bulk(x[7])

                counter += 1

        # Set up and train the advantage networks
        self.init_advantage_nets()
        self.train_advantage_nets()

    @staticmethod
    def run_process(input_queue: Queue, output_queue: Queue, traverser: Traverser, num_of_players: int):
        """
        Process that runs in each thread during game tree traversal
        :param input_queue: input queue containing traversals that need to be done
        :param output_queue: output queue containing results from the thread
        :param traverser: the Traverser object for this thread
        :param num_of_players: total number of players in the game
        """

        while not input_queue.empty():
            p = input_queue.get()
            traverser.traverse(Game(num_of_players), p)

        output_queue.put((traverser.action_mem
                          , traverser.block_mem
                          , traverser.counteract_mem
                          , traverser.lose_mem
                          , traverser.action_strategy_mem
                          , traverser.block_strategy_mem
                          , traverser.counteract_strategy_mem
                          , traverser.lose_strategy_mem))
