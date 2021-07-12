from countercoup.model.game import Game
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.net_group import NetworkGroup
from countercoup.shared.memory import Memory
from countercoup.trainer.traverser import Traverser
from multiprocessing import Queue, Process


class Trainer:
    """
    Overarching class for generating the neural networks needed for Deep CFR in CounterCoup
    """

    def __init__(self, num_of_traversals, memory_size):
        self.num_of_player = 4
        self.num_of_traversals = num_of_traversals
        self.memory_size = memory_size
        self.iteration = 0

        self.action_nets = None
        self.action_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.action_strategy_mem = Memory(self.memory_size)

        self.block_nets = None
        self.block_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.block_strategy_mem = Memory(self.memory_size)

        self.counteract_nets = None
        self.counteract_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.counteract_strategy_nets = BlockCounteractNet()
        self.counteract_strategy_mem = Memory(self.memory_size)

        self.lose_nets = None
        self.lose_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.lose_strategy_mem = Memory(self.memory_size)

        self.strategy_nets = NetworkGroup()

        self.init_advantage_nets()

    def init_advantage_nets(self):
        """
        Set up empty advantage networks
        :return:
        """
        self.action_nets = [ActionNet() for _ in range(self.num_of_player)]
        self.block_nets = [BlockCounteractNet() for _ in range(self.num_of_player)]
        self.counteract_nets = [BlockCounteractNet() for _ in range(self.num_of_player)]
        self.lose_nets = [LoseNet() for _ in range(self.num_of_player)]

    def train_advantage_nets(self):
        """
        Train the advantage networks
        """
        for x in range(self.num_of_player):
            self.action_nets[x].train(self.action_mem[x])
            self.block_nets[x].train(self.block_mem[x])
            self.counteract_nets[x].train(self.counteract_mem[x])
            self.lose_nets[x].train(self.lose_mem[x])

    def perform_iteration(self, num_of_processes: int = 2):
        """
        Perform one iteration of the Deep CFR algorithm
        """
        self.iteration += 1

        input_queue = Queue()
        output_queue = Queue()

        for k in range(self.num_of_traversals):
            for p in range(self.num_of_player):
                input_queue.put(p)

        processes = []

        for x in range(num_of_processes):
            traverser = Traverser(self.action_nets
                                  , self.block_nets
                                  , self.counteract_nets
                                  , self.lose_nets
                                  , self.iteration)
            processes.append(Process(target=self.run_process, args=(input_queue
                                                                    , output_queue
                                                                    , traverser
                                                                    , self.num_of_player)))
            processes[x].start()

        # Now wait for all the tree traversals to finish
        for x in processes:
            x.join()

        while not output_queue.empty():
            x = output_queue.get()

            for p in range(self.num_of_player):
                self.action_mem[p].add_bulk(x.action_mem)
                self.block_mem[p].add_bulk(x.block_mem)
                self.counteract_mem[p].add_bulk(x.counteract_mem)
                self.lose_mem[p].add_bulk(x.lose_mem)

            self.action_strategy_mem.add_bulk(x.action_strategy_mem)
            self.block_strategy_mem.add_bulk(x.block_strategy_mem)
            self.counteract_strategy_mem.add_bulk(x.counteract_strategy_mem)
            self.lose_strategy_mem.add_bulk(x.lose_strategy_mem)

        self.init_advantage_nets()
        self.train_advantage_nets()

    @staticmethod
    def run_process(input_queue: Queue, output_queue: Queue, traverser: Traverser, num_of_players: int):
        while not input_queue.empty():
            p = input_queue.get()
            traverser.traverse(Game(num_of_players), p)
        output_queue.put(traverser)
