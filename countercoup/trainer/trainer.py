from countercoup.model.game import Game
from countercoup.model.hand import Hand
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from countercoup.model.items.states import SelectAction, DecideToBlock, DecideToCounteract, SelectCardsToDiscard\
    , GameFinished, DecideToBlockCounteract, SelectCardToLose
from countercoup.trainer.networks.ActionNet import ActionNet
from countercoup.trainer.networks.BlockCounteractNet import BlockCounteractNet
from countercoup.trainer.networks.LoseNet import LoseNet
from countercoup.trainer.network import Network
from countercoup.trainer.memory import Memory
from countercoup.shared.infoset import Infoset
from countercoup.shared.tools import Tools
from copy import deepcopy
from random import sample


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
        self.action_strategy_net = ActionNet()
        self.action_strategy_mem = Memory(self.memory_size)

        self.block_nets = None
        self.block_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.block_strategy_net = BlockCounteractNet()
        self.block_strategy_mem = Memory(self.memory_size)

        self.counteract_nets = None
        self.counteract_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.counteract_strategy_nets = BlockCounteractNet()
        self.counteract_strategy_mem = Memory(self.memory_size)

        self.lose_nets = None
        self.lose_mem = [Memory(self.memory_size) for _ in range(self.num_of_player)]
        self.lose_strategy_net = LoseNet()
        self.lose_strategy_mem = Memory(self.memory_size)

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

    def perform_iteration(self):
        """
        Perform one iteration of the Deep CFR algorithm
        """
        self.iteration += 1

        for player in range(1, self.num_of_player + 1):
            for k in range(self.num_of_traversals):
                self.traverse(Game(self.num_of_player), player)

        self.init_advantage_nets()
        self.train_advantage_nets()

    def train_strategy_networks(self):
        """
        Train the strategy networks, at the end
        """
        self.action_strategy_net.train(self.action_strategy_mem)
        self.block_strategy_net.train(self.block_strategy_mem)
        self.counteract_strategy_nets.train(self.counteract_strategy_mem)
        self.lose_strategy_net.train(self.lose_strategy_mem)

    def traverse(self, game: Game, curr_play: int):
        """
        Traverse the Coup game tree recursively
        :param game: the Game object from the model being played
        :param curr_play: the current player model
        :return: the instantaneous regret value for all histories at this prefix
        """

        # If the game is finished, then return 1 if the player won the game, else -1
        if game.state == GameFinished:
            if game.winning_player == curr_play:
                return 1
            else:
                return -1
        else:
            # Game continues...
            infoset = Infoset(game)
            values = {}

            if game.current_player == curr_play:
                if game.state == SelectAction:
                    strategy = self.get_regret_strategy(self.action_nets[curr_play - 1]
                                                        , infoset
                                                        , self.get_actions(game))

                    for x in sample(strategy.keys(), 3):
                        next_game = deepcopy(game)

                        if x[0].attack_action:
                            next_game.select_action(x[0], next_game.get_opponents()[x[1]])
                        else:
                            next_game.select_action(x[0])

                        values[x] = self.traverse(next_game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.action_mem[curr_play - 1]
                                                  , infoset
                                                  , ActionNet.create_train_data)

                elif game.state == DecideToBlock:
                    strategy = self.get_regret_strategy(self.block_nets[curr_play - 1], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_block(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.block_mem[curr_play - 1]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == DecideToCounteract:
                    strategy = self.get_regret_strategy(self.counteract_nets[curr_play - 1], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_counteract(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.counteract_mem[curr_play - 1]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == DecideToBlockCounteract:
                    strategy = self.get_regret_strategy(self.block_nets[curr_play - 1], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_block_counteract(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.block_mem[curr_play - 1]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == SelectCardsToDiscard:
                    strategy = self.get_regret_strategy(self.lose_nets[curr_play - 1]
                                                        , infoset
                                                        , Hand.get_all_hands(game.get_curr_player().cards))

                    # For Exchange, select 2 possible discard selections
                    for x in sample(strategy.keys(), 2):
                        next_game = deepcopy(game)

                        next_game.select_cards_to_discard(x.card1, x.card2)
                        values[x] = self.traverse(next_game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.lose_mem[curr_play - 1]
                                                  , infoset
                                                  , LoseNet.create_train_data)

                elif game.state == SelectCardToLose:
                    lose_hand = [Hand([game.get_curr_player().cards[0]]), Hand([game.get_curr_player().cards[1]])]
                    strategy = self.get_regret_strategy(self.lose_nets[curr_play - 1], infoset, lose_hand)

                    choice = sample(strategy.keys(), 1)[0]
                    game.select_card_to_lose(choice.card1)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.lose_mem[curr_play - 1]
                                                  , infoset
                                                  , LoseNet.create_train_data)

            else:
                if game.state == SelectAction:
                    strategy = self.get_regret_strategy(self.action_nets[game.current_player - 1]
                                                        , infoset
                                                        , self.get_actions(game))
                    self.action_strategy_mem.add(ActionNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    if choice[0].attack_action:
                        game.select_action(choice[0], choice[1])
                    else:
                        game.select_action(choice[0])

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlock:
                    strategy = self.get_regret_strategy(self.block_nets[game.current_player - 1], infoset)
                    self.block_strategy_mem.add(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_block(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToCounteract:
                    strategy = self.get_regret_strategy(self.counteract_nets[game.current_player - 1], infoset)
                    self.counteract_strategy_mem.add(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_counteract(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlockCounteract:
                    strategy = self.get_regret_strategy(self.block_nets[game.current_player - 1], infoset)
                    self.block_strategy_mem.add(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_block_counteract(choice)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardsToDiscard:
                    strategy = self.get_regret_strategy(self.lose_nets[game.current_player - 1]
                                                        , infoset
                                                        , Hand.get_all_hands(game.get_curr_player().cards))
                    self.lose_strategy_mem.add(LoseNet.create_train_data(infoset, strategy))

                    choice = Tools.select_from_strategy(strategy)

                    game.select_cards_to_discard(choice.card1, choice.card2)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardToLose:
                    lose_hand = [Hand([game.get_curr_player().cards[0]]), Hand([game.get_curr_player().cards[1]])]
                    strategy = self.get_regret_strategy(self.lose_nets[game.current_player - 1]
                                                        , infoset
                                                        , lose_hand)
                    self.lose_strategy_mem.add(LoseNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.select_card_to_lose(choice.card1)

                    return self.traverse(game, curr_play)

    @staticmethod
    def get_regret_strategy(network: Network, infoset: Infoset, filt: [] = None):
        output = network.get_output(infoset, filt)
        total = sum(filter(lambda x: x > 0, output.values()))

        if total == 0:
            return {x: 1 / len(output) for x in output}
        else:
            return {x: output[x] if output[x] > 0 else 0 / total for x in output}

    def calculate_regrets(self, values: {}, strategy: {}, memory: Memory, infoset: Infoset, output_formatter):
        """
        Calculate the regret values (and insert them into memory)
        :param values:
        :param strategy:
        :param memory:
        :param infoset:
        :param output_formatter:
        :return:
        """

        instr_regret = 0

        for x in values:
            instr_regret += strategy[x] * values[x]

        # Scale the instantaneous regret by the inverse of the fraction of actions selected
        instr_regret *= len(strategy) / len(values)

        new_regrets = {}
        for x in strategy:
            if x in values:
                new_regrets[x] = values[x] - instr_regret
            else:
                new_regrets[x] = 0 - instr_regret

        memory.add(output_formatter(infoset, new_regrets, self.iteration))

        return instr_regret

    @staticmethod
    def get_actions(g: Game):
        """
        Returns available actions for the current player, assuming that we're at the SelectAction state
        :param g: the Coup game
        :return: a list of tuples of allowable actions
        """

        act_actions = []
        act = [Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal]

        opponents = g.get_opponents()

        if g.players[g.action_player].coins >= 10:
            act = [Coup]

        for x in act:
            if g.players[g.action_player].coins >= x.cost:
                if x.attack_action:
                    for p in range(len(opponents)):
                        if g.players[opponents[p]].in_game:
                            act_actions.append((x, p))
                else:
                    act_actions.append((x, None))

        return act_actions

