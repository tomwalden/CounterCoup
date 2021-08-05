from countercoup.model.game import Game
from countercoup.model.hand import Hand
from countercoup.model.items.states import SelectAction, DecideToBlock, DecideToCounteract, SelectCardsToDiscard\
    , GameFinished, DecideToBlockCounteract, SelectCardToLose
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.network import Network
from countercoup.shared.infoset import Infoset
from countercoup.shared.tools import Tools
from countercoup.trainer.trainer_stats import TrainerStats
from copy import deepcopy
from random import sample


class Traverser:
    """
    Class that allows for parallel traversals of the game tree
    """

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

    def traverse(self, game: Game, curr_play: int):
        """
        Traverse the Coup game tree recursively
        :param game: the Game object from the model being played
        :param curr_play: the current player model
        :return: the instantaneous regret value for all histories at this prefix
        """

        self.stats.total_nodes_traversed += 1

        # If the game is finished, then return 1 if the player won the game, else -1
        if game.state == GameFinished:

            self.stats.total_terminal_nodes += 1
            self.stats.total_hist_length += game.get_game_length()

            if game.winning_player == curr_play:
                self.stats.game_wins += 1
                return 1
            else:
                self.stats.game_loses += 1
                return -1

        else:

            # If, for some reason the game has gone on for 50 turns, everyone declares the game
            # a draw and goes to the pub
            if game.get_game_length() >= 50:
                return 0

            # Game continues...
            infoset = Infoset(game)
            values = {}

            if game.current_player == curr_play:
                if game.state == SelectAction:
                    self.stats.total_turns += 1

                    strategy = self.get_regret_strategy(self.action_nets[curr_play]
                                                        , infoset
                                                        , Tools.get_actions(game))

                    for x in sample(strategy.keys(), min(3 if game.get_game_length() < 16 else 1, len(strategy))):
                        next_game = deepcopy(game)

                        if x[0].attack_action:
                            next_game.select_action(x[0], next_game.get_opponents()[x[1]])
                        else:
                            next_game.select_action(x[0])

                        values[x] = self.traverse(next_game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.action_mem[curr_play]
                                                  , infoset
                                                  , ActionNet.create_train_data)

                elif game.state == DecideToBlock:
                    strategy = self.get_regret_strategy(self.block_nets[curr_play], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_block(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.block_mem[curr_play]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == DecideToCounteract:
                    strategy = self.get_regret_strategy(self.counteract_nets[curr_play], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_counteract(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.counteract_mem[curr_play]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == DecideToBlockCounteract:
                    strategy = self.get_regret_strategy(self.block_nets[curr_play], infoset)

                    choice = sample(strategy.keys(), 1)[0]
                    game.decide_to_block_counteract(choice)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.block_mem[curr_play]
                                                  , infoset
                                                  , BlockCounteractNet.create_train_data)

                elif game.state == SelectCardsToDiscard:
                    strategy = self.get_regret_strategy(self.lose_nets[curr_play]
                                                        , infoset
                                                        , Hand.get_all_hands(game.get_curr_player().cards))

                    # For Exchange, select 2 possible discard selections
                    for x in sample(strategy.keys(), min(2 if game.get_game_length() < 16 else 1, len(strategy))):
                        next_game = deepcopy(game)

                        next_game.select_cards_to_discard(x.card1, x.card2)
                        values[x] = self.traverse(next_game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.lose_mem[curr_play]
                                                  , infoset
                                                  , LoseNet.create_train_data)

                elif game.state == SelectCardToLose:
                    lose_hand = Hand.get_singular_hands(game.get_curr_player().cards)
                    strategy = self.get_regret_strategy(self.lose_nets[curr_play], infoset, lose_hand)

                    choice = sample(strategy.keys(), 1)[0]
                    game.select_card_to_lose(choice.card1)
                    values[choice] = self.traverse(game, curr_play)

                    return self.calculate_regrets(values
                                                  , strategy
                                                  , self.lose_mem[curr_play]
                                                  , infoset
                                                  , LoseNet.create_train_data)

            else:
                if game.state == SelectAction:
                    self.stats.total_turns += 1

                    strategy = self.get_regret_strategy(self.action_nets[game.current_player]
                                                        , infoset
                                                        , Tools.get_actions(game))
                    self.action_strategy_mem.append(ActionNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    if choice[0].attack_action:
                        game.select_action(choice[0], game.get_opponents()[choice[1]])
                    else:
                        game.select_action(choice[0])

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlock:
                    strategy = self.get_regret_strategy(self.block_nets[game.current_player], infoset)
                    self.block_strategy_mem.append(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_block(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToCounteract:
                    strategy = self.get_regret_strategy(self.counteract_nets[game.current_player], infoset)
                    self.counteract_strategy_mem.append(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_counteract(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlockCounteract:
                    strategy = self.get_regret_strategy(self.block_nets[game.current_player], infoset)
                    self.block_strategy_mem.append(BlockCounteractNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.decide_to_block_counteract(choice)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardsToDiscard:
                    strategy = self.get_regret_strategy(self.lose_nets[game.current_player]
                                                        , infoset
                                                        , Hand.get_all_hands(game.get_curr_player().cards))
                    self.lose_strategy_mem.append(LoseNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.select_cards_to_discard(choice.card1, choice.card2)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardToLose:
                    lose_hand = Hand.get_singular_hands(game.get_curr_player().cards)
                    strategy = self.get_regret_strategy(self.lose_nets[game.current_player]
                                                        , infoset
                                                        , lose_hand)
                    self.lose_strategy_mem.append(LoseNet.create_train_data(infoset, strategy, self.iteration))

                    choice = Tools.select_from_strategy(strategy)

                    game.select_card_to_lose(choice.card1)

                    return self.traverse(game, curr_play)

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
