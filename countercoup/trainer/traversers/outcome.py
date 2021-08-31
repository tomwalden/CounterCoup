from countercoup.model.game import Game
from countercoup.model.hand import Hand
from countercoup.model.items.states import SelectAction, DecideToBlock, DecideToCounteract, SelectCardsToDiscard\
    , GameFinished, DecideToBlockCounteract, SelectCardToLose
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.infoset import Infoset
from countercoup.shared.tools import Tools
from countercoup.trainer.traverser import Traverser
from copy import deepcopy
from random import sample


class Outcome(Traverser):
    """
    Traverser for outcome sampling - select one action per turn
    """

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

            # Game continues...
            infoset = Infoset(game)
            values = {}

            if game.current_player == curr_play:
                if game.state == SelectAction:
                    self.stats.total_turns += 1

                    strategy = self.get_regret_strategy(self.action_nets[curr_play]
                                                        , infoset
                                                        , Tools.get_actions(game))

                    for x in sample(strategy.keys(), 1):
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
                    for x in sample(strategy.keys(), 1):
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
