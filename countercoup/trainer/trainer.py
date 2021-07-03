from countercoup.model.game import Game
from countercoup.model.hand import Hand
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from countercoup.model.items.states import SelectAction, DecideToBlock, DecideToCounteract, SelectCardsToDiscard\
    , GameFinished, DecideToBlockCounteract, SelectCardToLose
from countercoup.trainer.networks.ActionNet import ActionNet
from countercoup.trainer.networks.BlockCounteractNet import BlockCounteractNet
from countercoup.trainer.networks.LoseNet import LoseNet
from countercoup.shared.infoset import Infoset
from countercoup.shared.tools import Tools
from copy import deepcopy
from itertools import combinations
from random import sample


class Trainer:
    """
    Overarching class for generating the neural networks needed for Deep CFR in CounterCoup
    """

    def __init__(self, num_of_player, num_of_traversals):
        self.num_of_player = num_of_player
        self.num_of_traversals = num_of_traversals

        self.action_advantage_nets = []
        self.action_advantage_mem = []
        self.action_strategy_nets = None
        self.action_strategy_mem = None

        self.block_nets = []
        self.block_mem = []
        self.block_strategy_nets = None
        self.block_strategy_mem = None

        self.counteract_nets = []
        self.counteract_mem = []
        self.counteract_strategy_mem = None
        self.counteract_strategy_nets = None

        self.lose_nets = []
        self.lose_mem = []
        self.lose_strategy_nets = None
        self.lose_strategy_mem = None

    def perform_iteration(self):
        """
        Perform one iteration of the Deep CFR algorithm
        """

        for player in range(self.num_of_player):
            for k in range(self.num_of_traversals):
                self.traverse(Game(self.num_of_player), player)

    def traverse(self, game: Game, curr_play: int):

        # If the game is finished, then return 1 if the player won the game, else 0
        if game.state == GameFinished:
            if game.winning_player == curr_play:
                return 1
            else:
                return 0
        else:
            # Game continues...
            infoset = Infoset(game)

            if game.current_player == curr_play:
                if game.state == SelectAction:
                    new_adv = self.action_advantage_nets[curr_play].get_output(infoset, self.get_actions(game))
                    strat_adv = self.normalize(new_adv)

                    for x in sample(new_adv.keys(), 3):
                        next_game = deepcopy(game)

                        if x[0].attack_action:
                            next_game.select_action(x[0], x[1])
                        else:
                            next_game.select_action(x[0])

                        new_adv[x] = self.traverse(next_game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.action_advantage_mem[curr_play].add((infoset, ActionNet.create_output({x: new_adv[x] - inst_regret for x in new_adv})))
                    return inst_regret

                elif game.state == DecideToBlock:
                    new_adv = self.block_nets[curr_play].get_output(infoset)
                    strat_adv = self.normalize(new_adv)

                    choice = sample(new_adv.keys(), 1)[0]
                    game.decide_to_block(choice)
                    new_adv[choice] = self.traverse(game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.block_mem[curr_play].add((infoset, BlockCounteractNet.create_output({x: new_adv[x] - inst_regret for x in new_adv})))
                    return inst_regret

                elif game.state == DecideToCounteract:
                    new_adv = self.counteract_nets[curr_play].get_output(infoset)
                    strat_adv = self.normalize(new_adv)

                    choice = sample(new_adv.keys(), 1)[0]
                    game.decide_to_counteract(choice)
                    new_adv[choice] = self.traverse(game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.block_mem[curr_play].add((infoset, BlockCounteractNet.create_output({x: new_adv[x] - inst_regret for x in new_adv})))
                    return inst_regret

                elif game.state == DecideToBlockCounteract:
                    new_adv = self.block_nets[curr_play].get_output(infoset)
                    strat_adv = self.normalize(new_adv)

                    choice = sample(new_adv.keys(), 1)[0]
                    game.decide_to_block_counteract(choice)
                    new_adv[choice] = self.traverse(game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.block_mem[curr_play].add((infoset, BlockCounteractNet.create_output({x: new_adv[x] - inst_regret for x in new_adv})))
                    return inst_regret

                elif game.state == SelectCardsToDiscard:
                    new_adv = self.lose_nets[curr_play].get_output(infoset, Hand.get_all_hands(game.current_player().cards))
                    strat_adv = self.normalize(new_adv)

                    # For Exchange, select 2 possible discard selections
                    for x in sample(new_adv.keys(), 2):
                        next_game = deepcopy(game)

                        next_game.select_cards_to_discard(x.card1, x.card2)
                        new_adv[x] = self.traverse(next_game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.lose_mem[curr_play].add((infoset, LoseNet.create_output({x: new_adv[x] - inst_regret for x in new_adv})))
                    return inst_regret

                elif game.state == SelectCardToLose:
                    lose_hand = [Hand([game.get_curr_player().cards[0]]), Hand([game.get_curr_player().cards[1]])]
                    new_adv = self.lose_nets[curr_play].get_output(infoset, lose_hand)
                    strat_adv = self.normalize(new_adv)

                    choice = sample(new_adv.keys(), 1)[0]
                    game.select_card_to_lose(choice.card1)
                    new_adv[choice] = self.traverse(game, curr_play)

                    inst_regret = sum([strat_adv[x] * new_adv[x] for x in new_adv])
                    self.lose_mem[curr_play].add((infoset, {x: new_adv[x] - inst_regret for x in new_adv}))
                    return inst_regret
            else:
                if game.state == SelectAction:
                    new_adv = self.action_advantage_nets[game.current_player].get_output(infoset, self.get_actions(game))
                    strat_adv = self.normalize(new_adv)
                    self.action_strategy_mem.add((infoset, strat_adv))

                    choice = Tools.select_from_strategy(strat_adv)
                    if choice[0].attack_action:
                        game.select_action(choice[0], choice[1])
                    else:
                        game.select_action(choice[0])

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlock:
                    new_adv = self.block_nets[game.current_player].get_output(infoset)
                    strat_adv = self.normalize(new_adv)
                    self.block_strategy_mem.add((infoset, strat_adv))

                    choice = Tools.select_from_strategy(strat_adv)
                    game.decide_to_block(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToCounteract:
                    new_adv = self.counteract_nets[game.current_player].get_output(infoset)
                    strat_adv = self.normalize(new_adv)
                    self.block_strategy_mem.add((infoset, strat_adv))

                    choice = Tools.select_from_strategy(strat_adv)
                    game.decide_to_block(choice)

                    return self.traverse(game, curr_play)

                elif game.state == DecideToBlockCounteract:
                    new_adv = self.block_nets[game.current_player].get_output(infoset)
                    strat_adv = self.normalize(new_adv)
                    self.block_strategy_mem.add((infoset, strat_adv))

                    choice = Tools.select_from_strategy(strat_adv)
                    game.decide_to_block(choice)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardsToDiscard:
                    new_adv = self.lose_nets[game.current_player].get_output(infoset, Hand.get_all_hands(game.current_player().cards))
                    strat_adv = self.normalize(new_adv)
                    self.lose_strategy_mem.add((infoset, strat_adv))

                    choice = Tools.select_from_strategy(strat_adv)
                    game.select_cards_to_discard(choice.card1, choice.card2)

                    return self.traverse(game, curr_play)

                elif game.state == SelectCardToLose:
                    lose_hand = [Hand([game.get_curr_player().cards[0]]), Hand([game.get_curr_player().cards[1]])]
                    new_adv = self.lose_nets[curr_play].get_output(infoset, lose_hand)
                    strat_adv = self.normalize(new_adv)

                    choice = Tools.select_from_strategy(strat_adv)
                    game.select_card_to_lose(choice.card1)

                    return self.traverse(game, curr_play)

    @staticmethod
    def normalize(output: {}):
        sum = 0

        for x in output:
            sum += output[x]

        if sum == 0:
            return {x: 1 / len(output) for x in output}
        else:
            return {x: output[x] / sum for x in output}

    @staticmethod
    def get_actions(g: Game):
        """
        Returns available actions for the current player, assuming that we're at the SelectAction state
        :param g: the Coup game
        :return: a list of tuples of allowable actions
        """

        act_actions = []
        act = [Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal]

        opponents = [i for i, p in enumerate(g.get_opponents()) if p.in_game]

        if g.players[g.action_player].coins >= 10:
            act = [Coup]

        for x in act:
            if g.players[g.action_player].coins >= x.cost:
                if x.attack_action:
                    for p in opponents:
                        act_actions.append((x, p))
                else:
                    act_actions.append((x, None))

        return act_actions

