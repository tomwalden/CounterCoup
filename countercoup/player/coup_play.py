from countercoup.model.game import Game
from countercoup.model.items.states import SelectAction, SelectCardToLose, SelectCardsToDiscard, \
    DecideToBlockCounteract, DecideToCounteract, DecideToBlock, GameFinished
from countercoup.shared.tools import Tools


class CoupPlay:
    """
    Have a bunch of agents play Coup against each other
    """

    def __init__(self, agents: []):
        """
        Set up the tester
        :param agents: a list of agents that will be playing
        """
        self.agents = agents
        self.tally = [0 for _ in agents]

    def run(self):
        """
        Play a full game of Coup using the agents
        :return: the winning agent
        """

        game = Game(len(self.agents))

        while game.state != GameFinished:

            if game.state == SelectAction:
                strategy = self.agents[game.current_player].get_action_strategy(game)
                action = Tools.select_from_strategy(strategy)

                if action[0].attack_action:
                    game.select_action(action[0], game.get_opponents()[action[1]])
                else:
                    game.select_action(action[0])

            elif game.state == DecideToBlock:
                strategy = self.agents[game.current_player].get_block_strategy(game)
                game.decide_to_block(Tools.select_from_strategy(strategy))

            elif game.state == DecideToCounteract:
                strategy = self.agents[game.current_player].get_counteract_strategy(game)
                game.decide_to_counteract(Tools.select_from_strategy(strategy))

            elif game.state == DecideToBlockCounteract:
                strategy = self.agents[game.current_player].get_block_counteract_strategy(game)
                game.decide_to_block_counteract(Tools.select_from_strategy(strategy))

            elif game.state == SelectCardToLose:
                strategy = self.agents[game.current_player].get_lose_card_strategy(game)
                game.select_card_to_lose(Tools.select_from_strategy(strategy).card1)

            elif game.state == SelectCardsToDiscard:
                strategy = self.agents[game.current_player].get_discard_strategy(game)
                hand = Tools.select_from_strategy(strategy)
                game.select_cards_to_discard(hand.card1, hand.card2)

        self.tally[game.winning_player] += 1
        return game.winning_player


