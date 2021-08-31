from countercoup.model.game import Game
from countercoup.model.items.states import SelectAction, SelectCardToLose, SelectCardsToDiscard, \
    DecideToBlockCounteract, DecideToCounteract, DecideToBlock, GameFinished
from countercoup.shared.networks.action_net import ActionNet
from countercoup.shared.networks.block_counteract_net import BlockCounteractNet
from countercoup.shared.networks.lose_net import LoseNet
from countercoup.shared.tools import Tools


class SelfPlay:
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

        # Log the actions that take place
        self.action_tally = [{x: 0 for x in ActionNet.outputs} for _ in agents]
        self.block_tally = [{x: 0 for x in BlockCounteractNet.outputs} for _ in agents]
        self.counteract_tally = [{x: 0 for x in BlockCounteractNet.outputs} for _ in agents]
        self.block_counteract_tally = [{x: 0 for x in BlockCounteractNet.outputs} for _ in agents]
        self.lose_tally = [{x: 0 for x in LoseNet.outputs} for _ in agents]

        self.histories = [[] for _ in agents]

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

                self.action_tally[game.current_player][action] += 1

            elif game.state == DecideToBlock:
                strategy = self.agents[game.current_player].get_block_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_block(decision)

                self.block_tally[game.current_player][decision] += 1

            elif game.state == DecideToCounteract:
                strategy = self.agents[game.current_player].get_counteract_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_counteract(decision)

                self.counteract_tally[game.current_player][decision] += 1

            elif game.state == DecideToBlockCounteract:
                strategy = self.agents[game.current_player].get_block_counteract_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_block_counteract(decision)

                self.block_counteract_tally[game.current_player][decision] += 1

            elif game.state == SelectCardToLose:
                strategy = self.agents[game.current_player].get_lose_card_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.select_card_to_lose(decision.card1)

                self.lose_tally[game.current_player][decision] += 1

            elif game.state == SelectCardsToDiscard:
                strategy = self.agents[game.current_player].get_discard_strategy(game)
                hand = Tools.select_from_strategy(strategy)
                game.select_cards_to_discard(hand.card1, hand.card2)

                self.lose_tally[game.current_player][hand] += 1

        self.tally[game.winning_player] += 1
        self.histories[game.winning_player].append(game.history)

        return game.winning_player

    def run_batch(self, num):
        """
        Run a batch of games
        :param num: the number of games to play
        """

        for _ in range(num):
            self.run()
