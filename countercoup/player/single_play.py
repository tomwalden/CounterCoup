from copy import deepcopy
from countercoup.model.game import Game
from countercoup.model.items.states import SelectAction, SelectCardToLose, SelectCardsToDiscard, \
    DecideToBlockCounteract, DecideToCounteract, DecideToBlock, GameFinished
from countercoup.shared.tools import Tools
from countercoup.player.agent import Agent


class SinglePlay:
    """Get the history for a single game"""

    @staticmethod
    def run(self, agents: []) -> []:

        game = Game(len(agents))
        history = [deepcopy(game)]

        while game.state != GameFinished:

            if game.state == SelectAction:
                strategy = agents[game.current_player].get_action_strategy(game)
                action = Tools.select_from_strategy(strategy)

                if action[0].attack_action:
                    game.select_action(action[0], game.get_opponents()[action[1]])
                else:
                    game.select_action(action[0])

            elif game.state == DecideToBlock:
                strategy = agents[game.current_player].get_block_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_block(decision)

            elif game.state == DecideToCounteract:
                strategy = agents[game.current_player].get_counteract_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_counteract(decision)

            elif game.state == DecideToBlockCounteract:
                strategy = agents[game.current_player].get_block_counteract_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.decide_to_block_counteract(decision)

            elif game.state == SelectCardToLose:
                strategy = agents[game.current_player].get_lose_card_strategy(game)
                decision = Tools.select_from_strategy(strategy)
                game.select_card_to_lose(decision.card1)

            elif game.state == SelectCardsToDiscard:
                strategy = agents[game.current_player].get_discard_strategy(game)
                hand = Tools.select_from_strategy(strategy)
                game.select_cards_to_discard(hand.card1, hand.card2)

            history.append(deepcopy(game))

        return history

    @staticmethod
    def get_strategies(agent: Agent, history: []):

        strategies = []

        for h in history:

            if h.state == SelectAction:
                strategies.append((h.current_player, h.action_player, h.state, agent.get_action_strategy(h)))
            elif h.state == DecideToBlock:
                strategies.append((h.current_player, h.action_player, h.state, h.current_action, agent.get_block_strategy(h)))
            elif h.state == DecideToCounteract:
                strategies.append((h.current_player, h.action_player, h.state, h.current_action, agent.get_counteract_strategy(h)))
            elif h.state == DecideToBlockCounteract:
                strategies.append((h.current_player, h.action_player, h.state, h.current_action, agent.get_block_counteract_strategy(h)))
            elif h.state == SelectCardToLose:
                strategies.append((h.current_player, h.action_player, h.state, h.current_action, agent.get_lose_card_strategy(h)))
            elif h.state == SelectCardsToDiscard:
                strategies.append((h.current_player, h.action_player, h.state, h.current_action, agent.get_discard_strategy(h)))

        return strategies



