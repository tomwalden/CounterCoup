from countercoup.player.agent import Agent
from countercoup.model.game import Game
from countercoup.model.hand import Hand
from countercoup.shared.tools import Tools


class Random(Agent):
    """
    Completely random agent
    """

    def get_action_strategy(self, g: Game) -> dict:
        actions = Tools.get_actions(g)
        return {x: 1 / len(actions) for x in actions}

    def get_block_strategy(self, g: Game) -> dict:
        return {True: 0.5, False: 0.5}

    def get_counteract_strategy(self, g: Game) -> dict:
        return {True: 0.5, False: 0.5}

    def get_block_counteract_strategy(self, g: Game) -> dict:
        return {True: 0.5, False: 0.5}

    def get_lose_card_strategy(self, g: Game) -> dict:
        hands = Hand.get_singular_hands(g.get_curr_player().cards)
        return {x: 1 / len(hands) for x in hands}

    def get_discard_strategy(self, g: Game) -> dict:
        hands = Hand.get_all_hands(g.get_curr_player().cards)
        return {x: 1 / len(hands) for x in hands}

