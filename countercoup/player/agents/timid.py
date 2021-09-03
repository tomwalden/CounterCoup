from countercoup.player.agent import Agent
from countercoup.model.game_info import GameInfoSet
from countercoup.model.hand import Hand
from countercoup.shared.tools import Tools


class Timid(Agent):
    """
    Agent that never bluffs, and never blocks, but always counteracts if it has the card
    """

    def get_action_strategy(self, g: GameInfoSet) -> dict:
        actions = [x for x in Tools.get_actions(g)
                   if x[0].action_card in g.get_curr_player().cards or x[0].action_card is None]
        return {x: 1 / len(actions) for x in actions}

    def get_block_strategy(self, g: GameInfoSet) -> dict:
        return {True: 0, False: 1}

    def get_counteract_strategy(self, g: GameInfoSet) -> dict:
        cards = [x for x in g.current_action.c_action_cards if x in g.get_curr_player().cards]
        if cards:
            return {True: 1, False: 0}
        else:
            return {True: 0, False: 1}

    def get_block_counteract_strategy(self, g: GameInfoSet) -> dict:
        return {True: 0, False: 1}

    def get_lose_card_strategy(self, g: GameInfoSet) -> dict:
        hands = Hand.get_singular_hands(g.get_curr_player().cards)
        return {x: 1 / len(hands) for x in hands}

    def get_discard_strategy(self, g: GameInfoSet) -> dict:
        hands = Hand.get_all_hands(g.get_curr_player().cards)
        return {x: 1 / len(hands) for x in hands}

