from countercoup.shared.net_group import NetworkGroup
from countercoup.shared.tools import Tools
from countercoup.shared.infoset import Infoset
from countercoup.player.agent import Agent
from countercoup.model.game_info import GameInfoSet
from countercoup.model.hand import Hand


class CounterCoup(Agent):
    """
    Agent that plays according to our CounterCoup networks
    """

    def __init__(self, file_path):
        self.net_group = NetworkGroup(file_path)

    def get_action_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.action.get_output(Infoset(g), Tools.get_actions(g)))

    def get_block_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.block.get_output(Infoset(g)))

    def get_counteract_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.counteract.get_output(Infoset(g)))

    def get_block_counteract_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.block.get_output(Infoset(g)))

    def get_lose_card_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.lose.get_output(Infoset(g), Hand.get_singular_hands(g.get_curr_player().cards)))

    def get_discard_strategy(self, g: GameInfoSet) -> dict:
        return Tools.normalise(self.net_group.lose.get_output(Infoset(g), Hand.get_all_hands(g.get_curr_player().cards)))
