from countercoup.model.game import Game


class Agent:
    """
    Base class for agents
    """

    def get_action_strategy(self, g: Game) -> dict:
        pass

    def get_block_strategy(self, g: Game) -> dict:
        pass

    def get_counteract_strategy(self, g: Game) -> dict:
        pass

    def get_block_counteract_strategy(self, g: Game) -> dict:
        pass

    def get_lose_card_strategy(self, g: Game) -> dict:
        pass

    def get_discard_strategy(self, g: Game) -> dict:
        pass
