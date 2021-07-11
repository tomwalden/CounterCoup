from countercoup.model.game import Game
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from random import random


class Tools:
    """
    Tools shared by everything in CounterCoup
    """

    @staticmethod
    def select_from_strategy(dist: dict):
        """
        Select a key from a strategy
        :param dist: the strategy distribution in the form {action: prob}
        :return: the action selected
        """

        val = random()
        for x in dist:
            val -= dist[x]
            if val <= 0:
                return x

        # Not properly normalised!
        return None

    @staticmethod
    def get_actions(g: Game):
        """
        Returns available actions for the current player, assuming that we're at the SelectAction state
        :param g: the Coup game
        :return: a list of tuples of allowable actions
        """

        act_actions = []
        act = [Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal]

        opponents = g.get_opponents()

        if g.players[g.action_player].coins >= 10:
            act = [Coup]

        for x in act:
            if g.players[g.action_player].coins >= x.cost:
                if x.attack_action:
                    for p in range(len(opponents)):
                        if g.players[opponents[p]].in_game:
                            act_actions.append((x, p))
                else:
                    act_actions.append((x, None))

        return act_actions
