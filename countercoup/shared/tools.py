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
