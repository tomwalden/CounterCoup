from countercoup.model.items.cards import Duke, Assassin, Ambassador, Captain, Contessa
from itertools import combinations


class Hand:
    """
    Representation of a hand in Coup. Not used in the implementation per se - easier to use lists - but useful
    when doing manipulation.
    """

    card1 = None
    card2 = None

    def __init__(self, cards):
        self.card1 = cards[0]

        if len(cards) > 1:
            self.card2 = cards[1]

    def __eq__(self, other):
        if (self.card1 == other.card1) and (self.card2 == other.card2):
            return True
        elif (self.card2 == other.card1) and (self.card1 == other.card2):
            return True
        else:
            return False

    def __hash__(self):
        # Hope to god that they don't collide!
        return hash(self.card1) * hash(self.card2)

    @staticmethod
    def get_all_hands(cards: []) -> {}:
        """
        Return all possible hands
        :return: all possible hands
        """

        return set([Hand(x) for x in combinations(cards, 2)])








