from countercoup.model.action import Action
from countercoup.model.items.cards import Assassin, Ambassador, Captain, Duke, Contessa


class Assassinate(Action):
    """
    Assassinate - get rid of an opponents card
    """

    action_card = Assassin
    c_action_cards = [Contessa]
    attack_action = True

    cost = 3


class Coup(Action):
    """
    Coup - pick a player to remove influence. Can't be blocked or counteractioned.
    """

    cost = 7
    attack_action = True


class ForeignAid(Action):
    """
    Foreign aid - take 2 coins from treasury. Can't be blocked, but can be countered by Duke
    """

    c_action_cards = [Duke]


class Income(Action):
    """
    Income - add 1 coin from treasury. Can't be blocked or counteractioned
    """
    pass


class Steal(Action):
    """
    Steal - take 2 coins from another player
    """

    action_card = Captain
    c_action_cards = [Captain, Ambassador]
    attack_action = True


class Tax(Action):
    """
    Tax - get 3 coins. Can be blocked though.
    """

    action_card = Duke


class Exchange(Action):
    """
    Exchange - take 2 cards, put two back.
    """

    action_card = Ambassador
