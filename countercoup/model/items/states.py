from countercoup.model.state import State


class SelectAction(State):
    pass


class DecideToBlock(State):
    pass


class DecideToCounteract(State):
    pass


class SelectCardsToDiscard(State):
    pass


class GameFinished(State):
    pass


class DecideToBlockCounteract(State):
    pass


class SelectCardToLose(State):
    pass
