from countercoup.trainer.network import Network
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal


class ActionNet(Network):

    outputs = [(Income, None), (ForeignAid, None), (Coup, 1), (Coup, 2), (Coup, 3), (Tax, None), (Assassinate, 1)
               , (Assassinate, 2), (Assassinate, 3), (Exchange, None), (Steal, 1), (Steal, 2), (Steal, 3)]
