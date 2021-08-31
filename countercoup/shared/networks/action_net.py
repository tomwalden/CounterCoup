from countercoup.shared.network import Network
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal


class ActionNet(Network):
    """Network used for action decisions"""

    outputs = [(Income, None), (ForeignAid, None), (Coup, 0), (Coup, 1), (Coup, 2), (Tax, None), (Assassinate, 0)
               , (Assassinate, 1), (Assassinate, 2), (Exchange, None), (Steal, 0), (Steal, 1), (Steal, 2)]
