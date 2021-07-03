from countercoup.trainer.network import Network
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal


class BlockCounteractNet(Network):

    outputs = [True, False]
