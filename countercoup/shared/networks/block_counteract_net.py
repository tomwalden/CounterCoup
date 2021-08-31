from countercoup.shared.network import Network


class BlockCounteractNet(Network):
    """Network used for counteraction and blocking decisions"""

    outputs = [True, False]
