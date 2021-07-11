from countercoup.shared.network import Network
from countercoup.model.hand import Hand
from countercoup.model.items.cards import Duke, Assassin, Ambassador, Captain, Contessa


class LoseNet(Network):

    outputs = [Hand([Duke, Duke]), Hand([Duke, Assassin]), Hand([Duke, Ambassador]), Hand([Duke, Captain]),
               Hand([Duke, Contessa]), Hand([Assassin, Assassin]), Hand([Assassin, Ambassador]),
               Hand([Assassin, Captain]), Hand([Assassin, Contessa]), Hand([Ambassador, Ambassador]),
               Hand([Ambassador, Captain]), Hand([Ambassador, Contessa]), Hand([Captain, Captain]),
               Hand([Captain, Contessa]), Hand([Contessa, Contessa]), Hand([Duke]), Hand([Assassin]),
               Hand([Ambassador]), Hand([Captain]), Hand([Contessa])]
