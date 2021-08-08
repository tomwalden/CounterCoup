from countercoup.player.agent import Agent
from countercoup.model.game_info import GameInfoSet
from countercoup.model.player import Player
from countercoup.model.history import History
from countercoup.model.items.cards import Duke, Assassin, Ambassador, Captain, Contessa
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from countercoup.shared.tools import Tools
from random import sample
from socketio.client import Client

import re
import logging


class OnlinePlay:
    """
    Play CounterCoup using ChickenKoup (https://www.chickenkoup.com)
    Uses socket.io to communicate directly with the server
    """

    _card_map = {Duke: 'duke'
        , Assassin: 'assassin'
        , Ambassador: 'ambassador'
        , Captain: 'captain'
        , Contessa: 'contessa'}

    _r_card_map = {'duke': Duke
        , 'assassin': Assassin
        , 'ambassador': Ambassador
        , 'captain': Captain
        , 'contessa': Contessa}

    _action_map = {Income: 'income'
        , ForeignAid: 'foreign_aid'
        , Coup: 'coup'
        , Tax: 'tax'
        , Assassinate: 'assassinate'
        , Exchange: 'exchange'
        , Steal: 'steal'}

    _r_action_map = {'income': Income
        , 'foreign_aid': ForeignAid
        , 'coup': Coup
        , 'tax': Tax
        , 'assassinate': Assassinate
        , 'exchange': Exchange
        , 'steal': Steal}

    client = Client()
    nspace = None

    _log = logging.getLogger('socketio.client')

    def __init__(self, agent: Agent
                 , game_code: str
                 , uri: str = 'https://rocky-stream-49978.herokuapp.com'
                 , bot_name: str = 'CCoup'):

        self.agent = agent
        self.uri = uri
        self.bot_name = bot_name

        self.nspace = "/{game_code}".format(game_code=game_code)
        self.add_handlers()

        self.names = []
        self.game = GameInfoSet()
        self.complete_games = []

    def update_players(self, payload: []):
        """Update the game with player names and states"""

        self._log.info("Updating players")

        if not self.game.players:
            for p in payload:
                self.names.append(p["name"])
                self.game.players.append(Player())

        for p in payload:
            pl = self.game.players[self.names.index(p["name"])]
            self._log.info("{player} has {cards}, {coins} coins"
                           .format(player=p["name"], cards=p["influences"], coins=p["money"]))

            pl.cards = []
            for c in p["influences"]:
                pl.cards.append(self._r_card_map[c])

            pl.in_game = not p["isDead"]
            pl.coins = p["money"]

        self.game.current_player = self.names.index(self.bot_name)

    def select_action(self):
        """Select an action and send it to the server"""

        strategy = self.agent.get_action_strategy(self.game)
        action = Tools.select_from_strategy(strategy)

        self._log.info("Selecting action")
        self._log.info("Obtained strategy: {strat}"
                       .format(strat={(self._action_map[x[0]], x[1]): strategy[x] for x in strategy}))
        self._log.info("Selected action: {action}, attack: {attack}"
                       .format(action=self._action_map[action[0]], attack=action[1]))

        self.client.emit("g-actionDecision",
                        {"action": {"action": self._action_map[action[0]]
                        , "target": self.names[self.game.get_opponents()[action[1]]] if action[1] is not None else None
                        , "source": self.bot_name}}
                         , namespace=self.nspace)

    def decide_to_block(self, payload: []):
        """Decide to block or not"""

        if self.game.current_player == self.game.action_player:
            return

        if not self.game.get_curr_player().in_game:
            return

        self.game.current_action = self._r_action_map[payload["action"]]
        self.game.action_player = self.names.index(payload["source"])

        if self.game.current_action.attack_action:
            self.game.attack_player = self.names.index(payload["target"])

        strategy = self.agent.get_block_strategy(self.game)
        decision = Tools.select_from_strategy(strategy)

        self._log.info("Making block decision")
        self._log.info("Obtained strategy: {strat}".format(strat=strategy))
        self._log.info("Decision:{dec} blocked".format(dec=" not" if not decision else None))

        self.client.emit("g-challengeDecision"
                                 , {"action": payload
                                 , "isChallenging": decision
                                 , "challengee": payload["source"]
                                 , "challenger": self.bot_name}
                         , namespace=self.nspace)

    def choose_card_to_lose(self):
        """Select a card to lose, if we have to"""

        if not self.game.get_curr_player().in_game:
            return

        strategy = self.agent.get_lose_card_strategy(self.game)
        decision = Tools.select_from_strategy(strategy).card1

        self._log.info("Selecting card to lose")
        self._log.info("Obtained strategy: {strat}"
                       .format(strat={self._card_map[x.card1]: strategy[x] for x in strategy}))
        self._log.info("Decision: {dec}".format(dec=self._card_map[decision]))

        self.client.emit("g-chooseInfluenceDecision"
                               , {"influence": self._card_map[decision]
                                   , "playerName": self.bot_name}
                         , namespace=self.nspace)

    def reveal_card(self, payload: []):
        """When being blocked, either reveal the card we have or select a card to lose"""

        if not self.game.get_curr_player().in_game:
            return

        action = self._r_action_map[payload["action"]["action"]]

        # If we have the card for the action, show that. Otherwise, use the agent to decide which
        # card to lose
        if action.action_card in self.game.get_curr_player().cards:
            card = action.action_card
        else:
            strategy = self.agent.get_lose_card_strategy(self.game)
            card = Tools.select_from_strategy(strategy).card1

            self._log.info("Selecting card to lose")
            self._log.info("Obtained strategy: {strat}"
                           .format(strat={self._card_map[x.card1]: strategy[x] for x in strategy}))
            self._log.info("Decision: {dec}".format(dec=self._card_map[card]))

        self.client.emit("g-revealDecision", {"revealedCard": self._card_map[card]
            , "prevAction": payload["action"]
            , "counterAction": payload["counterAction"]
            , "challengee": payload["challengee"]
            , "challenger": payload["challenger"]
            , "isBlock": payload["isBlock"]}
                         , namespace=self.nspace)

    def choose_cards_to_discard(self, payload: []):
        """Decide which cards to lose when Exchanging"""

        if not self.game.get_curr_player().in_game:
            return

        cards = [self._r_card_map[p] for p in payload]
        self.game.get_curr_player().cards.append(cards[0])
        self.game.get_curr_player().cards.append(cards[1])

        strategy = self.agent.get_discard_strategy(self.game)
        decision = Tools.select_from_strategy(strategy)

        self._log.info("Selecting cards to discard")
        self._log.info("Obtained strategy: {strat}"
                       .format(strat={(self._card_map[h.card1], self._card_map[h.card2]): strategy[h] for h in strategy}))
        self._log.info("Cards to discard: {card1}, {card2}"
                       .format(card1=self._card_map[decision.card1], card2=self._card_map[decision.card2]))

        self.game.get_curr_player().cards.remove(decision.card1)
        self.game.get_curr_player().cards.remove(decision.card2)

        self.client.emit("g-chooseExchangeDecision",
                      {"playerName": self.bot_name
                          , "kept": [self._card_map[p] for p in self.game.get_curr_player().cards]
                          , "putBack": [self._card_map[decision.card1], self._card_map[decision.card2]]}
                         , namespace=self.nspace)

    def decide_to_counteract(self, payload: []):
        """Decide to counteract another players actions"""

        if self.game.current_player == self.game.action_player:
            return

        if not self.game.get_curr_player().in_game:
            return

        self.game.current_action = self._r_action_map[payload["action"]]
        self.game.action_player = self.names.index(payload["source"])

        if self.game.current_action.attack_action:
            self.game.attack_player = self.names.index(payload["target"])

        strategy = self.agent.get_counteract_strategy(self.game)
        decision = Tools.select_from_strategy(strategy)

        self._log.info("Making counteraction decision")
        self._log.info("Obtained strategy: {strat}".format(strat=strategy))
        self._log.info("Decision:{dec} blocked".format(dec=" not" if not decision else None))

        if decision:
            card_intersect = [c for c in self.game.get_curr_player().cards if c in self.game.current_action.c_action_cards]

            if not card_intersect:
                card = sample(self.game.current_action.c_action_cards, 1)[0]
            else:
                card = card_intersect[0]

            self.client.emit("g-blockDecision",
                    {"prevAction": payload
                     , "counterAction": {"counterAction": "block_{action}".format(action=payload["action"])
                                         , "claim": self._card_map[card]
                                         , "source": self.bot_name}
                        , "isBlocking": True
                        , "blockee": payload["source"]
                        , "blocker": self.bot_name}
                             , namespace=self.nspace)
        else:
            self.client.emit("g-blockDecision", {"action": payload, "isBlocking": False}
                             , namespace=self.nspace)

    def decide_to_block_counteract(self, payload: []):
        """Decide if to block a counteraction"""

        if payload["counterAction"]["source"] == self.bot_name:
            return

        if not self.game.get_curr_player().in_game:
            return

        self.game.current_action = self._r_action_map[payload["prevAction"]["action"]]
        self.game.action_player = self.names.index(payload["prevAction"]["source"])
        self.game.counteract_player = self.names.index(payload["counterAction"]["source"])

        if self.game.current_action.attack_action:
            self.game.attack_player = self.names.index(payload["prevAction"]["target"])

        strategy = self.agent.get_block_strategy(self.game)
        decision = Tools.select_from_strategy(strategy)

        self._log.info("Making counteraction block decision")
        self._log.info("Obtained strategy: {strat}".format(strat=strategy))
        self._log.info("Decision:{dec} blocked".format(dec=" not" if not decision else None))

        if decision:
            self.client.emit("g-blockChallengeDecision",
                        {"counterAction": payload["counterAction"]
                        , "prevAction": payload["prevAction"]
                        , "isChallenging": True
                        , "challengee": payload["counterAction"]["source"]
                        , "challenger": self.bot_name}
                             , namespace=self.nspace)
        else:
            self.client.emit("g-blockChallengeDecision", {"isChallenging": False}
                             , namespace=self.nspace)

    def add_log(self, ls: str):
        """Use the game log sent by the server to build up the game history"""

        self._log.info(ls)

        m = re.match("(\w+) used (\w+)( on (\w+))?", ls)
        if m is not None:
            self.game.current_history.action = self._r_action_map[m[2]]
            self.game.current_history.action_player = self.names.index(m[1])

            if m[3] is not None:
                self.game.current_history.attacking_player = self.names.index(m[4])

        m = re.match("(\w+)'s challenge on (\w+) (succeeded|failed)", ls)
        if m is not None:
            self.game.current_history.blocking_player = self.names.index(m[1])
            self.game.current_history.block_successful = True if m[3] == 'succeeded' else False

        m = re.match("(\w+) blocked (\w+)", ls)
        if m is not None:
            self.game.current_history.counteracting_player = self.names.index(m[1])

        m = re.match("(\w+)'s challenge on (\w+)'s block (succeeded|failed)", ls)
        if m is not None:
            self.game.current_history.counteract_block_player = self.names.index(m[1])
            self.game.current_history.counteract_block_successful = True if m[3] == 'succeeded' else False

    def update_current_player(self, payload: []):
        """Update the action player"""

        self.game.action_player = self.names.index(payload)

        if self.game.current_history is not None:
            self.game.history.append(self.game.current_history)

        self.game.current_history = History()

    def game_over(self, payload: str):
        """The game is over, create a new game"""

        self.game.winning_player = self.names.index(payload)
        self.complete_games.append(self.game)
        self.game = GameInfoSet()

    def leader_leaves(self, payload):
        """If everyone leaves, end the connection"""
        self.client.disconnect()

    def add_handlers(self):
        """Add handlers for socket.io"""

        self.client.on("g-updatePlayers", self.update_players, namespace=self.nspace)
        self.client.on("g-chooseAction", self.select_action, namespace=self.nspace)
        self.client.on("g-openChallenge", self.decide_to_block, namespace=self.nspace)
        self.client.on("g-chooseInfluence", self.choose_card_to_lose, namespace=self.nspace)
        self.client.on("g-openExchange", self.choose_cards_to_discard, namespace=self.nspace)
        self.client.on("g-openBlock", self.decide_to_counteract, namespace=self.nspace)
        self.client.on("g-addLog", self.add_log, namespace=self.nspace)
        self.client.on("g-openBlockChallenge", self.decide_to_block_counteract, namespace=self.nspace)
        self.client.on("g-updateCurrentPlayer", self.update_current_player, namespace=self.nspace)
        self.client.on("g-chooseReveal", self.reveal_card, namespace=self.nspace)
        self.client.on("g-gameOver", self.game_over, namespace=self.nspace)
        self.client.on("leaderDisconnect", self.leader_leaves, namespace=self.nspace)

    def run(self):
        """Run the client"""

        self.client.connect(self.uri, namespaces=[self.nspace])

        self.client.emit("setName", self.bot_name, namespace=self.nspace)
        self.client.emit("setReady", True, namespace=self.nspace)

        self.client.wait()
        self.client.disconnect()
