from countercoup.model.action import Action
from countercoup.model.player import Player
from countercoup.model.card import Card
from countercoup.model.exceptions import IllegalMoveException, IllegalGameException
from countercoup.model.items.cards import Captain, Assassin, Contessa, Duke, Ambassador
from countercoup.model.items.states import SelectAction, DecideToBlock, DecideToCounteract, SelectCardsToDiscard\
    , GameFinished, DecideToBlockCounteract, SelectCardToLose
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from countercoup.model.history import History
from random import shuffle


class Game:
    """
    A complete model for the card game Coup
    """

    def __init__(self, num_of_players: int):

        if num_of_players < 2 or num_of_players > 6:
            raise IllegalGameException("Number of players must be between 2 and 6")

        self.players = []

        self.deck = [Duke, Contessa, Captain, Assassin, Ambassador] * 3
        shuffle(self.deck)

        for x in range(num_of_players):
            p = Player()
            p.cards = [self.deck.pop(0), self.deck.pop(0)]
            self.players.append(p)

        self.state = SelectAction
        self.current_player = 0
        self.action_player = 0
        self.attack_player = None
        self.counteract_player = None
        self.winning_player = None
        self.current_action = None
        self.counteract_card = None
        self.lose_card_state = None
        self.lose_card_player = None

        self.history = []
        self.current_history = None

    def __next_player(self, player_id):
        """
        Return the next active player
        :param player_id: the current player ID
        :return: the next active player ID. If all players are non-active, return None
        """

        for x in range(1, len(self.players) + 1):
            next_player = (player_id + x) % len(self.players)
            if self.players[next_player].in_game:
                return next_player

        return None

    def __play_action(self):
        """
        Play the current action, and either move to the next player or end the game
        """

        current_player = self.get_action_player()
        attack_player = None

        if self.attack_player is not None:
            attack_player = self.players[self.attack_player]

        if self.current_action == Exchange:
            self.state = SelectCardsToDiscard
            self.current_player = self.action_player
        else:
            self.state = SelectAction
            self.action_player = self.__next_player(self.action_player)
            self.current_player = self.action_player

        # Do the action depending on the action selected.
        if self.current_action == Income:
            current_player.coins += 1
        elif self.current_action == ForeignAid:
            current_player.coins += 2
        elif self.current_action == Tax:
            current_player.coins += 3
        elif self.current_action in [Coup, Assassinate]:
            self.__lose_card(self.attack_player)
        elif self.current_action == Steal:
            steal_value = min([2, attack_player.coins])
            current_player.coins += steal_value
            attack_player.coins -= steal_value
        elif self.current_action == Exchange:
            shuffle(self.deck)
            current_player.cards.append(self.deck.pop(0))
            current_player.cards.append(self.deck.pop(0))

        self.history.append(self.current_history)
        self.current_history = None
        self.current_action = None
        self.attack_player = None
        self.counteract_player = None

    def get_curr_player(self):
        """
        Get the current player
        :return: the current player
        """
        return self.players[self.current_player]

    def get_action_player(self):
        """
        Get the action player
        :return: the action player
        """
        return self.players[self.action_player]

    def get_counteract_player(self):
        """
        Get the counteract player
        :return: the counteract player
        """
        return self.players[self.counteract_player]

    def get_attack_player(self):
        """
        Get the player being attacked
        :return: the attacked player
        """
        return self.players[self.attack_player]

    def get_opponents(self):
        """
        Return a list of opponents to the current player
        :return: the list of opponents
        """

        output = []
        for n in range(len(self.players)):
            if n != self.current_player:
                output.append(n)

        return output

    def get_game_length(self):
        """
        Get the length of the game so far
        :return: the number of rounds in the game
        """
        return len(self.history)

    def select_action(self, action: Action, attack_player: int = None):
        """
        Play an action, and either play the action (if no-one can block or counteract it), or move
        to the block/counteract stage
        :param action: the action to be played
        :param attack_player: the player to be attacked, if playing Coup, Assassinate or Steal
        """

        if self.state != SelectAction:
            raise IllegalMoveException("Not at correct state to play this")

        # If we don't have enough coins, we can't play the action
        if self.get_action_player().coins - action.cost < 0:
            raise IllegalMoveException("Too few coins to play this action")

        # If we have ten or more coins, we can only Coup
        if self.get_action_player().coins >= 10 and action != Coup:
            raise IllegalMoveException("Too many coins, can only Coup")

        self.current_action = action

        self.current_history = History()
        self.current_history.action = action
        self.current_history.action_player = self.current_player

        if action.attack_action:
            if attack_player is None:
                raise IllegalMoveException("Need attacked player for this action")
            elif not self.players[attack_player].in_game:
                raise IllegalMoveException("Attacked player not in game")
            else:
                self.attack_player = attack_player
                self.current_history.attacking_player = attack_player

        # Player pays up even if the action is unsuccessful.
        self.get_curr_player().coins -= self.current_action.cost

        # If the current action cannot be blocked or counteracted, then jump straight to playing the action
        if action.c_action_cards == [] and action.action_card is None:
            self.__play_action()
        # If this is not a character action, then jump straight to the counteraction stage
        elif action.action_card is None:
            self.__determine_counteract()
        # Else move onto the blocking stage
        else:
            self.state = DecideToBlock
            self.current_player = self.__next_player(self.action_player)

    def decide_to_block(self, decision: bool):
        """
        Make the decision to block. If true, then either the current player or the blocking player is losing
        a card!
        :param decision: the decision to block or not.
        """

        # First check to ensure that we're in the right state
        if self.state != DecideToBlock:
            raise IllegalMoveException("Not at correct state to play this")

        # If the current player doesn't think that they have the card...
        if decision:
            action_card = self.current_action.action_card
            self.current_history.blocking_player = self.current_player

            if action_card in self.get_action_player().cards:
                # The block was unsuccessful - the challenging player loses a card
                self.current_history.block_successful = False

                # If the block was incorrect, then we need to discard the action card and draw a new card
                # The existing action will continue, however.
                self.deck.append(action_card)
                shuffle(self.deck)
                self.get_action_player().cards.remove(action_card)
                self.get_action_player().cards.append(self.deck.pop(0))

                self.__lose_card(self.current_player)
            else:
                # Block successful. Nice job!
                self.current_history.block_successful = True
                self.state = SelectAction

                self.__lose_card(self.action_player)

        else:
            self.current_player = self.__next_player(self.current_player)

            if self.current_player == self.action_player:
                self.__determine_counteract()

    def __determine_counteract(self):
        """
        Determine what to do when we move to the counteract stage
        """

        if not self.current_action.c_action_cards \
                or (self.attack_player is not None and not self.get_attack_player().in_game):
            # No counteraction for the current action, or the attacking player got knocked out - play the action
            self.__play_action()
        elif self.attack_player is not None:
            # Attacking actions are only counteracted by the player thats being attacked
            self.state = DecideToCounteract
            self.current_player = self.attack_player
        else:
            # Foreign aid can be blocked by anyone claiming a Duke, so go round everyone
            self.state = DecideToCounteract
            self.current_player = self.__next_player(self.action_player)

    def decide_to_counteract(self, decision: bool):
        """
        Decide if the current player should counteract or not
        :param decision: the decision to counteract
        """

        if self.state != DecideToCounteract:
            raise IllegalMoveException("Not at correct state to play this")

        if decision:
            # Current player has decided to counteract the action. Now check to see if someone wants
            # to block the counteraction
            self.state = DecideToBlockCounteract
            self.current_history.counteracting_player = self.current_player
            self.counteract_player = self.current_player
            self.current_player = self.__next_player(self.current_player)
        else:
            if self.attack_player is not None or self.__next_player(self.current_player) == self.action_player:
                self.counteract_player = None
                self.__play_action()
            else:
                self.current_player = self.__next_player(self.current_player)

    def decide_to_block_counteract(self, decision: bool):
        """
        Decide if the current player should block a declared counteraction
        :param decision: the decision to block the counteraction
        """

        if self.state != DecideToBlockCounteract:
            raise IllegalMoveException("Not at correct state to play this")

        if decision:
            self.current_history.counteract_player = self.current_player
            both_cards = [x for x in self.get_counteract_player().cards if x in self.current_action.c_action_cards]

            if both_cards:

                # TODO: we should allow the model to select the card to get rid of, if the counteracting player
                # has both the Ambassador and Captain
                c_action_card = both_cards[0]
                self.current_history.counteract_block_successful = False

                self.deck.append(c_action_card)
                shuffle(self.deck)
                self.get_counteract_player().cards.remove(c_action_card)
                self.get_counteract_player().cards.append(self.deck.pop(0))

                # History complete, can move it into list
                self.history.append(self.current_history)
                self.current_history = None

                self.__lose_card(self.current_player)

            else:
                self.state = DecideToCounteract
                self.current_history.counteract_block_successful = True

                # If the counteraction failed, see if the other players (if there are any) can counteract
                self.__lose_card(self.counteract_player)

        else:
            self.current_player = self.__next_player(self.current_player)

            if self.current_player == self.counteract_player:
                # No blocks, counteraction successful. Move to next player
                self.history.append(self.current_history)
                self.current_history = None

                self.state = SelectAction
                self.current_action = None
                self.counteract_player = None
                self.action_player = self.__next_player(self.action_player)
                self.current_player = self.action_player

    def __lose_card(self, player: int):
        """
        Begin process of player losing card. Either put game in state of picking card to lose, or lose only card
        and leaves the game
        :param player: the player thats losing a card
        """

        # Sometimes the player might still be losing a card after being knocked out of the game
        # So make sure that it's ignored if so.
        if self.players[player].in_game:
            self.lose_card_state = self.state
            self.lose_card_player = self.current_player
            self.current_player = player
            self.state = SelectCardToLose

            # If there is only one card.. well, that makes it easy doesn't it?
            if len(self.players[player].cards) == 1:
                self.select_card_to_lose(self.players[player].cards[0])

    def __determine_winner(self):
        """
        Determine if there is a winner
        :return: the winner! None if there isn't a winner.
        """

        players_in_game = [i for i, j in enumerate(self.players) if j.in_game]

        if len(players_in_game) == 1:
            return players_in_game[0]
        else:
            return None

    def select_card_to_lose(self, card: Card):
        """
        If the player is losing a card, determine which card to lose. Also used to determine if the game is finished
        :param card: the card that the player is discarding
        """

        if self.state != SelectCardToLose:
            raise IllegalMoveException("Not at correct state to play this")

        if card not in self.get_curr_player().cards:
            raise IllegalMoveException("You can't lose a card you don't already have")

        self.get_curr_player().cards.remove(card)
        self.get_curr_player().discard.append(card)

        # If the cards are empty for the player, the player is out of the game
        if not self.get_curr_player().cards:
            self.get_curr_player().in_game = False

        # If all the players except one are out, that player has won!
        winner = self.__determine_winner()
        if winner is not None:
            self.state = GameFinished
            self.winning_player = winner
            self.history.append(self.current_history)
            self.current_history = None
        else:
            if self.lose_card_state == DecideToBlock:
                self.__determine_counteract()
            elif self.lose_card_state in [SelectAction, DecideToBlockCounteract]:
                self.action_player = self.__next_player(self.action_player)
                self.current_player = self.action_player
                self.state = SelectAction
                self.counteract_player = None
            elif self.lose_card_state == DecideToCounteract:
                if self.attack_player is not None or self.__next_player(self.counteract_player) == self.action_player:
                    self.__play_action()
                else:
                    self.state = DecideToCounteract
                    self.current_player = self.__next_player(self.counteract_player)
            else:
                self.state = self.lose_card_state
                self.current_player = self.lose_card_player

    def select_cards_to_discard(self, card1: Card, card2: Card):
        """
        When playing Ambassador, select the cards that the player is NOT keeping in their hand
        :param card1: the first card to discard
        :param card2: the second card to discard
        """

        if self.state != SelectCardsToDiscard:
            raise IllegalMoveException("Not at correct state to play this")

        if card1 not in self.get_action_player().cards or card2 not in self.get_action_player().cards:
            raise IllegalMoveException("You can't lose a card you don't already have")

        self.get_action_player().cards.remove(card1)
        self.deck.append(card1)

        self.get_action_player().cards.remove(card2)
        self.deck.append(card2)

        self.state = SelectAction
        self.action_player = self.__next_player(self.action_player)
        self.current_player = self.action_player
