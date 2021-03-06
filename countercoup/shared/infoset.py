from countercoup.model.game import GameInfoSet
from countercoup.model.items.cards import Duke, Assassin, Ambassador, Captain, Contessa
from countercoup.model.items.actions import Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal
from numpy import array, zeros, int16


class Infoset:
    """
    Represents the information set at a given stage of the game, in a compacted form that can be fed into
    a neural network
    """

    def __init__(self, g: GameInfoSet):

        self.fixed_vector = self.__return_fixed_vector(g)
        self.history_vectors = self.__return_history_vectors(g)

    @staticmethod
    def __return_fixed_vector(g: GameInfoSet):
        """
        Generate a vector that serializes the non-history parts of the game state
        :param g: the Game object from the model
        :return: a Numpy array with the current players hand, the number of cards each player has, and
                 the current moves.
        """

        vec = []
        cards = [Duke, Assassin, Ambassador, Captain, Contessa]

        # First part of vector is the current players hand
        vec += [g.players[g.current_player].cards.count(x) for x in cards]

        # Next is the discard for the other players
        for play_num, player in enumerate(g.players):
            if play_num != g.current_player:
                vec += [player.discard.count(x) for x in cards]

        # Number of coins for the current player
        vec.append(g.players[g.current_player].coins)

        # Number of coins for all other players
        for play_num, player in enumerate(g.players):
            if play_num != g.current_player:
                vec.append(player.coins)

        # Current action encoding
        for action in [Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal]:
            vec.append(1 if g.current_action == action else 0)

        # Action, counteraction and attacking - first for current player
        vec.append(1 if g.current_player == g.action_player else 0)
        vec.append(1 if g.current_player == g.counteract_player else 0)
        vec.append(1 if g.current_player == g.attack_player else 0)

        # And for other players, including in-game flag (technically could just rely on discard, but having
        # it explicit should make network easier to train)
        for play_num, player in enumerate(g.players):
            if play_num != g.current_player:
                vec.append(1 if g.action_player == play_num else 0)
                vec.append(1 if g.counteract_player == play_num else 0)
                vec.append(1 if g.attack_player == play_num else 0)
                vec.append(1 if g.players[play_num].in_game else 0)

        return array([vec], dtype=int16)

    @staticmethod
    def __return_history_vectors(g: GameInfoSet):
        """
        Return the history vectors for each player, to be fed sequentially into a recurrent neural network
        such as a LSTM cell
        :param g: the Game object
        :return: a list of list of history vectors
        """

        flat_history = []

        actions = [Income, ForeignAid, Coup, Tax, Assassinate, Exchange, Steal]
        c_actions = [ForeignAid, Steal, Assassinate]

        blank_actions = [0, 0, 0, 0, 0, 0, 0]
        blank_c_actions = [0, 0, 0]

        for history in g.history:
            act_vec = [1 if history.action == x else 0 for x in actions] + blank_c_actions
            act_vec.append(1 if history.blocking_player is not None else 0)
            act_vec.append(1 if history.block_successful else 0)

            flat_history.append((history.action_player, act_vec))

            if history.counteracting_player is not None:
                c_act_vec = blank_actions + [1 if history.action == x else 0 for x in c_actions]
                c_act_vec.append(1 if history.counteract_block_player is not None else 0)
                c_act_vec.append(1 if history.counteract_block_successful else 0)

                flat_history.append((history.counteracting_player, c_act_vec))

        # First LSTM cell is for current player - ensure that an empty history has the
        # correct shape
        curr_history = [x[1] for x in flat_history if x[0] == g.current_player]
        vec = [zeros((1, 1, 12), dtype=int16) if not curr_history else array([curr_history], dtype=int16)]

        # All other players go in
        for play_num, player in enumerate(g.players):
            if play_num != g.current_player:
                play_history = [x[1] for x in flat_history if x[0] == play_num]
                vec.append(zeros((1, 1, 12), dtype=int16) if not play_history else array([play_history], dtype=int16))

        return vec
