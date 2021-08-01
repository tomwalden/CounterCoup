class GameInfoSet:
    """
    The parts of a Coup game that are only visible to the player
    """

    players = []
    history = []

    current_player = None
    attack_player = None
    counteract_player = None
    action_player = None

    current_action = None
    current_history = None

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
