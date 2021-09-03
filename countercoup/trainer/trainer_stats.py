class TrainerStats:
    """Holder for stats we pick up whilst training"""

    total_nodes_traversed = 0
    total_turns = 0
    total_terminal_nodes = 0
    total_hist_length = 0
    game_wins = 0
    game_loses = 0

    def get_data(self) -> []:
        return [self.total_nodes_traversed
            , self.total_turns
            , self.total_terminal_nodes
            , self.total_hist_length
            , self.game_wins
            , self.game_loses]

    def add_data(self, stats: []):
        self.total_nodes_traversed += stats[0]
        self.total_turns += stats[1]
        self.total_terminal_nodes += stats[2]
        self.total_hist_length += stats[3]
        self.game_wins += stats[4]
        self.game_loses += stats[5]



