class TrainerStats:
    """Holder for stats we pick up whilst training"""

    total_nodes_traversed = 0
    total_turns = 0
    total_terminal_nodes = 0

    def get_data(self) -> []:
        return [self.total_nodes_traversed, self.total_turns, self.total_terminal_nodes]

    def add_data(self, stats: []):
        self.total_nodes_traversed += stats[0]
        self.total_turns += stats[1]
        self.total_terminal_nodes += stats[2]
