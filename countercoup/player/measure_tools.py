from statistics import stdev, mean
from countercoup.player.self_play import SelfPlay


class MeasureTools:
    """Tools used to measure the effectiveness of the agents"""

    @staticmethod
    def measure_win_rate(agents: [], chunk: int, count: int):
        """
        Measure the win rate of agents along with the error
        :param agents: the list of agents to play against
        :param chunk: the chunk size
        :param count: the number of chunks
        :return: a list of tuples containing the win rate and error for each agent
        """

        stats = [[] for _ in agents]

        # Play each chunk
        for _ in range(count):
            cp = SelfPlay(agents)

            for _ in range(chunk):
                cp.run()

            for n, x in enumerate(cp.tally):
                stats[n].append(x / chunk)

        results = []

        # Calculate the mean and the error for each agent
        for s in stats:
            results.append((mean(s), stdev(s) / (count ** 0.5)))

        return results



