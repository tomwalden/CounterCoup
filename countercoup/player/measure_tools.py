from statistics import stdev, mean
from countercoup.player.self_play import SelfPlay


class MeasureTools:

    @staticmethod
    def measure_win_rate(agents: [], chunk: int, count: int):

        stats = [[] for _ in agents]

        for _ in range(count):
            cp = SelfPlay(agents)

            for _ in range(chunk):
                cp.run()

            for n, x in enumerate(cp.tally):
                stats[n].append(x / chunk)

        results = []

        for s in stats:
            results.append((mean(s), stdev(s) / (count ** 0.5)))

        return results



