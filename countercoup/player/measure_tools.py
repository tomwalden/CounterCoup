from statistics import stdev, mean
from countercoup.player.coup_play import CoupPlay


class MeasureTools:

    @staticmethod
    def measure(agents: [], chunk: int, count: int):

        stats = [[] for _ in agents]

        for _ in range(count):
            cp = CoupPlay(agents)

            for _ in range(chunk):
                cp.run()

            for n, x in enumerate(cp.tally):
                stats[n].append(x)

        results = []

        for s in stats:
            results.append((mean(s), stdev(s) / (count ** 0.5)))

        return results



