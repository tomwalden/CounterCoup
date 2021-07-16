from random import randint


class Memory:
    """
    Memory for the trainer, utilising reservoir sampling
    """

    def __init__(self, size: int):
        self.size = size
        self.counter = 0
        self.data = []

    def add(self, item):
        """
        Add to the reservoir
        :param item: the item being added
        """

        if self.counter < self.size:
            self.data.append(item)
        else:
            i = randint(0, self.counter)
            if i < self.size - 1:
                self.data[i] = item
        self.counter += 1

    def add_bulk(self, items: []):
        """
        Add many items to the reservoir
        :param items: the [item] being added
        """
        for x in items:
            self.add(x)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
