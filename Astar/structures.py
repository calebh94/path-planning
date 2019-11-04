import collections


class SimpleGraph:
    def __init__(self):
        self.edges = {}

    def neighbors(self, id):
        return self.edges[id]


class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()

class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x,y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x,y) = id
        results = [(x+1, y), (x, y-1), (x-1,y), (x,y+1)]
        if (x+y) % 2 == 0: results.reverse() # Ordering for visual
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results
