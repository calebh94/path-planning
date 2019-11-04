# Algorithms and Data Structures for AI for AE Class, Georgia Tech, 2019
# Modifications by Caleb Harris
# Original code from https://www.redblobgames.com/pathfinding/a-star/
# Copyright 2014 Red Blob Games <redblobgames@gmail.com>
#
# Feel free to use this code in your own projects, including commercial projects
# License: Apache v2.0 <http://www.apache.org/licenses/LICENSE-2.0.html>
import timeit
import math
import heapq
import collections


class Queue:
    def __init__(self):
        self.elements = collections.deque()

    def empty(self):
        return len(self.elements) == 0

    def put(self, x):
        self.elements.append(x)

    def get(self):
        return self.elements.popleft()


# utility functions for dealing with square grids
def from_id_width(id, width):
    return (id % width, id // width)


def draw_tile(graph, id, style, width):
    r = "."
    if 'number' in style and id in style['number']: r = "%d" % style['number'][id]
    if 'point_to' in style and style['point_to'].get(id, None) is not None:
        (x1, y1) = id
        (x2, y2) = style['point_to'][id]
        if x2 == x1 + 1: r = ">"
        if x2 == x1 - 1: r = "<"
        if y2 == y1 + 1: r = "v"
        if y2 == y1 - 1: r = "^"
    if 'start' in style and id == style['start']: r = "A"
    if 'goal' in style and id == style['goal']: r = "Z"
    if 'path' in style and id in style['path']: r = "@"
    # if id in graph.walls: r = "#" * width
    if id in graph.walls: r = "#"
    return r


def draw_grid(graph, width=2, **style):
    for y in range(graph.height):
        for x in range(graph.width):
            print("%%-%ds" % width % draw_tile(graph, (x, y), style, width), end="")
        print()


class PriorityQueue:
    def __init__(self):
        self.elements = []

    def empty(self):
        return len(self.elements) == 0

    def put(self, item, priority):
        heapq.heappush(self.elements, (priority, item))

    def get(self):
        return heapq.heappop(self.elements)[1]


class SquareGrid:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = []

    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    def passable(self, id):
        return id not in self.walls

    def neighbors(self, id):
        (x, y) = id
        results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1)]
        if (x + y) % 2 == 0: results.reverse()  # aesthetics
        results = filter(self.in_bounds, results)
        results = filter(self.passable, results)
        return results


class GridWithWeights(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        return 1


def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)  # optional
    path.reverse()  # optional
    return path


def euclidean_heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return math.sqrt(abs(x1 - x2)**2 + abs(y1 - y2)**2)


def breadth_first_search(graph, start, goal, debug = False):
    time1 = timeit.default_timer()  # Timer Start

    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    cnt = 0
    while not frontier.empty():
        current = frontier.get()
        cnt = cnt + 1  # Counter for iteration until goal is reached
        if current == goal:
            break

        for node in graph.neighbors(current):
            if node not in came_from:
                frontier.put(node)
                came_from[node] = current
    time2 = timeit.default_timer()
    if debug:
        print('Solution after %0.f search iterations and %.4f seconds!' % (cnt, (time2-time1)))
        draw_grid(graph, width=2, point_to=came_from, start=start, goal=goal)
    return came_from


def dijkstra_search(graph, start, goal, debug=False):
    time1 = timeit.default_timer()  # Timer Start

    frontier = PriorityQueue()
    frontier.put(start,0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    cnt = 0
    fail = True
    path = [start]
    while not frontier.empty():
        current = frontier.get()
        cnt = cnt + 1  # Counter for iteration until goal is reached

        if current == goal:
            fail = False
            break

        for node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, node)
            if node not in cost_so_far or new_cost < cost_so_far[node]:
                cost_so_far[node] = new_cost
                priority = new_cost
                frontier.put(node, priority)
                came_from[node] = current
    time2 = timeit.default_timer()
    if debug and fail is not True:
        print('Solution after %0.f search iterations and %.4f seconds!' % (cnt, (time2-time1)))
        path = reconstruct_path(came_from, start=start, goal=goal)
        draw_grid(graph, width=2, path=path, start=start, goal=goal)
        draw_grid()
    elif fail:
        print('FAIL: Solution unable to be found after %0.f search iterations and %.4f seconds!' % (cnt, (time2 - time1)))
    return path


def a_star_search(graph, start, goal, debug=False):
    time1 = timeit.default_timer()  # Timer Start

    frontier = PriorityQueue()
    frontier.put(start, 0)
    came_from = {}
    cost_so_far = {}
    came_from[start] = None
    cost_so_far[start] = 0

    cnt = 0
    fail = True
    path = [start]
    while not frontier.empty():
        current = frontier.get()
        cnt = cnt + 1  # Counter for iteration until goal is reached

        if current == goal:
            fail = False
            break

        for node in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current,node)
            if node not in cost_so_far or new_cost < cost_so_far[node]:
                cost_so_far[node] = new_cost
                priority = new_cost + euclidean_heuristic(goal, node)
                frontier.put(node, priority)
                came_from[node] = current

    time2 = timeit.default_timer()
    if debug and fail is not True:
        print('Solution after %0.f search iterations and %.4f microseconds!' % (cnt, (time2 - time1)*1000000))
        path = reconstruct_path(came_from, start=start, goal=goal)
        draw_grid(graph, width=3, path=path, start=start, goal=goal)
        print('\n')
        draw_grid(graph, width=3, number=cost_so_far, start=start, goal=goal)
    elif fail:
        print('FAIL: Solution unable to be found after %0.f search iterations and %.4f seconds!' % (cnt, (time2 - time1)))
    return path



