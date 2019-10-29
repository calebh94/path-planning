from implementation import *
import timeit
import math

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
                priority = new_cost + heuristic(goal, node)
                frontier.put(node, priority)
                came_from[node] = current

    time2 = timeit.default_timer()
    if debug and fail is not True:
        print('Solution after %0.f search iterations and %.4f seconds!' % (cnt, (time2 - time1)))
        path = reconstruct_path(came_from, start=start, goal=goal)
        draw_grid(graph, width=2, path=path, start=start, goal=goal)
    elif fail:
        print('FAIL: Solution unable to be found after %0.f search iterations and %.4f seconds!' % (cnt, (time2 - time1)))
    return path


class GridWithManhattenDistance(SquareGrid):
    def __init__(self, width, height):
        super().__init__(width, height)
        self.weights = {}

    def cost(self, from_node, to_node):
        distance = math.sqrt( math.pow( (to_node[1] - from_node[1]), 2 ) + math.pow( (to_node[0] - from_node[0]), 2 ) )
        return distance



