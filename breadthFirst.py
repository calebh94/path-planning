# Set of Path Planning algorithms with examples from (https://www.redblobgames.com/pathfinding)
from implementation import *
import timeit
from algorithms import *

# TIME THE PROCESSES AND THE NUMBER OF ITERATIONS UNTIL SOLUTION

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


# example_graph = SimpleGraph()
# example_graph.edges = {
#     'A': ['B'],
#     'B': ['A', 'C', 'D'],
#     'C': ['A'],
#     'D': ['E', 'A'],
#     'E': ['B']
# }
#
# breadth_first_search_1(example_graph, 'A')

g = SquareGrid(30,15)
g.walls = DIAGRAM1_WALLS
# parents = breadth_first_search(g, (8, 7), (17,2), debug=True)

# parents2 = dijkstra_search(diagram4, (1,4), (7,8), debug=True)

parents3 = a_star_search(diagram4, (1,4), (7,8), debug=True)

# draw_grid(g, width=2, point_to=parents, start=(8, 7), goal = (17,2))