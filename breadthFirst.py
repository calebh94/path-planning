# Set of Path Planning algorithms with examples from (https://www.redblobgames.com/pathfinding)
from implementation import *
import time

# TIME THE PROCESSES AND THE NUMBER OF ITERATIONS UNTIL SOLUTION

def breadth_first_search(graph, start, goal):
    frontier = Queue()
    frontier.put(start)
    came_from = {}
    came_from[start] = None

    while not frontier.empty():
        current = frontier.get()

        if current == goal:
            break

        for next in graph.neighbors(current):
            if next not in came_from:
                frontier.put(next)
                came_from[next] = current
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
parents = breadth_first_search(g, (8, 7), (17,2))
draw_grid(g, width=2, point_to=parents, start=(8, 7), goal = (17,2))
