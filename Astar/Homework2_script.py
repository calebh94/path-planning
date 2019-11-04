from Astar.algorithms import *

# Define
grid = SquareGrid(width=8, height=7)
WALLS = [from_id_width(id, width=8) for id in [0,5,19,27,28,31,33,41,45,53]]

grid.walls = WALLS

start = (0,6)  # Column, Row (indexing from 0)
goal = (7,1)
wtd_grid = GridWithWeights(width=8, height=7)
wtd_grid.walls = [(0, 0), (5, 0), (3, 2), (3, 3), (4, 3), (7, 3), (1, 4), (1, 5), (5, 5), (5, 6)]
path = a_star_search(wtd_grid, start, goal, debug=True)

