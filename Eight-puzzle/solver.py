import numpy as np
import time


def find(l, elem):
    for row, i in enumerate(l):
        try:
            column = i.index(elem)
        except ValueError:
            continue
        return row, column
    return -1


class Node:
    def __init__(self, data, level, fval):
        """ Initialize the node with the data, level of the node and the calculated fvalue """
        self.data = data
        self.level = level
        self.fval = fval

    def generate_child(self):
        """ Generate child nodes from the given node by moving the blank space
            either in the four directions {up,down,left,right} """
        x, y = self.find(self.data, '_')
        """ val_list contains position values for moving the blank space in either of
            the 4 directions [up,down,left,right] respectively. """
        val_list = [[x, y - 1], [x, y + 1], [x - 1, y], [x + 1, y]]
        children = []
        for i in val_list:
            child = self.shuffle(self.data, x, y, i[0], i[1])
            if child is not None:
                child_node = Node(child, self.level + 1, 0)
                children.append(child_node)
        return children

    def shuffle(self, puz, x1, y1, x2, y2):
        """ Move the blank space in the given direction and if the position value are out
            of limits the return None """
        if x2 >= 0 and x2 < len(self.data) and y2 >= 0 and y2 < len(self.data):
            temp_puz = []
            temp_puz = self.copy(puz)
            temp = temp_puz[x2][y2]
            temp_puz[x2][y2] = temp_puz[x1][y1]
            temp_puz[x1][y1] = temp
            return temp_puz
        else:
            return None

    def copy(self, root):
        """ Copy function to create a similar matrix of the given node"""
        temp = []
        for i in root:
            t = []
            for j in i:
                t.append(j)
            temp.append(t)
        return temp

    def find(self, puz, x):
        """ Specifically used to find the position of the blank space """
        for i in range(0, len(self.data)):
            for j in range(0, len(self.data)):
                if puz[i][j] == x:
                    return i, j


class Puzzle:
    def __init__(self, size):
        """ Initialize the puzzle size by the specified size,open and closed lists to empty """
        self.n = size
        self.open = []
        self.closed = []

    def accept(self):
        """ Accepts the puzzle from the user """
        puz = []
        for i in range(0, self.n):
            temp = input().split(" ")
            puz.append(int(temp))
        return puz

    def f(self, start, goal):
        """ Heuristic Function to calculate hueristic value f(x) = h(x) + g(x) """
        return self.h(start.data, goal) + start.level

    def h(self, start, goal):
        """ Calculates the different between the given puzzles """
        # Choose Heuristic
        return self.h1(start, goal)

    def h1(self, start, goal):
        """ Heuristic - Number of Misplaced Tiles """
        temp = 0
        for i in range(0, self.n):
            for j in range(0, self.n):
                if start[i][j] != goal[i][j] and start[i][j] != '_':
                    temp += 1
        return temp

    def h2(self, start, goal):
        """ Heuristic - Total Manhattan Distance for each tile """
        temp = 0
        for i in range(0, self.n):
            for j in range(0, self.n):
                # For each value, find the index of the goal position
                (x, y) = find(goal, start[i][j])
                # Calculate manhatten distance
                dis = abs(x - i) + abs(y - j)
                temp = temp + dis
        return temp

    def process(self, type='input'):
        if type == 'input':
            """ Accept Start and Goal Puzzle state"""
            print("Enter the start state matrix \n")
            start = self.accept()
            print("Enter the goal state matrix \n")
            goal = self.accept()
        elif type == 'example':
            # start = [['4', '5', '6'], ['1', '7', '_'], ['8', '3', '2']]  # B (COULDNT SOLVE??)
            # start = [['4','5','1'],['6','7','_'],['8','3','2']] # A
            start = [['2', '8', '3'], ['1', '6', '4'], ['7', '_', '5']]  # A example
            # goal = [['_', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]  # GOAL STATE B
            goal = [['1','2','3'],['8','_','4'],['7','6','5']] # GOAL STATE A
        elif type == 'homework':
            start = [['5', '4', '_'], ['6', '1', '8'], ['7', '3', '2']]
            goal = [['_', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]  # GOAL STATE B
        elif type == "random":
            random_start = np.random.choice(['_','1','2','3','4','5','6','7','8'],size=9, replace=False)
            (a, b, c) = np.split(random_start, 3)
            start = [a.tolist(), b.tolist(), c.tolist()]
            goal = [['_', '1', '2'], ['3', '4', '5'], ['6', '7', '8']]  # B
        else:
            return 0

        # Check whether the start and goal are in the same disjoint sets (A or B)
        start_set = self.setFromState(start)
        goal_set = self.setFromState(goal)

        if start_set != goal_set:
            print('The start puzzle are not in the same state set!  Impossible to solve!')
            return 0

        self.f_limit = 500

        start = Node(start, 0, 0)
        start.fval = self.f(start, goal)
        """ Put the start node in the open list"""
        self.open.append(start)
        self.current_cost = 0
        self.current_depth = 0
        self.total_nodes = 0
        print("\n\n")
        while len(self.open) > 0:
            cur = self.open[0]

            print("")
            print("  | ")
            print("  | ")
            print(" \\\'/ \n")
            for i in cur.data:
                for j in i:
                    print(j, end=" ")
                print("")
            """ If the difference between current and goal node is 0 we have reached the goal node"""
            self.current_cost = self.h(cur.data, goal)
            self.current_depth = cur.level
            self.total_nodes = len(self.closed)
            print("The current cost to go is: " + str(self.current_cost))
            print("The current depth of tree is: " + str(self.current_depth))
            print("Total nodes expanded: " + str(self.total_nodes))
            if self.current_cost == 0:
                break
            for i in cur.generate_child():
                # If node already seen then throw away
                repeat = False
                for elem in self.closed:
                    if elem.data == i.data:
                        repeat = True
                if repeat is False:
                    i.fval = self.f(i, goal)
                    # Pruning
                    if i.fval <= self.f_limit:
                        self.open.append(i)
            self.closed.append(cur)
            del self.open[0]

            """ sort the open list based on f value """
            self.open.sort(key=lambda x: x.fval, reverse=False)
        else:
            print('No solution found, it may be that a limit was set')

    def setFromState(self, puzzle):
        # Vectorize
        grid = np.array(puzzle)
        vector = grid.flatten('C')
        # loop through and cnt
        total_cnt = 0
        less_cnt = 0
        for i in range(0, 9):
            curr_val = vector[i]
            if curr_val == '_':
                continue
            else:
                curr_val = int(curr_val)
            for j in range(i, 9):
                next_val = vector[j]
                if next_val == '_':
                    continue
                else:
                    next_val = int(next_val)

                less_cnt = 0
                if 0 < next_val < curr_val:
                    less_cnt = less_cnt + 1
                    total_cnt = less_cnt + total_cnt
        # print(total_cnt)
        (quo, mod) = divmod(total_cnt, 2)
        if mod == 1:
            group = 'A'
        elif mod == 0:
            group = 'B'
        return group

t1 = time.time()
puz = Puzzle(3)
puz.process('homework')
t2 = time.time()
print('Time taken: %.2f seconds' % (t2-t1))
