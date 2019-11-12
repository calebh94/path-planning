import numpy as np

# For implementation, I need
# 1) Value Iteration function
# 2) State and Action space

class BoardState:
    def __init__(self, id, m, n):
        self.id = id  # Name
        self.m = m  # rows
        self.n = n  # columns
        self.walls = []  # indices of wall locations
        self.dist = []  # w, disturbance
        self.a = []  # actions
        self.board = np.empty([self.m, self.n], dtype=str)  # {row, col}
        self.values = np.empty([self.m, self.n])
        self.dist = np.zeros([self.m, self.n])

    def initialize(self, init_value=0):
        # initialize initial values in board state
        # for i in range(0, self.m):
        #     for j in range(0,self.n):
        self.values.fill(init_value)
        self.board.fill('O')
        # print(self.values)

    def add_actions(self, act_set):
        self.a = act_set

    def add_disturbance(self, directions):
        # submit as a vector of the row values
        directs = np.array(directions)
        self.dist = np.reshape(directs,(self.m, self.n))
        # print(self.dist)

    def add_walls(self, indices):
        for i in range(0, len(indices)):
            row = indices[i][0]
            col = indices[i][1]
            self.board[row][col] = 'X'
        # print(self.board)

    def search(self, start, goal):
        print('SEARCHING')

    def get_states(self, here, inds=False):
        x = here[0]
        y = here[1]
        dne_list = []
        states = [[x-1,y],[x,y+1],[x+1,y],[x,y-1]]  # Actions: {U,R,D,L}
        for i in range(3,-1,-1):
            # Check outside bounds of board
            if states[i][0] < 0 or states[i][0] >= self.m:
                del states[i]
                dne_list.append(i)
            elif states[i][1] < 0 or states[i][1] >= self.n:
                del states[i]
                dne_list.append(i)
            # PROBLEM FORMULATIONS JUST ADDS HIGH COST FOR OBSTACLES
            # elif self.board[states[0],states[0]] == 'X'
            #     del states[i]
            else:
                continue
        if inds:
            return (states,dne_list)
        else:
            return states

    def reward(self, indices):
        x = indices[0]
        y = indices[1]
        reward = 0
        if (x,y) == self.goal:
            reward = 10
        elif self.board[x,y] == 'X':
            reward = -10
        else:
            reward = -1
        return reward

    def ifopposite(self, act, dist):
        if act == 'U':
            if dist == 'D':
                return True
        elif act == 'D':
            if dist == 'U':
                return True
        elif act == 'L':
            if dist == 'R':
                return True
        elif act == 'R':
            if dist == 'L':
                return True
        else:
            return False

    def get_values(self, location):
        (states, inds) = self.get_states(location, inds=True)
        probs = []
        values = []
        for i in range(0,len(states)):
            if i in inds:
                continue

            if self.a[i] == self.dist[location[0],location[1]]:
                probs = [0.75,0.1,0.1,0.05]
            elif self.ifopposite(self.a[i], self.dist[location[0],location[1]]):
                probs = [0.05,0.2,0.2,0.55]
            else:
                probs = [0.2,0.6,0.05,0.15]

            for j in inds:
                del probs[j]

            value = 0
            for t in range(0, len(probs)):
                value += self.values[states[t][0],states[t][1]]*probs[t]

            values.append(value)
            return values

    def max_value(self, start):
        values = []
        # for act in self.a:
            # sum over all possibilities
            # Cases, Action = dist, +90 from dist, -90 from dist, -180 from dist
        values = self.get_values(start)
        return max(values)

    def value_iteration(self, start, goal, discount=1.00, epsilon=0.01, steps=1000):
        self.start = [start[0],start[1]]
        self.goal = [goal[0],goal[1]]
        k = 0
        update_list = [self.start]
        new_states = [self.start]
        new_states_new = []
        delta = 0
        while k < steps:
            # [update_list.append(x) for x in new_states if x not in update_list]
            new_states_new = []
            for p in range(0, len(new_states)):
                if new_states[p] not in update_list:
                    update_list.append(new_states[p])
                add_states = self.get_states(new_states[p], inds=False)
                for q in range(0, len(add_states)):
                    new_states_new.append(add_states[q])
            # new_states_new = [self.get_states(n, inds=False) for n in new_states]
            # new_states = []
            for h in range(0, len(new_states_new)):
                if new_states_new[h] in update_list:
                    continue
                else:
                    new_states.append(new_states_new[h])
            for elem in update_list:
                prev_val = self.values[elem[0],elem[1]]
                self.values[elem[0],elem[1]] += self.reward(elem) + \
                discount*self.max_value(elem)

                if abs(self.values[elem[0],elem[1]] - prev_val) > delta:
                    delta = abs(self.values[elem[0],elem[1]] - prev_val)

                if delta < epsilon*(1-discount)/discount:
                    break
            k += 1
        print('Stopped at iteration ' + str(k))
        print(self.values.round())
        return self.values


if __name__ == "__main__":
    mdp_board = BoardState('mdp', 7, 8)
    mdp_board.initialize(0)
    mdp_board.add_actions(['U','R','D','L'])
    mdp_board.add_disturbance(('O','D','D','R','U','O',
                               'U','D','L','R','R','D','L',
                               'L','L','O','U','R','D','O',
                               'D','D','L','L','U','L','D','O',
                               'O','R','R','O','R','O','U','D',
                               'R','D','U','L','U','O','R','U',
                               'U','O','U','D','O','R','R','U',
                               'L','O','U','L'))
    mdp_board.add_walls([[0,0],[0,5],[2,3],[3,3],[3,4],[3,7],
                         [4,1],[5,1],[5,5],[6,5]])
    # mdp_board.search((7,1),(2,8))
    start = (6,0)  # (7,1)
    goal = (1,7)   # (2,8)
    mdp_board.value_iteration(start, goal, discount=0.95, epsilon=0.001, steps=100)


