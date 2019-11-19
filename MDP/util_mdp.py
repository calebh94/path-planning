import numpy as np
import matplotlib.pyplot as plt
import cv2


class GridWorldMDP:

    # up, right, down, left
    _direction_deltas = [
        (-1, 0),
        (0, 1),
        (1, 0),
        (0, -1),
    ]
    _num_actions = len(_direction_deltas)

    def __init__(self,
                 reward_grid,
                 terminal_mask,
                 obstacle_mask,
                 disturbances,
                 action_probabilities,
                 no_action_probability):

        self._reward_grid = reward_grid
        self._terminal_mask = terminal_mask
        self._obstacle_mask = obstacle_mask
        self._disturbances = disturbances
        self._T = self._create_transition_matrix(
            action_probabilities,
            no_action_probability,
            disturbances,
            obstacle_mask)
        (a,b,c,d,e,f) = self._T.shape
        self._T2 = np.zeros((a,b,c,e,f), dtype=float)



    @property
    def shape(self):
        return self._reward_grid.shape

    @property
    def size(self):
        return self._reward_grid.size

    @property
    def reward_grid(self):
        return self._reward_grid


    def run_value_iterations(self, discount=1.0,
                             iterations=10, epsilon=0.001):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

        utility_grid = np.zeros_like(self._reward_grid)
        for i in range(iterations):
            utility_grid = self._value_iteration(utility_grid=utility_grid)
            policy_grids[:, :, i] = self.best_policy(utility_grid)
            utility_grids[:, :, i] = utility_grid

            # delta = 0
            # if i > 1:
            #     utility_delta = abs(utility_grids[:,:,i] - utility_grids[:,:,i-1])
            #     delta_i = np.max(utility_delta)
            #     if delta_i > delta:
            #         delta = delta_i
            #     if delta < epsilon*(1-discount)/discount:
            #         print('Converged at iteration ' + str(i))
            #         return policy_grids, utility_grids

        return policy_grids, utility_grids


    def run_policy_iterations(self, discount=1.0,
                              iterations=10):
        utility_grids, policy_grids = self._init_utility_policy_storage(iterations)

        policy_grid = np.random.randint(0, self._num_actions,
                                        self.shape)
        utility_grid = self._reward_grid.copy()

        for i in range(iterations):
            policy_grid, utility_grid = self._policy_iteration(
                policy_grid=policy_grid,
                utility_grid=utility_grid
            )
            policy_grids[:, :, i] = policy_grid
            utility_grids[:, :, i] = utility_grid
        return policy_grids, utility_grids

    # def generate_experience(self, current_state_idx, action_idx):
    #     sr, sc = self.grid_indices_to_coordinates(current_state_idx)
    #     next_state_probs = self._T[sr, sc, action_idx, :, :].flatten()
    #
    #     next_state_idx = np.random.choice(np.arange(next_state_probs.size),
    #                                       p=next_state_probs)
    #
    #     return (next_state_idx,
    #             self._reward_grid.flatten()[next_state_idx],
    #             self._terminal_mask.flatten()[next_state_idx])

    def grid_indices_to_coordinates(self, indices=None):
        if indices is None:
            indices = np.arange(self.size)
        return np.unravel_index(indices, self.shape)

    def grid_coordinates_to_indices(self, coordinates=None):
        # Annoyingly, this doesn't work for negative indices.
        # The mode='wrap' parameter only works on positive indices.
        if coordinates is None:
            return np.arange(self.size)
        return np.ravel_multi_index(coordinates, self.shape)

    def best_policy(self, utility_grid):
        M, N = self.shape
        return np.argmax((utility_grid.reshape((1, 1, 1, M, N)) * self._T2)
                         .sum(axis=-1).sum(axis=-1), axis=2)

    def _init_utility_policy_storage(self, depth):
        M, N = self.shape
        utility_grids = np.zeros((M, N, depth))
        policy_grids = np.zeros_like(utility_grids)
        return utility_grids, policy_grids

    def _create_transition_matrix(self,
                                  action_probabilities,
                                  no_action_probability,
                                  disturbances,
                                  obstacle_mask):
        M, N = self.shape

        T = np.zeros((M, N, self._num_actions, self._num_actions, M, N))   # M,N,A,D,M,N

        r0, c0 = self.grid_indices_to_coordinates()

        T[r0, c0, :, :, r0, c0] += no_action_probability

        for action in range(self._num_actions):
            for disturb in range(self._num_actions):
                case = (action - disturb) % self._num_actions
                probs = action_probabilities[case][1]
                for direction in range(self._num_actions):
                    step = (action + direction) % self._num_actions
                    dr, dc = self._direction_deltas[step]
                    r1 = np.clip(r0 + dr, 0, M - 1)
                    c1 = np.clip(c0 + dc, 0, N - 1)
                    temp_mask = obstacle_mask[r1, c1].flatten()
                    r1[temp_mask] = r0[temp_mask]
                    c1[temp_mask] = c0[temp_mask]
                    index = (case+direction) % self._num_actions
                    T[r0, c0, action, disturb, r1, c1] += probs[index]

        terminal_locs = np.where(self._terminal_mask.flatten())[0]
        T[r0[terminal_locs], c0[terminal_locs], :, :, :, :] = 0

        # Fixing matrix
        T[6,0,0,:,:,:] = 0
        T[6, 0, 0, :, 5, 0] = 1.0
        T[6, 0, 1, :, 6, 1] = 1.0
        return T

    def _value_iteration(self, utility_grid, discount=1.0):
        out = np.zeros_like(utility_grid)
        M, N = self.shape
        for i in range(M):
            for j in range(N):
                out[i, j] = self._calculate_utility((i, j),
                                                    discount,
                                                    utility_grid)
        return out

    def _policy_iteration(self, *, utility_grid,
                          policy_grid, discount=1.0):
        r, c = self.grid_indices_to_coordinates()

        M, N = self.shape

        # Need to form T2 matrix here (not efficient but)
        for i in range(0,M):
            for j in range(0,N):
                row = i
                col = j
                dist = self._disturbances[row][col]
                self._T2[row, col, :, :, :] = self._T[row, col, :,dist,:,:]

        utility_grid = (
            self._reward_grid +
            discount * ((utility_grid.reshape((1, 1, 1, M, N)) * self._T2)
                        .sum(axis=-1).sum(axis=-1))[r, c, policy_grid.flatten()]
            .reshape(self.shape)
        )

        utility_grid[self._terminal_mask] = self._reward_grid[self._terminal_mask]

        return self.best_policy(utility_grid), utility_grid

    def _calculate_utility(self, loc, discount, utility_grid):
        if self._terminal_mask[loc]:
            return self._reward_grid[loc]
        row, col = loc
        # Need to remove all probabilities from T Matrix as I calculate utility
        dist = self._disturbances[row][col]
        for w in range(0,4):
            if w == dist:
                continue
            else:
                (self._T[row, col, :, w, :, :]).fill(0)
        self._T2[row, col, :, :, :] = self._T[row, col, :,dist,:,:]
        return np.max(
            discount * np.sum(
                np.sum(self._T[row, col, :, dist, :, :] * utility_grid,
                       axis=-1),
                axis=-1)
        ) + self._reward_grid[loc]

    def plot_policy(self, utility_grid, policy_grid=None):
        if policy_grid is None:
            policy_grid = self.best_policy(utility_grid)
        markers = "^>v<"
        marker_size = 200 // np.max(policy_grid.shape)
        marker_edge_width = marker_size // 10
        marker_fill_color = 'w'

        no_action_mask = self._terminal_mask | self._obstacle_mask

        utility_normalized = (utility_grid - utility_grid.min()) / \
                             (utility_grid.max() - utility_grid.min())

        utility_normalized = (255*utility_normalized).astype(np.uint8)

        utility_rgb = cv2.applyColorMap(utility_normalized, cv2.COLORMAP_JET)
        for i in range(3):
            channel = utility_rgb[:, :, i]
            channel[self._obstacle_mask] = 0

        plt.imshow(utility_rgb[:, :, ::-1], interpolation='none')

        for i, marker in enumerate(markers):
            y, x = np.where((policy_grid == i) & np.logical_not(no_action_mask))
            plt.plot(x, y, marker, ms=marker_size, mew=marker_edge_width,
                     color=marker_fill_color)

        y, x = np.where(self._terminal_mask)
        plt.plot(x, y, 'o', ms=marker_size, mew=marker_edge_width,
                 color=marker_fill_color)

        tick_step_options = np.array([1, 2, 5, 10, 20, 50, 100])
        tick_step = np.max(policy_grid.shape)/8
        best_option = np.argmin(np.abs(np.log(tick_step) - np.log(tick_step_options)))
        tick_step = tick_step_options[best_option]
        plt.xticks(np.arange(0, policy_grid.shape[1] - 0.5, tick_step))
        plt.yticks(np.arange(0, policy_grid.shape[0] - 0.5, tick_step))
        plt.xlim([-0.5, policy_grid.shape[0]-0.5])
        plt.xlim([-0.5, policy_grid.shape[1]-0.5])


    def sim_best_policy(self, start, goal, traps, best_policy, utility_grid, iterations=50):
        print('todo')
        cnt = 0
        successes = 0
        success_rewards = np.empty((1))
        cases = [(-1, 0), (0, 1), (1, 0), (0, -1), (0, 0)]
        while cnt < iterations:
            # print('coding;')
            current = start
            reward = 0
            while current not in traps and current != goal:
                if current is start:
                    probs = [0.5,0.5,0,0,0]
                else:
                    # print('working')
                    opt_action = int(best_policy[current[0], current[1]])
                    probs = np.empty([len(cases)])
                    for i in range(len(probs)):
                        if 0 <= cases[i][0] + current[0] < self.shape[0] and 0 <= cases[i][1] + current[1] < self.shape[1]:
                            probs[i] = self._T2[current[0], current[1], opt_action,
                                                cases[i][0] + current[0], cases[i][1] + current[1]]
                        else:
                            # probs[i] = self._T2[current[0], current[1], opt_action,
                            #                     current[0], current[1]]
                            probs[i] = 0.0
                choices = [0,1,2,3,4]
                if sum(probs) != 1.0:
                    print('PROBS ERROR')
                move = np.random.choice(choices, p=probs)
                current = (cases[move][0] + current[0], cases[move][1] + current[1])
                reward = reward + utility_grid[current[0],current[1]]
            else:
                if current == goal:
                    successes += 1
                    success_rewards = np.append(success_rewards, reward)
                    print('Iteration {}: Successfully made it to goal {} with reward {}'.format(cnt+1,goal, reward))
                else:
                    print('Iteration {}: Failed by landing in {} with reward'.format(cnt+1,current, reward))
            cnt+=1
        else:
            print('Successful tries: {}/{} with an average reward of {}'.format(successes, iterations, np.average(success_rewards)))
            print('REWARDS: \n -----------')
            print(success_rewards)


