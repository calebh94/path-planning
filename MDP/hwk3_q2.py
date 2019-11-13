from util_mdp import GridWorldMDP
import numpy as np
import matplotlib.pyplot as plt


def plot_convergence(utility_grids, policy_grids):
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    utility_ssd = np.sum(np.square(np.diff(utility_grids)), axis=(0, 1))
    ax1.plot(utility_ssd, 'b.-')
    ax1.set_ylabel('Change in Utility', color='b')

    policy_changes = np.count_nonzero(np.diff(policy_grids), axis=(0, 1))
    ax2.plot(policy_changes, 'r.-')
    ax2.set_ylabel('Change in Best Policy', color='r')


if __name__ == '__main__':
    shape = (7, 8)
    goal = (1, 7)
    traps = [(0,0),(0,5),(2,3),(3,3),(3,4),(3,7),
             (4,1),(5,1),(5,5),(6,5)]
    # trap = (3,3)
    # obstacle = (5, 1)
    # obstacles = [(0,0),(0,5),(2,3),(3,3),(3,4),(3,7),
    #          (4,1),(5,1),(5,5),(6,5),(6,0)]
    # obstacles = [(0,0),(0,5),(2,3),(3,3),(3,4),(3,7),
    #          (4,1),(5,1),(5,5),(6,5)]
    # obstacles = [(6, 0)]
    start = (6, 0)
    default_reward = -1.0
    goal_reward = 10.0
    trap_reward = -10.0
    special_reward = 10.0  # CHANGE FOR PART B

    reward_grid = np.zeros(shape) + default_reward
    reward_grid[goal] = goal_reward

    # reward_grid2 = reward_grid
    specials = [(3,2),(0,4),(1,4)]
    for special in specials:
        reward_grid[special] = special_reward

    # reward_grid[trap] = trap_reward
    # reward_grid[obstacle] = 0

    terminal_mask = np.zeros_like(reward_grid, dtype=np.bool)

    for trap in traps:
        reward_grid[trap] = trap_reward
        terminal_mask[trap] = True

    terminal_mask[goal] = True
    # terminal_mask[trap] = True

    obstacle_mask = np.zeros_like(reward_grid, dtype=np.bool)

    # for obstacle in obstacles:
    #     obstacle_mask[obstacle] = True

    # obstacle_mask[1, 1] = True
    #
    # disturbance_directions = ['O','D','D','R','U','O',
    #                         'U','D','L','R','R','D','L',
    #                        'L','L','O','U','R','D','O',
    #                        'D','D','L','L','U','L','D','O',
    #                        'O','R','R','O','R','O','U','D',
    #                        'R','D','U','L','U','O','R','U',
    #                        'U','O','U','D','O','R','R','U',
    #                        'L','O','U','L']

    disturbance_directions = [[-1,2,2,1,0,-1,0,2],
                              [3,1,1,2,3,3,3,0],
                              [0,1,2,-1,2,2,3,3],
                              [0,3,2,-1,-1,1,1,-1],
                              [1,-1,0,2,1,2,0,3],
                              [0,-1,1,0,0,-1,0,2],
                              [0,1,1,0,3,-1,0,3]]

    # NEED TO PLOT DISTURNANCES TO CHECK
    gw = GridWorldMDP(reward_grid=reward_grid,
                      obstacle_mask=obstacle_mask,
                      terminal_mask=terminal_mask,
                      # action_probabilities=[
                      #     (-1, 0.1),
                      #     (0, 0.8),
                      #     (1, 0.1),
                      # ],
                      action_probabilities=[
                          # ('WithWind', [0.75,0.10,0.10,0.05]),
                          #  ('SideWindL', [0.20, 0.60, 0.05, 0.15])
                          # ('AgainstWind', [0.05,0.20,0.20,0.55]),
                          # ('SideWindR', [0.20,0.60,0.05,0.15]),
                          # (0, [0.75, 0.10, 0.10, 0.05]),
                          # (1, [0.20, 0.60, 0.05, 0.15]),
                          # (2, [0.05, 0.20, 0.20, 0.55]),
                          # (3, [0.20, 0.60, 0.05, 0.15])
                          (0, [0.75, 0.10, 0.05, 0.10]),   # Chances to go [Desired, +90,  +180, +270]
                          (1, [0.60, 0.20, 0.05, 0.15]),
                          # (1, [0.20, 0.60, 0.05, 0.15]),
                          (2, [0.05, 0.20, 0.55, 0.20]),
                          (3, [0.20, 0.60, 0.15, 0.05])
                      ],
                      disturbances=disturbance_directions,
                      no_action_probability=0.0)

    # gw2 = GridWorldMDP(reward_grid=reward_grid2,
    #                   obstacle_mask=obstacle_mask,
    #                   terminal_mask=terminal_mask,
    #                   # action_probabilities=[
    #                   #     (-1, 0.1),
    #                   #     (0, 0.8),
    #                   #     (1, 0.1),
    #                   # ],
    #                   action_probabilities=[
    #                       # ('WithWind', [0.75,0.10,0.10,0.05]),
    #                       # ('AgainstWind', [0.05,0.20,0.20,0.55]),
    #                       # ('SideWindR', [0.20,0.60,0.05,0.15]),
    #                       #  ('SideWindL', [0.20, 0.60, 0.05, 0.15])
    #                       (0, [0.75, 0.10, 0.10, 0.05]),
    #                       (1, [0.20, 0.60, 0.05, 0.15]),
    #                       (2, [0.05, 0.20, 0.20, 0.55]),
    #                       (3, [0.20, 0.60, 0.05, 0.15])
    #                   ],
    #                   disturbances=disturbance_directions,
    #                   no_action_probability=0.0)
    
    # CHOOSE CASE AND SOLVER HERE
    mdp_solvers = {'Value Iteration': gw.run_value_iterations}
                   # 'Policy Iteration': gw.run_policy_iterations}

    for solver_name, solver_fn in mdp_solvers.items():
        print('Final result of {}:'.format(solver_name))
        policy_grids, utility_grids = solver_fn(iterations=50, discount=0.95, epsilon=0.1)
        print(policy_grids[:, :, -1])
        print(utility_grids[:, :, -1].round())
        plt.figure()
        gw.plot_policy(utility_grids[:, :, -1])
        plot_convergence(utility_grids, policy_grids)
        plt.show()