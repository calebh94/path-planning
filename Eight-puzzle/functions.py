import numpy as np


def setFromState(puzzle):
    # Vectorize
    grid = np.array(puzzle)
    vector = grid.flatten('C')
    # loop through and cnt
    total_cnt = 0
    less_cnt = 0
    for i in range(0,9):
        curr_val = vector[i]
        if curr_val == '_':
            continue
        else:
            curr_val = int(curr_val)
        for j in range(i,9):
            next_val = vector[j]
            if next_val == '_':
                continue
            else:
                next_val = int(next_val)

            less_cnt = 0
            if 0 < next_val < curr_val:
                less_cnt = less_cnt + 1
                total_cnt = less_cnt + total_cnt
    print(total_cnt)
    (quo, mod) = divmod(total_cnt, 2)
    if mod == 1:
        group = 'A'
    elif mod == 0:
        group = 'B'
    return group


# test_puzzle = [[5,4,0],[6,1,8],[7,3,2]]
# test_puzzle = [[2,8,3],[1,6,4],[7,0,5]] # online example (A)
# test_puzzle = [[4,5,1],[6,7,0],[8,3,2]] # example A
# test_puzzle = [[1,2,3],[4,5,6],[7,8,0]] # GOAL STATE B
# test_puzzle = [[1,2,3],[8,'_',4],[7,6,5]] # GOAL STATE a
# test_puzzle = [['4','5','6'],['1','7','_'],['8','3','2']] # B
test_puzzle = [[5,4,'_'],[6,1,8],[7,3,2]]  # Homework Puzzle

result = setFromState(test_puzzle)
print(result)