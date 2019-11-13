import pandas as pd

probs_B = {
    'B State': ['T', 'F'],
    'Prob': [0.9, 0.1]
}

probs_M = {
    'M State': ['T', 'F'],
    'Prob': [0.1, 0.9]
}

probs_I = {
    'B State': ['T', 'T', 'F', 'F'],
    'M State': ['T', 'F', 'T', 'F'],
    'Prob': [0.9, 0.5, 0.1, 0.5]
}

probs_G = {
    'B State': ['T', 'T', 'T', 'T', 'F', 'F', 'F', 'F'],
    'I State': ['T', 'T', 'F', 'F', 'T', 'T', 'F', 'F'],
    'M State': ['T', 'F', 'T', 'F', 'T', 'F', 'T', 'F'],
    'Prob': [0.9, 0.8, 0.0, 0.0, 0.2, 0.1, 0.0, 0.0]
}

probs_J = {
    'G State': ['T', 'F'],
    'Prob': [0.9, 0.0]
}

# df_B = pd.dataFrame(probs_B)
# df_B = pd.dataFrame(probs_B)
# df_B = pd.dataFrame(probs_B)
# df_B = pd.dataFrame(probs_B)
# df_B = pd.dataFrame(probs_B)

print('hello')



