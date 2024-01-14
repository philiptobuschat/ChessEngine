import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from Resources.Model import Model_v25
from Resources.Game import *
from Resources.TS_ModelGuided import *
from Resources.TS_ModelGuided_MCRollout import *
from Resources.TS_ModelGuided_SensAnalysis import *

from test_games import test_games

model = Model_v25()
model.eval()
model.load_state_dict(torch.load('../Monte Carlo/Model Saves MC v25/model_2000_batches'))

n_games = 1
tmax = 2

factors = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1e-3]
# [wins, value sum, value ind, grad sum, grad ind, exploration]
factors = np.array(factors)

factor_i_hist = []

factor_i = 4

while True:

    for j in [0, 2, 3, 4, 5, 6]:
        try:
            factor_j = torch.load('latest factor_{}'.format(j))
            # print('loaded factor_{} = {}'.format(j, factor_j))
            factors[j] = factor_j
        except:
            pass

    factors_test = factors.copy()
    factors_test[factor_i] += 0.1

    wins_1, draws, wins_2 = test_games(n_games, model, tmax, 
                                       factors, factors_test)

    if wins_1 > wins_2:
        factors[factor_i] -= 0.1
    elif wins_2 > wins_1:
        factors[factor_i] += 0.1
    
    factor_i_hist.append(factors[factor_i])
    torch.save(factor_i_hist, 'hist factor_{}'.format(factor_i))

    torch.save(factors[factor_i], 'latest factor_{}'.format(factor_i))

    print('factor {} = {}'.format(factor_i, factors[factor_i]))

