import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
import torch.optim as optim
import matplotlib.pyplot as plt
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(project_root)

from Resources.Model import Model_v25
from Resources.Game import *
from Resources.TS_ModelGuided import *
from Resources.TS_ModelGuided_MCRollout import *
from Resources.TS_ModelGuided_SensAnalysis import *

def run_test_games(process_id, arg_list):
    n_games, model, tmax, factors_1, factors_2 = arg_list
    result = test_games(n_games, model, tmax, factors_1, factors_2, process_id)
    # print(process_id, result)
    torch.save(result, 'temp {}'.format(process_id))
    # output_queue.put(result)
    return

def test_games(n_games, model, tmax, factors_1, factors_2, sensitivity_extraction=sensitivity_extraction_1):

    draws = 0; wins_1 = 0; wins_2 = 0

    for _ in range(n_games): # loop through test games

        game = Game()
        i = 0

        color_choice = np.random.choice([True, False])

        if color_choice:
            color_1 = 'white'; color_2 = 'black'
        else:
            color_1 = 'black'; color_2 = 'white'

        while not game.is_over(): # loop through moves in current test game

            if game.turn == color_1:
                factors = factors_1
            else:
                factors = factors_2

            chosen_move, root = SensAnalysis_TS(game, model, factors=factors, 
                                                root=None, tmax=tmax, prints=False, 
                                                sensitivity_extraction=sensitivity_extraction)

            game.PlayMove(chosen_move)
            game.FlipBoard()

            i += 1

            # if i % 20 == 0:
            #     print(process_id, i)

        winner = game.get_winner()

        if winner == color_1:
            wins_1 += 1
        elif winner == color_2:
            wins_2 += 1
        elif winner == 'draw':
            draws += 1

    return wins_1, draws, wins_2