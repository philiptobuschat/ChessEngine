import time
import numpy as np
import random
import torch
from Game import *

def MC_TS(game, model, root=None, tmax=60, prints=False, factor_explore=1e-2):
    '''
    Monte Carlo Tree Search
    Use only wins (+exploration) for navigating search tree
    Roll out every leaf node and backprop win
    '''

    # Create the root node if not passed
    # pass root node if previous calculations can be reused
    if root is None:
        root = MC_Node(None, None, game, factor_explore) 

    t0 = time.time()
    while time.time() - t0 < tmax:

        node = root

        # Selection
        while not node.is_leaf() and node.is_fully_expanded():
            node = node.select_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        # Rollout (or get winner)
        if node.game.is_over():
            winner = node.game.get_winner()
        else:
            winner = node.rollout(model)
    
        # Backpropagation
        node.backpropagate(winner, prints)

    # Choose the best move based on the visit counts of child nodes
    best_move = root.get_best_move()

    return best_move, root


class MC_Node:
    '''
    every instance corresponds to a move in the expansion tree.
    game is the state after playing the move
    batched: value function evaluation in batches for acceleration
    batched v3: collect not only wins but wins, matdiff, value for positions / backpropagation
    '''
    def __init__(self, move, parent, game, factor_explore):
        self.move = move;   self.parent = parent
        self.children = []; self.wins = 0
        self.visits = 0 ;   self.game = game

        self.factor_explore = factor_explore

        # game.turn is after our move if played and the board is flipped, player should be before our move 
        if game.turn == 'white': self.player = 'black'
        if game.turn == 'black': self.player = 'white'

        # next child nodes
        self.untried_moves = game.PossibleMoves()

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_child(self):

        self.values_win = [child.wins / max(child.visits, 1)
                        for child in self.children]

        self.values_explore = [-np.log10(max(child.visits, 1))
                        for child in self.children]

        self.children_values = [self.values_win[c]  + self.factor_explore * self.values_explore[c]
            for c, child in enumerate(self.children)
        ]

        max_value = max(self.children_values)
        max_indices = [i for i, value in enumerate(self.children_values) if value == max_value]
        selected_index = random.choice(max_indices)

        return self.children[selected_index]

    def expand(self):
        # Choose an untried move and create a new child node
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        child_game = self.game.copy()
        child_game.PlayMove(move)
        child_game.FlipBoard()
        child_node = MC_Node(move, self, child_game, self.factor_explore)
        self.children.append(child_node)
        return child_node
    
    def rollout(self, model):

        rollout_game = self.game.copy()

        i = 0

        while not rollout_game.is_over():

            i += 1

            if i > 150:
                return 'draw'
        
            moves = rollout_game.PossibleMoves()

            game_ini = rollout_game.copy()
            board_batch = []
            mate = False

            for move in moves:
                rollout_game.PlayMove(move)
                board_batch.append(board_to_tensor(rollout_game.pieces))
                rollout_game.FlipBoard()

                # if a move finishes the game, chose that move
                if rollout_game.is_over():
                    mate = True
                    chosen_move = move
                    rollout_game = game_ini.copy()
                    break
                rollout_game = game_ini.copy()

            # chose move that achieves highest value
            if not mate:
                board_tensor = torch.stack(board_batch)
                values = model(board_tensor).detach().numpy()

                max_value = max(values)
                max_indices = [i for i, value in enumerate(values) if value == max_value]
                selected_index = random.choice(max_indices)

                chosen_move = moves[selected_index]
                
            rollout_game.PlayMove(chosen_move)
            rollout_game.FlipBoard()

        return rollout_game.get_winner()

    def backpropagate(self, winner, prints):
        ''' 
        main function backpropagation
        after leaf reached: backpropagate winner and matdiff of leaf through tree
        self.player is from whos perspective a node is
        '''
        if prints:
            print('backprob curr move: ', self.move)
        self.visits += 1

        # winner (if decisive, i.e. game terminates and no draw)
        if winner is not None:
            if winner == self.player:
                self.wins += 1
            elif winner == self.game.opponent[self.player]:
                self.wins -= 1

        if self.parent is not None:
            self.parent.backpropagate(winner, prints)

    def get_best_move(self):
        best_child = None
        highest_visit_count = 0
        
        # Iterate through the children and find the best one
        for child in self.children:
            if child.visits > highest_visit_count:
                best_child = child
                highest_visit_count = child.visits
        
        if best_child is None:
            choice = np.random.choice([i for i in range(len(self.children))])
            return [child.move for child in self.children][choice]
        
        return best_child.move