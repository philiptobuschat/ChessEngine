# Tree Search guided by value function and sensitivity analysis of the value function

import time
import numpy as np
import random
import torch
import torch.autograd.functional as Func
import sys, os

project_root = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
sys.path.append(project_root)

from Resources.Game import *
from Resources.RollingMedian import RollingMedian

def sensitivity_extraction_1(grad):
    '''
    extract scalar value from model gradient.
    R^(12*8*8) -> R
    method 1: 2-norm of gradient
    '''
    return torch.norm(grad, p=2).item()

def sensitivity_extraction_2(grad):
    '''
    extract scalar value from model gradient.
    R^(12*8*8) -> R
    method 2: unweighted sum of gradient
    '''
    return torch.sum(grad).item()

def sensitivity_extraction_3(grad):
    '''
    extract scalar value from model gradient.
    R^(12*8*8) -> R
    method 3: activation function -> sum of gradient
    '''
    return torch.sum(torch.tanh(grad)).item()

def sensitivity_extraction_4(grad):
    '''
    extract scalar value from model gradient.
    R^(12*8*8) -> R
    method 4: pooling over each piece type -> act. function -> then sum
    '''
    return torch.sum(torch.tanh(torch.sum(grad, dim=(1, 2)))).item()

def sensitivity_extraction_5(grad):
    '''
    extract scalar value from model gradient.
    R^(12*8*8) -> R
    method 5: learned extraction function (e.g. NN)

    possible future implementation
    '''
    return None

def SensAnalysis_TS(game, model, root=None, tmax=60, batches=10, prints=False, factors=None, factor_wins=1, 
                   factor_mat=0, factor_value_sum=0.5, factor_value_indi=0.5, 
                   factor_grad_sum=0.1, factor_grad_indi=0.1,
                   factor_explore=1e-3,
                   sensitivity_extraction = sensitivity_extraction_1, 
                   use_gradients=True):
    '''
    Tree Search Guided by model and its sensitivity via model gradient
    '''

    # Create the root node if not passed
    if root is None:
        if factors is None:
            root = SensAnalysis_Node(None, None, game, [factor_wins, factor_mat, 
                                factor_value_sum, factor_value_indi, 
                                factor_grad_sum, factor_grad_indi,
                                factor_explore]) 
        if factors is not None:
            root = SensAnalysis_Node(None, None, game, factors, use_gradients=use_gradients) 

    root.game.FlipBoard()
    board_batch = [board_to_tensor(root.game.pieces)]
    nodes = [root]
    root.game.FlipBoard()

    for _ in range(batches):

        t0 = time.time()
        while time.time() - t0 < float(tmax) / float(batches):

            node = root

            # Selection
            while not node.is_leaf() and node.is_fully_expanded():
                node = node.select_child()

            # Expansion
            if not node.is_fully_expanded():
                node, board_batch = node.expand(board_batch)

            if node.game.is_over_eff():
                winner = node.game.get_winner()
            else:
                winner = None

            # backpropagation of winner, matdiff
            for_side = node.player

            node.backpropagate(winner, node.matdiff, for_side, prints)
            
            nodes.append(node)
            # board_batch.append(board_to_tensor(node.game.pieces))

        batch_size = len(board_batch)

        if batch_size == 0:
            continue

        # get model values
        tens_board_batch = torch.stack(board_batch)
        tens_board_batch.requires_grad = True
        values = model(tens_board_batch).detach().numpy()

        if use_gradients:

            # get each gradient w.r.t. input by vectorising input and using jacobian functionality
            vectorized_input = tens_board_batch.view(batch_size, -1)
            jacobian_matrix = Func.jacobian(lambda x: model(x.view(-1, 12, 8, 8)).sum(), vectorized_input)
            gradients = jacobian_matrix.view(batch_size, 12, 8, 8)

            for node, value, grad in zip(nodes, values, gradients):

                for_side = node.player
                grad_ext = sensitivity_extraction(grad)

                # node.backpropagate_value(value, for_side)
                node.backpropagate_value_grad(value.item(), grad_ext, for_side)
                node.value = value
                node.grad = grad_ext
        
        else:

            for node, value in zip(nodes, values):

                for_side = node.player
                node.backpropagate_value(value.item(), for_side)
                node.value = value.item()

        root.rollmed_update()

        board_batch = []
        nodes = []

    # Choose the best move based on the visit counts of child nodes
    best_move = root.get_best_move()

    return best_move, root


class SensAnalysis_Node:
    '''
    every instance corresponds to a move in the expansion tree.
    game is the state after playing the move
    '''
    def __init__(self, move, parent, game, factors, use_gradients=False):
        self.move = move;   self.parent = parent
        self.children = []; self.wins = 0
        self.value = None; self.grad = None
        self.visits = 0 ;   self.game = game
        self.factors = factors

        self.use_gradients = use_gradients

        self.value_rollmed = RollingMedian()
        self.value_rollmed_curr = None

        if self.use_gradients:
            self.grad_rollmed = RollingMedian()
            self.grad_rollmed_curr = None
        
        if self.factors[1] != 0:
            self.matdiff_rollmed = RollingMedian()
            self.matdiff_rollmed_curr = None
            self.matdiff = - self.game.MaterialDiff() # minus since board is flipped after move already
        else:
            self.matdiff = 0

        self.values_value_sum = None
        self.values_value_indi = None
        self.values_comb = None

        # game.turn is after our move if played and the board is flipped, player should be before our move 
        if game.turn == 'white': self.player = 'black'
        if game.turn == 'black': self.player = 'white'

        # next child nodes
        self.untried_moves = game.PossibleMoves()

    def rollmed_update(self):
        '''
        update rolling median values of this node and all children
        '''
        
        self.value_rollmed_curr = self.value_rollmed.get_median()

        if self.use_gradients:
            self.grad_rollmed_curr = self.grad_rollmed.get_median()
        
        if self.factors[1] != 0:
            self.matdiff_rollmed_curr = self.matdiff_rollmed.get_median()
        
        for child in self.children:
            child.rollmed_update()
        
        # prepare values for child selection to avoid redundant calculation
        if self.is_fully_expanded():

            children_values = [child.value for child in self.children if child.value is not None]
            if len(children_values) != 0:
                mean_next_value = np.mean(children_values)
            else:
                mean_next_value = None

            self.values_value_sum = [np.tanh(child.value_rollmed_curr - mean_next_value).item()
                                if (child.value_rollmed_curr is not None and mean_next_value is not None) else 0
                        for child in self.children]
            self.values_value_indi = [child.value - mean_next_value
                                  if (child.value is not None and mean_next_value is not None) else 0
                        for child in self.children]
        else:
            self.values_value_sum = None
            self.values_value_indi = None

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_child(self):

        # for child in self.children:
        #     print(child.value)
        #     print(child.value_rollmed.get_median())

        children_values = [child.value for child in self.children if child.value is not None]
        if len(children_values) != 0:
            mean_next_value = np.mean(children_values)
        else:
            mean_next_value = None

        self.values_win = [child.wins / max(child.visits, 1)
                        for child in self.children]
        
        if self.factors[1] != 0:
            self.values_mat = [np.tanh(child.matdiff_rollmed_curr + self.matdiff)   
                               if child.matdiff_rollmed_curr is not None else 0
                            for child in self.children]
        else:
            self.values_mat = [0 for _ in self.children]
        
        if self.values_value_sum is None:
            self.values_value_sum = [np.tanh(child.value_rollmed_curr - mean_next_value).item()
                                    if (child.value_rollmed_curr is not None and mean_next_value is not None) else 0
                            for child in self.children]
        
        if self.values_value_indi is None:
            self.values_value_indi = [child.value - mean_next_value
                                    if (child.value is not None and mean_next_value is not None) else 0
                            for child in self.children]
        
        self.values_comb = [self.values_value_indi[c] * max((1000 - child.visits) / (1000), 0) + self.values_value_sum[c] * min(child.visits / 1000, 1)
                                    if (self.values_value_indi[c] is not None and self.values_value_sum[c] is not None) else 0
                        for c, child in enumerate(self.children)]

        if self.use_gradients:
            self.values_grad_sum = [np.tanh(child.grad_rollmed_curr)
                                    if (child.grad_rollmed_curr is not None) else 0
                            for child in self.children]
            
            self.values_grad_indi = [np.tanh(child.grad)
                                    if (child.grad is not None) else 0
                            for child in self.children]
        else:
            self.values_grad_sum    = [0 for _ in self.children]
            self.values_grad_indi   = [0 for _ in self.children]
        
        # self.values_explore = [(self.visits) / max(child.visits, 1)
        #                 for child in self.children]

        self.values_explore = [-np.log10(max(child.visits, 1))
                        for child in self.children]
        

        self.children_values = [
              self.factors[0]   * self.values_win[c] 
            + self.factors[1]   * self.values_mat[c]
            + self.factors[2]   * self.values_value_sum[c]
            + self.factors[3]   * self.values_value_indi[c]
            + self.factors[4]   * self.values_grad_sum[c]
            + self.factors[5]   * self.values_grad_indi[c]
            + self.factors[6]   * self.values_explore[c]
            + self.factors[7]   * self.values_comb[c]
            for c, child in enumerate(self.children)
        ]

        # # select child with highest ucb value or random one of the best ones
        # max_value = max(self.children_values)
        # max_indices = [i for i, value in enumerate(self.children_values) if value == max_value]
        # selected_index = random.choice(max_indices)

        max_ind = np.argmax(self.children_values)

        return self.children[max_ind]

    def expand(self, board_batch):
        # Choose an untried move and create a new child node
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        child_game = self.game.copy()
        child_game.PlayMove(move)
        board_batch.append(board_to_tensor(child_game.pieces))
        child_game.FlipBoard()
        child_node = SensAnalysis_Node(move, self, child_game, self.factors, use_gradients=self.use_gradients)
        self.children.append(child_node)
        return child_node, board_batch

    def backpropagate_value(self, value, for_side):
        ''' 
        backpropagate only value of leaf through tree
        self.player is from whos perspective a node is
        '''

        # value (applies independent of outcome)
        if for_side == self.player:
            self.value_rollmed.add_number(value)
        else:
            self.value_rollmed.add_number(-value)

        if self.parent is not None:
            self.parent.backpropagate_value(value, for_side)

    def backpropagate_value_grad(self, value, grad, for_side):
        ''' 
        backpropagate only value of leaf through tree
        self.player is from whos perspective a node is
        '''

        # value (applies independent of outcome)
        if for_side == self.player:
            self.value_rollmed.add_number(value)
            self.grad_rollmed.add_number(grad)
        else:
            self.value_rollmed.add_number(-value)
            self.grad_rollmed.add_number(-grad)

        if self.parent is not None:
            self.parent.backpropagate_value_grad(value, grad, for_side)

    def backpropagate(self, winner, matdiff, for_side, prints):
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

        if self.factors[1] != 0:
        # matdiff (applies independent of outcome)
            if for_side == self.player:
                self.matdiff_rollmed.add_number(matdiff)
            else:
                self.matdiff_rollmed.add_number(-matdiff)

        if self.parent is not None:
            self.parent.backpropagate(winner, matdiff, for_side, prints)

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