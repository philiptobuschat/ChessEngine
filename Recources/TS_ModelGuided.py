import time
import numpy as np
import random
import torch
from Game import *
from RollingMedian import RollingMedian

def ModelGuided_TS(game, model, root=None, tmax=60, batches=10, prints=False, factor_wins=1, 
                   factor_mat=0, factor_value_sum=0.5, factor_value_indi=0.5, factor_explore=1e-3):
    '''
    split whole duration in batches to make value calculation more efficient
    '''

    # Create the root node if not passed
    if root is None:
        root = ModelGuided_Node(None, None, game, [factor_wins, factor_mat, 
                                factor_value_sum, factor_value_indi, factor_explore]) 

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

            if node.game.is_over():
                winner = node.game.get_winner()
            else:
                winner = None

            # backpropagation of winner, matdiff
            for_side = node.player

            node.backpropagate(winner, node.matdiff, for_side, prints)
            
            nodes.append(node)
            # board_batch.append(board_to_tensor(node.game.pieces))

        if len(board_batch) == 0:
            continue

        # backpropagation of values
        board_batch = torch.stack(board_batch)
        values = model(board_batch).detach().numpy()

        for node, value in zip(nodes, values):

            for_side = node.player
            node.backpropagate_value(value, for_side)
            node.value = value

        board_batch = []
        nodes = []

    # Choose the best move based on the visit counts of child nodes
    best_move = root.get_best_move()

    return best_move, root


class ModelGuided_Node:
    '''
    every instance corresponds to a move in the expansion tree.
    game is the state after playing the move
    batched: value function evaluation in batches for acceleration
    batched v3: collect not only wins but wins, matdiff, value for positions / backpropagation
    '''
    def __init__(self, move, parent, game, factors):
        self.move = move;   self.parent = parent
        self.children = []; self.wins = 0
        self.value = None
        self.visits = 0 ;   self.game = game

        self.value_rollmed = RollingMedian()
        self.matdiff_rollmed = RollingMedian()

        self.factors = factors

        # game.turn is after our move if played and the board is flipped, player should be before our move 
        if game.turn == 'white': self.player = 'black'
        if game.turn == 'black': self.player = 'white'

        # next child nodes
        self.untried_moves = game.PossibleMoves()

        self.matdiff = - self.game.MaterialDiff() # minus since board is flipped after move already

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_child(self):

        self.values_win = [child.wins / max(child.visits, 1)
                        for child in self.children]
        
        self.values_mat = [np.tanh(child.matdiff_rollmed.get_median() + self.matdiff)    
                        for child in self.children]
        
        self.values_value_sum = [np.tanh(child.value_rollmed.get_median() + self.value)
                                 if (child.value_rollmed.get_median() is not None and self.value is not None) else 0
                        for child in self.children]
        
        self.values_value_indi = [child.value + self.value 
                                  if (child.value is not None and self.value is not None) else 0
                        for child in self.children]
        
        # self.values_explore = [(self.visits) / max(child.visits, 1)
        #                 for child in self.children]

        self.values_explore = [-np.log10(max(child.visits, 1))
                        for child in self.children]
        

        self.children_values = [
              self.factors[0]    * self.values_win[c] 
            + self.factors[1]     * self.values_mat[c]
            + self.factors[2]   * self.values_value_sum[c]
            + self.factors[3]   * self.values_value_indi[c]
            + self.factors[4] * self.values_explore[c]
            for c, child in enumerate(self.children)
        ]

        # select child with highest ucb value or random one of the best ones
        max_value = max(self.children_values)
        max_indices = [i for i, value in enumerate(self.children_values) if value == max_value]
        selected_index = random.choice(max_indices)

        return self.children[selected_index]

    def expand(self, board_batch):
        # Choose an untried move and create a new child node
        move = random.choice(self.untried_moves)
        self.untried_moves.remove(move)
        child_game = self.game.copy()
        child_game.PlayMove(move)
        board_batch.append(board_to_tensor(child_game.pieces))
        child_game.FlipBoard()
        child_node = ModelGuided_Node(move, self, child_game, self.factors)
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