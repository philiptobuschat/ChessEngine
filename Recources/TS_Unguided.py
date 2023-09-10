import time
import numpy as np
import random

def Unguided_TS(game, root=None, tmax=60, prints=False, factor_wins=2, factor_mat=0.5, 
                factor_check=0.1, factor_capture=0.1, factor_attack=0.01, factor_explore=1e-3):

    # Create the root node if not passed
    if root is None:
        root = Unguided_Node(None, None, game, factor_wins, factor_mat, factor_check, 
                             factor_capture, factor_attack, factor_explore)
        
    t0 = time.time()

    while time.time() - t0 < tmax:

        node = root

        # Selection
        while not node.is_leaf() and node.is_fully_expanded():
            node = node.select_child()

        # Expansion
        if not node.is_fully_expanded():
            node = node.expand()

        if node.game.is_over():
            winner = node.game.get_winner()
        else:
            winner = None

        # backpropagation of winner, matdiff
        for_side = node.player

        node.backpropagate(winner, node.matdiff, for_side, prints)

    # Choose the best move based on the visit counts of child nodes
    best_move = root.get_best_move()

    return best_move, root

class Unguided_Node:
    '''
    Node Class for TS with only simple heuristics as guide
    '''
    def __init__(self, move, parent, game, factor_wins, factor_mat, factor_check, 
                             factor_capture, factor_attack, factor_explore):
        
        self.move = move;   self.parent = parent
        self.children = []; self.wins = 0; self.matdiff_sum = 0
        self.visits = 0 ;   self.game = game

        self.factor_wins = factor_wins; self.factor_mat = factor_mat
        self.factor_check = factor_check; self.factor_capture = factor_capture
        self.factor_attack = factor_attack; self.factor_explore = factor_explore

        # game.turn is after our move if played and the board is flipped, player should be before our move 
        if game.turn == 'white': self.player = 'black'
        if game.turn == 'black': self.player = 'white'

        # next child nodes
        self.untried_moves = game.PossibleMoves()

        # collect info about this node's move: is it check, capture, threats, material difference
        dict = {0: 0, 11: 1, 12: 3, 13: 3, 14: 5, 15: 9, 16: 0}

        self.check          = 1 if self.game.InCheck() else 0
        self.capture        = np.tanh( dict[game.latest_capture] / 5 )
        self.threats        = [] 

        self.matdiff = - self.game.MaterialDiff() # minus since board is flipped after move already

    def is_leaf(self):
        return len(self.children) == 0

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def select_child(self):
        '''
        main logic that navigates the search tree.

        each child gets a value in each category between [0, 1] (for material difference between [-1, 1]).
        calculate weighted sum of all categories for each child based on weighting factors given.

        -> follow child with highest total value

        ties split randomly
        '''

        self.values_win = [child.wins / max(child.visits, 1)
                        for child in self.children]
        
        self.values_mat = [np.tanh(child.matdiff_sum / max(child.visits, 1) + self.matdiff_sum / max(self.visits, 1))    
                        for child in self.children]
                
        self.values_check = [child.check
                        for child in self.children]
        
        self.values_capture = [child.capture
                        for child in self.children]
        
        self.values_attack = [child.attack
                        for child in self.children]

        self.values_explore = [(self.visits) / max(child.visits, 1)
                        for child in self.children]

        self.children_values = [
              self.factor_wins    * self.values_win[c] 
            + self.factor_mat     * self.values_mat[c]
            + self.factor_check   * self.values_check[c]
            + self.factor_capture * self.values_capture[c]
            + self.factor_attack  * self.values_attack[c]
            + self.factor_explore * self.values_explore[c]
            for c, child in enumerate(self.children)
        ]

        # select child with highest value or random one of the best ones
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

        # if used, calculate threats here
        if self.factor_attack != 0:
            threats = child_game.ThreatTest(move[1])
        else: 
            threats = []
        
        child_game.FlipBoard()
        child_node = Unguided_Node(move, self, child_game, self.factor_wins, self.factor_mat, 
                                    self.factor_check, self.factor_capture, 
                                    self.factor_attack, self.factor_explore)
        
        child_node.attack = np.tanh(sum(threats) / 5)
        self.children.append(child_node)
        return child_node

    def backpropagate(self, winner, matdiff, for_side, prints):
        ''' 
        main function backpropagation
        after leaf reached: backpropagate winner, value and matdiff of leaf through tree
        self.player is from whos perspective a node is
        '''
        if prints:
            print('backprob curr move: ', self.move)
        self.visits += 1

        # winner (if decicive, i.e. game terminates and no draw)
        if winner is not None:
            if winner == self.player:
                self.wins += 1
            elif winner == self.game.opponent[self.player]:
                self.wins -= 1

        # matdiff (applies independent of outcome)
        if for_side == self.player:
            self.matdiff_sum += matdiff
        else:
            self.matdiff_sum -= matdiff

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