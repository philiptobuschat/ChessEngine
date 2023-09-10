'''
Contains definitions for chess game, the main environment
'''

import time
import numpy as np
import torch

class Game:

    def __init__(self):
        self.Reset()
        self.opponent = {'white': 'black', 'black': 'white'}
        return

    def Reset(self):
        # encoding: 
        # first digit: 0: white, 1:black
        # second digit:
        # 0: empty
        # 1: pawn
        # 2: knight
        # 3: bishop
        # 4: rook
        # 5: queen
        # 6: king
        self.pieces = np.array([
            [14, 12, 13, 15, 16, 13, 12, 14],
            [11, 11, 11, 11, 11, 11, 11, 11],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 0,  0,  0,  0,  0,  0,  0,  0],
            [ 1,  1,  1,  1,  1,  1,  1,  1],
            [ 4,  2,  3,  5,  6,  3,  2,  4]
        ], dtype='uint8')


        self.turn = 'white'

        self.can_castle_white_long = True
        self.can_castle_white_short = True
        self.can_castle_black_long = True
        self.can_castle_black_short = True

        self.pawn_to_be_en_passant = None

        # after 50 moves without a capture or pawn move, the game is declared a draw
        self.counter_draw_by_moves = 0

        self.latest_capture = 0

        return

    def copy(self):
        """
        Create a deep copy of the Game object.
        """
        new_game = Game()
        new_game.pieces = self.pieces.copy()
        new_game.turn = self.turn
        new_game.can_castle_white_long = self.can_castle_white_long
        new_game.can_castle_white_short = self.can_castle_white_short
        new_game.can_castle_black_long = self.can_castle_black_long
        new_game.can_castle_black_short = self.can_castle_black_short
        new_game.pawn_to_be_en_passant = self.pawn_to_be_en_passant
        new_game.counter_draw_by_moves = self.counter_draw_by_moves
        return new_game
    
    def is_over(self):
        '''
        Check if the game is over. Game is over when the player, who's turn it is, cannot make a move
        Returns True if the game is over, False otherwise.
        '''

        if self.counter_draw_by_moves >= 50:
            return True

        # Draw by impossibility of checkmate
        piece_counts = np.bincount(self.pieces.flatten().astype(int))
        tot = sum(self.pieces.flatten())

        # Insufficient material scenarios:
        # 1. King vs. King
        # 2. King and Bishop vs. King
        # 3. King and Knight vs. King

        # Scenario 1: King vs. King
        if tot == 22:
            return True

        # Scenario 2: King and Bishop vs. King
        if (tot == 22+3 and piece_counts[3] == 1) or (tot == 22+13 and piece_counts[13] == 1):
            return True

        # Scenario 3: King and Knight vs. King
        if (tot == 22+2 and piece_counts[2] == 1) or (tot == 22+12 and piece_counts[12] == 1):
            return True

        moves = self.PossibleMoves()

        # No moves available -> stalemate or checkmate
        if len(moves) == 0: 
            return True

        return False

    def get_winner(self):
        '''
        Check which player won if the game is finished
        '''

        if self.InCheck():
            return self.opponent[self.turn]

        return 'draw'

    def OnBoard(self, pos):
        if pos[0] < 0 or pos[0] > 7 or pos[1] < 0 or pos[1] > 7:
            return False
        else:
            return True

    def FlipPositions(self, positions):
        '''
        Transform positions from one perspective to the other
        '''
        new_positions = []
        for pos in positions:
            new_positions.append((7-pos[0], 7-pos[1]))
        return new_positions

    def FlipBoard(self):
        '''
        transform board and pieces between perspectives
        '''

        new_board = np.zeros((8, 8))
        for x in range(8):
            for y in range(8):
                piece = self.pieces[x, y]
                if piece > 0 and piece < 10:
                    new_board[7-x, 7-y] = piece + 10
                elif piece > 10:
                    new_board[7-x, 7-y] = piece - 10
        self.pieces = new_board

        if self.turn == 'white':
            self.turn = 'black'
        else:
            self.turn = 'white'

        if self.pawn_to_be_en_passant is not None:
            self.pawn_to_be_en_passant = (7 - self.pawn_to_be_en_passant[0], 7 - self.pawn_to_be_en_passant[1])

        return

    def InCheck_legacy_1(self):
        '''
        Return whether we are in check.
        '''
        king = np.where(self.pieces == 6)
        king = (king[0].item(0), king[1].item(0))
        king = self.FlipPositions([king])[0]

        # board_backup = self.pieces.copy()
        self.FlipBoard()
        possible_moves = self.PossibleMoves(mode='check detection')
        self.FlipBoard()
        # self.pieces = board_backup.copy()

        incheck = False
        #print('king: ', king)
        for move in possible_moves:
            #print('move: ', move)
            if move[1] == king:
                incheck = True

        return incheck

    def InCheck_legacy_2(self, pos = None):
        '''
        Return whether we are in check.

        New version to be faster. 
        Dont use Flip + Possbile Moves + Flip
        Instead directly check all pieces from opponent

        option using pos = a given position [x, y]
        to test an different position than king.
        Used to test where king can go

        '''

        if pos is not None:
            king = pos
        
        else:

            king = np.where(self.pieces == 6)
            king = np.array(king).flatten()

        # its important to include the case of touching kings in this function
        # precisely to discard such cases
        opp_king = np.where(self.pieces == 16)
        opp_king = np.array(opp_king).flatten()

        if abs(king[0] - opp_king[0]) < 2 and abs(king[1] - opp_king[1]) < 2:
            return True

        # pawns
        if king[0] > 0 and king[1] > 0:
            if self.pieces[king[0]-1, king[1]-1] == 11:
                return True
        if king[0] > 0 and king[1] < 7:
            if self.pieces[king[0]-1, king[1]+1] == 11:
                return True

        # knights 
        if np.any(self.pieces == 12): # if any knights are present at all
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                x_test = king[0] + x_off
                y_test = king[1] + y_off
                if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                    if self.pieces[x_test, y_test] == 12:
                        return True
                
        rem_pieces = [13, 14, 15] # remaining pieces: bishops, rooks, quuens
        directions_rem_pieces = [
            [(1, 1), (-1, 1), (1, -1), (-1, -1)], # directions for bishop to move
            [(1, 0), (-1, 0), (0, 1), (0, -1)], # directions for rook to move
            [(1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1)], # directions for queen to move
        ]

        # for each piece type start at king position and follow its direction until 
        # we find a opponent's piece of this type -> in check
        # or we find another piece or the edge of the board -> not in check
        for type in range(3):
            if np.any(self.pieces == rem_pieces[type]): # if any of the considered type are present
                test_piece = rem_pieces[type]
                for dir in directions_rem_pieces[type]:
                    x_test = king[0]
                    y_test = king[1]
                    for _ in range(1, 8):
                        x_test += dir[0]
                        y_test += dir[1]
                        # if we hit such a piece -> in check
                        # otherwise go to next direction
                        if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                            if self.pieces[x_test, y_test] == test_piece:
                                return True
                            elif self.pieces[x_test, y_test] != 0:
                                break
                        else:
                            break
    
        return False
    
    def InCheck_pieces(self, pawns, knights, bishops, rooks, queens, king, opp_king, pos = None):
        '''
        version of in check where positions of opponent's pieces are given
        Used in PossibleMoves for efficient playability test of potential moves

        statistics, which pieces produce check how ofter, to order the tests
        MCTS tmax = 300s
        print(output.count('pawn'))
        print(output.count('knight'))
        print(output.count('bishop'))
        print(output.count('rook'))
        output:
        280298
        309210
        630841
        940329
        '''

        if pos is not None:
            king = pos

        if abs(king[0] - opp_king[0]) < 2 and abs(king[1] - opp_king[1]) < 2:
            return True

        # row/column: rooks or queens
        for rook in rooks + queens:
            if king[0] == rook[0] or king[1] == rook[1]:
                dist_steps = max(abs(king[0] - rook[0]), abs(king[1] - rook[1]))
                if dist_steps == 1: # directly next to each other
                    # print('rook')
                    return True
                check = True
                for dist in range(1, dist_steps): # 1:dist-1
                    if self.pieces[king[0] + dist * np.sign(rook[0] - king[0]), king[1] + dist * np.sign(rook[1] - king[1])] !=0:
                        check = False
                        break
                if check:
                    # print('rook')
                    return True

        # diagonal: bishops or queens
        for bishop in bishops + queens:
            if king[0]-king[1] == bishop[0]-bishop[1] or king[0]+king[1] == bishop[0]+bishop[1]:
                dist_steps = abs(king[0] - bishop[0])
                if dist_steps == 1: # directly next to each other
                    # print('bishop')
                    return True
                check = True
                for dist in range(1, dist_steps): # 1:dist-1
                    if self.pieces[king[0] + dist * np.sign(bishop[0] - king[0]), king[1] + dist * np.sign(bishop[1] - king[1])] !=0:
                        check = False
                        break
                if check:
                    # print('bishop')
                    return True
                
        # pawns
        if king[0] > 0:
            if king[1] > 0:
                if (king[0]-1, king[1]-1) in pawns:
                    # print('pawn')
                    return True
            if king[1] < 7:
                if (king[0]-1, king[1]+1) in pawns:
                    # print('pawn')
                    return True

        # knights 
        for knight in knights:
            if (abs(knight[0] - king[0]) == 1 and abs(knight[1] - king[1]) == 2) or (abs(knight[0] - king[0]) == 2 and abs(knight[1] - king[1])) == 1:
                # print('knight')
                return True
        
        return False

    def InCheck(self, pos = None):
        '''
        Return whether we are in check.

        New version to be faster. 
        Dont use Flip + Possbile Moves + Flip
        Instead directly check all pieces from opponent

        option using pos = a given position [x, y]
        to test n different position than king.
        Used to test where king can go

        '''

        if pos is not None:
            king = pos
        
        else:
            king = np.where(self.pieces == 6)
            king = np.array(king).flatten()

        # its important to include the case of touching kings in this function
        # precisely to discard such cases
        opp_king = np.where(self.pieces == 16)
        opp_king = np.array(opp_king).flatten()

        if abs(king[0] - opp_king[0]) < 2 and abs(king[1] - opp_king[1]) < 2:
            return True

        # pawns
        if king[0] > 0:
            if king[1] > 0:
                if self.pieces[king[0]-1, king[1]-1] == 11:
                    return True
            if king[1] < 7:
                if self.pieces[king[0]-1, king[1]+1] == 11:
                    return True

        # knights 
        #if np.any(self.pieces == 12): # if any knights are present at all
        if True:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                x_test = king[0] + x_off
                y_test = king[1] + y_off
                #if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                if self.OnBoard((x_test, y_test)):
                    if self.pieces[x_test, y_test] == 12:
                        return True
                
        rem_pieces = [13, 14] # remaining pieces: bishops, rooks, quuens
        directions_rem_pieces = [
            [(1, 1), (-1, 1), (1, -1), (-1, -1)], # directions for bishop to move
            [(1, 0), (-1, 0), (0, 1), (0, -1)], # directions for rook to move
        ]

        # for each piece type start at king position and follow its direction until 
        # we find a opponent's piece of this type -> in check
        # or we find another piece or the edge of the board -> not in check
        for type in range(2):
            #if np.any(self.pieces in [rem_pieces[type], 15]): # if any of the considered type are present
            if True:
                test_piece = rem_pieces[type]
                for dir in directions_rem_pieces[type]:
                    x_test = king[0]
                    y_test = king[1]
                    for _ in range(1, 8):
                        x_test += dir[0]
                        y_test += dir[1]
                        # if we hit such a piece -> in check
                        # otherwise go to next direction
                        #if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                        if self.OnBoard((x_test, y_test)):
                            if self.pieces[x_test, y_test] == test_piece: # found piece
                                return True 
                            elif self.pieces[x_test, y_test] == 15: # found queen
                                return True
                            elif self.pieces[x_test, y_test] != 0:
                                break
                        else:
                            break
    
        return False

    def ThreatTest(self, pos):
        '''
        Test what other pieces the one at the given position is threatning, 
        i.e. could take in the next move
        '''

        dict = {11: 1, 12: 3, 13: 3, 14: 5, 15: 9, 16:0}

        threats = []

        # pawn
        if self.pieces[pos] == 1:
            if pos[1] > 0 and pos[0] > 0: # attack to the left possible
                if self.pieces[pos[0]-1, pos[1]-1] > 10:
                    threats.append(self.pieces[pos[0]-1, pos[1]-1])
            if pos[1] < 7 and pos[0] > 0: # attack to the right possible
                if self.pieces[pos[0]-1, pos[1]+1] > 10:
                    threats.append(self.pieces[pos[0]-1, pos[1]+1])

        # knight
        if self.pieces[pos] == 2:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                x_test = pos[0] + x_off
                y_test = pos[1] + y_off
                if self.OnBoard((x_test, y_test)):
                    if self.pieces[x_test, y_test] > 10:
                        threats.append(self.pieces[x_test, y_test])

        # bishop or queen
        if self.pieces[pos] == 3 or self.pieces[pos] == 5:
            for dir in [(1, 1), (-1, 1), (1, -1), (-1, -1)]:
                x_test = pos[0]
                y_test = pos[1]
                for _ in range(1, 8):
                    x_test += dir[0]
                    y_test += dir[1]
                    if self.OnBoard((x_test, y_test)):
                        if self.pieces[x_test, y_test] > 10: # found opponents piece
                            threats.append(self.pieces[x_test, y_test])
                        if self.pieces[x_test, y_test] != 0 : # other (our) piece -> stop this dir
                            break
                    else: # off board -> stop this dir
                        break

        # rook or queen
        if self.pieces[pos] == 4 or self.pieces[pos] == 5:
            for dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x_test = pos[0]
                y_test = pos[1]
                for _ in range(1, 8):
                    x_test += dir[0]
                    y_test += dir[1]
                    if self.OnBoard((x_test, y_test)):
                        if self.pieces[x_test, y_test] > 10: # found opponents piece
                            threats.append(self.pieces[x_test, y_test])
                        if self.pieces[x_test, y_test] != 0 : # other (our) piece -> stop this dir
                            break
                    else: # off board -> stop this dir
                        break

        # king
        # note: checking for move legality is not done here
        if self.pieces[pos] == 6: 
            for x_off in [-1, 0, 1]:
                for y_off in [-1, 0, 1]:
                    x_test = pos[0] + x_off
                    y_test = pos[1] + y_off
                    if self.OnBoard((x_test, y_test)):
                        if self.pieces[x_test, y_test] > 10: # found opponents piece
                            threats.append(self.pieces[x_test, y_test])

        for i in range(len(threats)):
            threats[i] = dict[threats[i]]
                    
        return threats

    def InCheck_bitmap(self, bitmap = None):
        '''
        Return whether we are in check.

        Bitmap version
        '''

        if bitmap is None:
            bitmap = board_to_bitmap(self.pieces)

        # bitmap[0]: 1
        # bitmap[1]: 2 
        # bitmap[2]: 3
        # bitmap[3]: 4
        # bitmap[4]: 5
        # bitmap[5]: 6
        # bitmap[6]: 11
        # bitmap[7]: 12
        # bitmap[8]: 13
        # bitmap[9]: 14
        # bitmap[10]: 15
        # bitmap[11]: 16

        king = np.where(bitmap[5])
        king = np.array(king).flatten()

        # its important to include the case of touching kings in this function
        # precisely to discard such cases
        opp_king = np.where(bitmap[11])
        opp_king = np.array(opp_king).flatten()

        if abs(king[0] - opp_king[0]) < 2 and abs(king[1] - opp_king[1]) < 2:
            return True
        
        # pawns
        if king[0] > 0:
            if king[1] > 0:
                if bitmap[6, king[0]-1, king[1]-1]:
                    return True
            if king[1] < 7:
                if bitmap[6, king[0]-1, king[1]+1]:
                    return True

        # knights 
        if np.any(bitmap[7]): # if any knights are present at all
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                x_test = king[0] + x_off
                y_test = king[1] + y_off
                #if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                if self.OnBoard((x_test, y_test)):
                    if bitmap[7, x_test, y_test]:
                        return True
                
        rem_pieces = [8, 9] # bitmap index of remaining pieces: bishops, rooks, quuens
        directions_rem_pieces = [
            [(1, 1), (-1, 1), (1, -1), (-1, -1)], # directions for bishop to move
            [(1, 0), (-1, 0), (0, 1), (0, -1)], # directions for rook to move
        ]

        # for each piece type start at king position and follow its direction until 
        # we find a opponent's piece of this type -> in check
        # or we find another piece or the edge of the board -> not in check
        for type in range(2):
            if np.any(bitmap[rem_pieces[type]]) or np.any(bitmap[10]): # if any of the considered type are present
                for dir in directions_rem_pieces[type]:
                    x_test = king[0]
                    y_test = king[1]
                    for _ in range(1, 8):
                        x_test += dir[0]
                        y_test += dir[1]
                        # if we hit such a piece or a queen-> in check
                        # otherwise go to next direction
                        #if x_test >= 0 and x_test <= 7 and y_test >= 0 and y_test <= 7:
                        if self.OnBoard((x_test, y_test)):
                            if bitmap[rem_pieces[type], x_test, y_test]:
                                return True
                            elif bitmap[10, x_test, y_test]:
                                return True
                            elif np.any(bitmap[:, x_test, y_test]):
                                break
                        else:
                            break
    
        return False
    
    def InCheck_Julia(self):
        '''
        provide handle to julia call
        '''

        return Main.InCheck(self.pieces)

    def PlayMove(self, move):

        # pawn move or piece capture resets draw by moves counter
        if self.pieces[move[1]] != 0 or move[2] == 'pawn' or move[2][0:4] == 'prom':
            self.counter_draw_by_moves = 0
        else:
            self.counter_draw_by_moves += 1

        # execute move, handle special moves first, then handle normal move

        # promotions to different pieces
        if move[2][0:4] == 'prom': 
            if self.pieces[move[1]] != 0:
                self.latest_capture = self.pieces[move[1]]
            if move[2] == 'promotion knight':
                self.pieces[move[0]] = 0
                self.pieces[move[1]] = 2
            if move[2] == 'promotion bishop':
                self.pieces[move[0]] = 0
                self.pieces[move[1]] = 3
            if move[2] == 'promotion rook':
                self.pieces[move[0]] = 0
                self.pieces[move[1]] = 4
            if move[2] == 'promotion queen':
                self.pieces[move[0]] = 0
                self.pieces[move[1]] = 5

        # castle
        elif move[2] == 'castle long':
            if self.turn == 'white':
                self.pieces[7, 0] = 0
                self.pieces[7, 2] = 6
                self.pieces[7, 3] = 4
                self.pieces[7, 4] = 0
                self.can_castle_white = False
            if self.turn == 'black':
                self.pieces[7, 7] = 0
                self.pieces[7, 5] = 6
                self.pieces[7, 4] = 4
                self.pieces[7, 3] = 0
                self.can_castle_black = False
                
        elif move[2] == 'castle short':
            if self.turn == 'white':
                self.pieces[7, 7] = 0
                self.pieces[7, 6] = 6
                self.pieces[7, 5] = 4
                self.pieces[7, 4] = 0
                self.can_castle_white = False
            if self.turn == 'black':
                self.pieces[7, 0] = 0
                self.pieces[7, 1] = 6
                self.pieces[7, 2] = 4
                self.pieces[7, 3] = 0
                self.can_castle_black = False
            
        # en passent
        elif move[2] == 'en passant':
            self.pieces[move[0]] = 0
            self.pieces[move[1]] = 1
            self.pieces[move[1][0]+1, move[1][1]] = 0 # capturing opponent's pawn en passant

        # normal move, possible a capture
        else: 
            self.latest_capture = self.pieces[move[1]]
            piece = self.pieces[move[0]]
            self.pieces[move[0]] = 0
            self.pieces[move[1]] = piece

        if move[2] == 'pawn double' and abs(move[0][0] - move[1][0])==2: # we make a pawn move that might allow en passant
            self.pawn_to_be_en_passant = move[1]
        else:
            self.pawn_to_be_en_passant = None

        if move[2] == 'rook' and move[0] == (7, 0): # rook move from initial position removes right to castle
            if self.turn == 'white' and self.can_castle_white_long:
                self.can_castle_white_long = False
            if self.turn == 'black' and self.can_castle_black_short:
                self.can_castle_black_short = False
        if move[2] == 'rook' and move[0] == (7, 7): # rook move from initial position removes right to castle
            if self.turn == 'white' and self.can_castle_white_short:
                self.can_castle_white_short = False
            if self.turn == 'black' and self.can_castle_black_long:
                self.can_castle_black_long = False
        
        if move[2] == 'king' and move[0] == (7, 4) and self.turn == 'white': # king move from initial position removes right to castle
            self.can_castle_white_long = False
            self.can_castle_white_short = False

        if move[2] == 'king' and move[0] == (7, 3) and self.turn == 'black': # king move from initial position removes right to castle
            self.can_castle_black_long = False
            self.can_castle_black_short = False

        return

    def Reward(self):
        '''
        Reward function:
            return -1 on loss
            return 1  on win
            return 0  otherwise (undecided or stalemate)
        '''

        moves = self.PossibleMoves()
        if len(moves) == 0: # no moves model 1, stalemate or checkmate
            checkmate = self.InCheck()
            if checkmate:
                return -1000
            else:
                return 0
            
        self.FlipBoard()
        moves_opponent = self.PossibleMoves()
        if len(moves_opponent) == 0: # no moves model 2, stalemate or checkmate
            checkmate = self.InCheck()
            if checkmate:
                self.FlipBoard()
                return 1000
            else:
                self.FlipBoard()
                return 0
        self.FlipBoard()

        return self.MaterialDiff() #max(0, 0.1 * self.MaterialDiff())

    def MaterialDiff(self):
        dict = {0: 0,
                 1:  1,  2:  3,  3:  3,  4:  5,  5:  9,  6:0, 
                11: -1, 12: -3, 13: -3, 14: -5, 15: -9, 16:0}
        diff = 0

        for x in range(8):
            for y in range(8):
                piece = int(self.pieces[x, y])
                diff += dict[piece]

        return diff

    def PossibleMoves(self):
        '''
        version of possiblemoves where positions of opponent's pieces are given to incheck
        '''
        # print('possible moves call mode = ', mode)
        possible_moves = [] # collect all possible moves as tuples (from, to)

        playable_moves = []

        # check for pawns
        pawns = []
        knights = []
        bishops = []
        rooks = []
        queens = []
        opp_pawns = []
        opp_knights = []
        opp_bishops = []
        opp_rooks = []
        opp_queens = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 1:
                    pawns.append((y, x))
                elif self.pieces[y, x] == 2:
                    knights.append((y, x))
                elif self.pieces[y, x] == 3:
                    bishops.append((y, x))
                elif self.pieces[y, x] == 4:
                    rooks.append((y, x))
                elif self.pieces[y, x] == 5:
                    queens.append((y, x))
                elif self.pieces[y, x] == 6:
                    king = (y, x)
                elif self.pieces[y, x] == 11:
                    opp_pawns.append((y, x))
                elif self.pieces[y, x] == 12:
                    opp_knights.append((y, x))
                elif self.pieces[y, x] == 13:
                    opp_bishops.append((y, x))
                elif self.pieces[y, x] == 14:
                    opp_rooks.append((y, x))
                elif self.pieces[y, x] == 15:
                    opp_queens.append((y, x))
                elif self.pieces[y, x] == 16:
                    opp_king = (y, x)

        incheck_before = self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                               opp_queens, king, opp_king)

        for pawn in pawns:
            test_pos_1 = (pawn[0]-1, pawn[1]+1)
            test_pos_2 = (pawn[0]-1, pawn[1]-1)
            for test_pos in [test_pos_1, test_pos_2]:
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 10: # opponent piece
                        if pawn[0] == 1: # can promote
                                possible_moves.append((pawn, test_pos, 'promotion knight'))
                                # possible_moves.append((pawn, test_pos_3, 'promotion bishop'))
                                # possible_moves.append((pawn, test_pos_3, 'promotion rook'))
                                possible_moves.append((pawn, test_pos, 'promotion queen'))
                        else:
                            possible_moves.append((pawn, test_pos, 'pawn'))

            test_pos_3 = (pawn[0]-1, pawn[1])
            if self.pieces[test_pos_3] == 0: # field ahead empty
                if pawn[0] == 6: # next field also empty, can move double
                    test_pos_4 = (pawn[0]-2, pawn[1])
                    if self.pieces[test_pos_4]==0:
                        possible_moves.append((pawn, test_pos_4, 'pawn double'))
                if pawn[0] == 1: # can promote
                        possible_moves.append((pawn, test_pos_3, 'promotion knight'))
                        # possible_moves.append((pawn, test_pos_3, 'promotion bishop'))
                        # possible_moves.append((pawn, test_pos_3, 'promotion rook'))
                        possible_moves.append((pawn, test_pos_3, 'promotion queen'))
                else:
                    possible_moves.append((pawn, test_pos_3, 'pawn'))
            
            if pawn[0] == 3:
                if self.pawn_to_be_en_passant == (3, pawn[1]-1):
                    possible_moves.append((pawn, (2, pawn[1]-1 ), 'en passant'))
                if self.pawn_to_be_en_passant == (3, pawn[1]+1):
                    possible_moves.append((pawn, (2, pawn[1]+1 ), 'en passant'))
            
        # check for knights
        for knight in knights:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                test_pos = (knight[0]+y_off, knight[1]+x_off)
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] == 0 or self.pieces[test_pos] > 10:
                        possible_moves.append((knight, test_pos, 'knight'))

        # check for bishops
        for bishop in bishops:
            for x_dir, y_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (bishop[0]+y_dir*scale, bishop[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        
        # check for rooks
        for rook in rooks:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for scale in range(1, 8):
                    test_pos = (rook[0]+y_dir*scale, rook[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((rook, test_pos, 'rook'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((rook, test_pos, 'rook'))

        # check for queens
        for queen in queens:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (queen[0]+y_dir*scale, queen[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((queen, test_pos, 'queen'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((queen, test_pos, 'queen'))

        # check for king
        for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            test_pos = (king[0]+y_dir, king[1]+x_dir)
            if self.OnBoard(test_pos):
                if self.pieces[test_pos] > 10 or self.pieces[test_pos] == 0: # opponent piece or empty
                    # simulate move and see if in check
                    backup_piece = self.pieces[test_pos]

                    if backup_piece == 11:
                        opp_pawns.remove((test_pos))
                    elif backup_piece == 12:
                        opp_knights.remove((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.remove((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.remove((test_pos))
                    elif backup_piece == 15:
                        opp_queens.remove((test_pos))

                    self.pieces[king] = 0
                    self.pieces[test_pos] = 6
                    if not self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                               opp_queens, test_pos, opp_king):
                        playable_moves.append((king, test_pos, 'king'))
                    self.pieces[king] = 6
                    self.pieces[test_pos] = backup_piece

                    if backup_piece == 11:
                        opp_pawns.append((test_pos))
                    elif backup_piece == 12:
                        opp_knights.append((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.append((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.append((test_pos))
                    elif backup_piece == 15:
                        opp_queens.append((test_pos))

        if not incheck_before:
            if self.can_castle_white_short and self.turn == 'white':
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)) and not self.InCheck(pos=(7, 6)): # opponent doesn't cover squares
                        playable_moves.append(((7, 4), (7, 6), 'castle short'))
    
            if self.can_castle_white_long and self.turn == 'white': 
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 3)] == 0 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)): # opponent doesn't cover squares
                        playable_moves.append(((7, 4), (7, 2), 'castle long'))

            if self.can_castle_black_short and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)) and not self.InCheck(pos=(7, 1)): # opponent doesn't cover squares
                        playable_moves.append(((7, 3), (7, 1), 'castle short'))
    
            if self.can_castle_black_long and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 4)] == 0 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)): # opponent doesn't cover squares
                        playable_moves.append(((7, 3), (7, 5), 'castle long'))

        # go through all moves and see if they leave us in check. Only playable when no
        for move in possible_moves:

            playable = False
            # discard trivial cases right away to make execution faster
            row         = True if king[0] == move[0][0] else False
            column      = True if king[1] == move[0][1] else False
            diag_down   = True if king[0]-king[1] == move[0][0]-move[0][1] else False
            diag_up     = True if king[0]+king[1] == move[0][0]+move[0][1] else False

            if not (row or column or diag_down or diag_up) and not incheck_before: # move[0] shares no line with king -> certainly playable
                playable = True

            if not playable: 

                if 'prom' in move[2] or move[2] == 'en passant' or 'castle' in move[2]:
                    # general playability test

                    # backup game instance
                    new_game = self.copy()

                    self.PlayMove(move)
                    incheck = self.InCheck()

                    # restore game from backup
                    self.pieces = new_game.pieces.copy()
                    self.turn = new_game.turn
                    self.can_castle_white_long = new_game.can_castle_white_long
                    self.can_castle_white_short = new_game.can_castle_white_short
                    self.can_castle_black_long = new_game.can_castle_black_long
                    self.can_castle_black_short = new_game.can_castle_black_short
                    self.pawn_to_be_en_passant = new_game.pawn_to_be_en_passant

                else:
                    # prepare env for in check test
                    moved_piece = self.pieces[move[0]]
                    captured_piece = self.pieces[move[1]]

                    if captured_piece == 11:
                        opp_pawns.remove(move[1])
                    elif captured_piece == 12:
                        opp_knights.remove(move[1])
                    elif captured_piece == 13:
                        opp_bishops.remove(move[1])
                    elif captured_piece == 14:
                        opp_rooks.remove(move[1])
                    elif captured_piece == 15:
                        opp_queens.remove(move[1])
                    elif moved_piece == 6:
                        king = move[1]

                    self.pieces[move[0]] = 0
                    self.pieces[move[1]] = moved_piece

                    # perform in check test
                    incheck = self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                               opp_queens, king, opp_king)

                    # reset env to state before
                    self.pieces[move[0]] = moved_piece
                    self.pieces[move[1]] = captured_piece

                    if captured_piece == 11:
                        opp_pawns.append(move[1])
                    elif captured_piece == 12:
                        opp_knights.append(move[1])
                    elif captured_piece == 13:
                        opp_bishops.append(move[1])
                    elif captured_piece == 14:
                        opp_rooks.append(move[1])
                    elif captured_piece == 15:
                        opp_queens.append(move[1])
                    elif moved_piece == 6:
                        king = move[0] 

                if not incheck:
                    playable = True

            if playable:
                if move[2]=='pawn' and move[1][0] == 0: # move onto last row
                    for piece in ['knight', 'bishop', 'rook', 'queen']:
                        playable_moves.append((move[0], move[1], 'promotion ' + piece))
                playable_moves.append(move)

        return playable_moves

    def PossibleMoves_no_pieces(self):
        # print('possible moves call mode = ', mode)
        possible_moves = [] # collect all possible moves as tuples (from, to)

        incheck_before = self.InCheck()
        playable_moves = []
        board_backup = self.pieces.copy()

        # check for pawns
        pawns = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 1:
                    pawns.append((y, x))
        for pawn in pawns:
            test_pos_1 = (pawn[0]-1, pawn[1]+1)
            test_pos_2 = (pawn[0]-1, pawn[1]-1)
            for test_pos in [test_pos_1, test_pos_2]:
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((pawn, test_pos, 'pawn'))

            test_pos_3 = (pawn[0]-1, pawn[1])
            if self.pieces[test_pos_3] == 0: # field ahead empty
                possible_moves.append((pawn, test_pos_3, 'pawn'))
                if pawn[0] == 6: # next field also empty, can move double
                    test_pos_4 = (pawn[0]-2, pawn[1])
                    if self.pieces[test_pos_4]==0:
                        possible_moves.append((pawn, test_pos_4, 'pawn double'))
                if pawn[0] == 1: # can promote
                        possible_moves.append((pawn, test_pos_3, 'promotion knight'))
                        possible_moves.append((pawn, test_pos_3, 'promotion bishop'))
                        possible_moves.append((pawn, test_pos_3, 'promotion rook'))
                        possible_moves.append((pawn, test_pos_3, 'promotion queen'))
            
            if pawn[0] == 3:
                if self.pawn_to_be_en_passant == (3, pawn[1]-1):
                    possible_moves.append((pawn, (2, pawn[1]-1 )), 'en passant')
                if self.pawn_to_be_en_passant == (3, pawn[1]+1):
                    possible_moves.append((pawn, (2, pawn[1]+1 )), 'en passant')
            
        # check for knights
        knights = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 2:
                    knights.append((y, x))
        for knight in knights:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                test_pos = (knight[0]+y_off, knight[1]+x_off)
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] == 0 or self.pieces[test_pos] > 10:
                        possible_moves.append((knight, test_pos, 'knight'))

        # check for bishops
        bishops = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 3:
                    bishops.append((y, x))
        for bishop in bishops:
            for x_dir, y_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (bishop[0]+y_dir*scale, bishop[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        

        # check for rooks
        rooks = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 4:
                    rooks.append((y, x))
        for rook in rooks:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for scale in range(1, 8):
                    test_pos = (rook[0]+y_dir*scale, rook[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((rook, test_pos, 'rook'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((rook, test_pos, 'rook'))

        # check for queens
        queens = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 5:
                    queens.append((y, x))
        for queen in queens:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (queen[0]+y_dir*scale, queen[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((queen, test_pos, 'queen'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((queen, test_pos, 'queen'))

        # check for king
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 6:
                    king = ((y, x))
                    break
        for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            test_pos = (king[0]+y_dir, king[1]+x_dir)
            if self.OnBoard(test_pos):
                if self.pieces[test_pos] > 10 or self.pieces[test_pos] == 0: # opponent piece or empty
                    # simulate move and see if in check
                    backup_piece = self.pieces[test_pos]
                    self.pieces[king] = 0
                    self.pieces[test_pos] = 6
                    if not self.InCheck():
                        playable_moves.append((king, test_pos, 'king'))
                    self.pieces[king] = 6
                    self.pieces[test_pos] = backup_piece

        if not incheck_before:
            if self.can_castle_white_short and self.turn == 'white':
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)) and not self.InCheck(pos=(7, 6)): # opponent doesn't cover squares
                        playable_moves.append(((7, 4), (7, 6), 'castle short'))
    
            if self.can_castle_white_long and self.turn == 'white': 
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 3)] == 0 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)): # opponent doesn't cover squares
                        playable_moves.append(((7, 4), (7, 2), 'castle long'))

            if self.can_castle_black_short and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)) and not self.InCheck(pos=(7, 1)): # opponent doesn't cover squares
                        playable_moves.append(((7, 3), (7, 1), 'castle short'))
    
            if self.can_castle_black_long and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 4)] == 0 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)): # opponent doesn't cover squares
                        playable_moves.append(((7, 3), (7, 5), 'castle long'))

        # go through all moves and see if they leave us in check. Only playable when no

        for move in possible_moves:

            playable = False
            # discard trivial cases right away to make execution faster
            row         = True if king[0] == move[0][0] else False
            column      = True if king[1] == move[0][1] else False
            diag_down   = True if king[0]-king[1] == move[0][0]-move[0][1] else False
            diag_up     = True if king[0]+king[1] == move[0][0]+move[0][1] else False

            if not (row or column or diag_down or diag_up) and not incheck_before: # move[0] shares no line with king -> certainly playable
                playable = True

            if not playable: # otherwise, lengthy check 

                self.PlayMove(move)

                incheck = self.InCheck()
                self.pieces = board_backup.copy()

                if not incheck:
                    playable = True

            if playable:
                if move[2]=='pawn' and move[1][0] == 0: # move onto last row
                    for piece in ['knight', 'bishop', 'rook', 'queen']:
                        playable_moves.append((move[0], move[1], 'promotion ' + piece))
                playable_moves.append(move)

        return playable_moves

    def PossibleMoves_bitmap(self, mode='actual', prints=False):
        possible_moves = [] # collect all possible moves as tuples (from, to)
        own_covered = []    # collect all pieces that are covered (for opponent's possible king moves)
        king_threatened = False # we could capture opponent's king, i.e. he is in check

        bitmap = board_to_bitmap(self.pieces)

        # check for pawns
        pawns = np.where(bitmap[0])
        pawns = [(i, j) for i,j in zip(pawns[0], pawns[1])]
        for pawn in pawns:
            test_pos_1 = (pawn[0]-1, pawn[1]+1)
            test_pos_2 = (pawn[0]-1, pawn[1]-1)
            for test_pos in [test_pos_1, test_pos_2]:
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((pawn, test_pos, 'pawn'))
                    if self.pieces[test_pos] < 10: # own piece or empty
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
            test_pos_3 = (pawn[0]-1, pawn[1])
            if self.pieces[test_pos_3] == 0: # field ahead empty
                possible_moves.append((pawn, test_pos_3, 'pawn'))
                if pawn[0] == 6: # next field also empty, can move double
                    test_pos_4 = (pawn[0]-2, pawn[1])
                    if self.pieces[test_pos_4]==0:
                        possible_moves.append((pawn, test_pos_4, 'pawn double'))
                if pawn[0] == 1: # can promote
                        possible_moves.append((pawn, test_pos_3, 'promotion knight'))
                        possible_moves.append((pawn, test_pos_3, 'promotion bishop'))
                        possible_moves.append((pawn, test_pos_3, 'promotion rook'))
                        possible_moves.append((pawn, test_pos_3, 'promotion queen'))
            
            if pawn[0] == 3:
                if self.pawn_to_be_en_passant == (3, pawn[1]-1):
                    possible_moves.append((pawn, (2, pawn[1]-1 )), 'en passant')
                if self.pawn_to_be_en_passant == (3, pawn[1]+1):
                    possible_moves.append((pawn, (2, pawn[1]+1 )), 'en passant')
            
        # check for knights
        knights = np.where(bitmap[1])
        knights = [(i, j) for i,j in zip(knights[0], knights[1])]
        for knight in knights:
            for x_off in [-2, -1, 1, 2]:
                for y_off in [-2, -1, 1, 2]:
                    if abs(x_off)+abs(y_off) != 3:
                        continue
                    test_pos = (knight[0]+y_off, knight[1]+x_off)
                    if self.OnBoard(test_pos):
                        if self.pieces[test_pos] == 0 or self.pieces[test_pos] > 10:
                            possible_moves.append((knight, test_pos, 'knight'))
                        if self.pieces[test_pos] < 10: # own piece or empty
                            if test_pos not in own_covered:
                                own_covered.append(test_pos)

        # check for bishops
        bishops = np.where(bitmap[2])
        bishops = [(i, j) for i,j in zip(bishops[0], bishops[1])]
        for bishop in bishops:
            for x_dir, y_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (bishop[0]+y_dir*scale, bishop[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
                        
        # check for rooks
        rooks = np.where(bitmap[3])
        rooks = [(i, j) for i,j in zip(rooks[0], rooks[1])]
        for rook in rooks:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for scale in range(1, 8):
                    test_pos = (rook[0]+y_dir*scale, rook[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((rook, test_pos, 'rook'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((rook, test_pos, 'rook'))
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)

        # check for queens
        queens = np.where(bitmap[4])
        queens = [(i, j) for i,j in zip(queens[0], queens[1])]
        for queen in queens:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (queen[0]+y_dir*scale, queen[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off baord
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((queen, test_pos, 'queen'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((queen, test_pos, 'queen'))
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
        
        if mode=='covered_only':
            # check for king
            for x in range(8):
                for y in range(8):
                    if self.pieces[y, x] == 6:
                        king = ((y, x))
                        break
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                try:
                    test_pos = (king[0]+y_dir, king[1]+x_dir)
                except:
                    print('king not found on board: ')
                    print(self.pieces)
                if self.OnBoard(test_pos): # off baord
                    if self.pieces[test_pos] < 10: # own piece or empty
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
            return own_covered
        
        if mode=='check detection':
            return possible_moves


        else:
            # check for king
            # find which pieces to opponent covers, i.e. where our king cannot go
            self.FlipBoard()
            opponent_covered = self.FlipPositions(self.PossibleMoves(mode='covered_only'))
            self.FlipBoard()

            king = -1
            for x in range(8):
                for y in range(8):
                    if self.pieces[y, x] == 6:
                        king = ((y, x))
                        break

            if king == -1:
                print('no king')
                print(self.pieces)
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                test_pos = (king[0]+y_dir, king[1]+x_dir)
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        if test_pos not in own_covered:
                            own_covered.append(test_pos)
                    elif self.pieces[test_pos] > 10 or self.pieces[test_pos] == 0: # opponent piece or empty
                        if test_pos not in opponent_covered:
                            possible_moves.append((king, test_pos, 'king'))
            
            if self.can_castle_white_short and self.turn == 'white':
                if (7, 4) not in opponent_covered and (7, 5) not in opponent_covered and (7, 6) not in opponent_covered: # opponent doesn't cover squares
                    if self.pieces[(7, 4)] == 6 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                        possible_moves.append(((7, 4), (7, 6), 'castle short'))
    
            if self.can_castle_white_long and self.turn == 'white':
                if (7, 4) not in opponent_covered and (7, 3) not in opponent_covered and (7, 2) not in opponent_covered: # opponent doesn't cover squares
                    if self.pieces[(7, 4)] == 6 and self.pieces[(7, 3)] == 0 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 0)] == 4:
                        possible_moves.append(((7, 4), (7, 2), 'castle long'))

            if self.can_castle_black_short and self.turn == 'black':
                if (7, 3) not in opponent_covered and (7, 2) not in opponent_covered and (7, 1) not in opponent_covered: # opponent doesn't cover squares
                    if self.pieces[(7, 3)] == 6 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                        possible_moves.append(((7, 3), (7, 1), 'castle short'))
    
            if self.can_castle_black_long and self.turn == 'black':
                if (7, 3) not in opponent_covered and (7, 4) not in opponent_covered and (7, 5) not in opponent_covered: # opponent doesn't cover squares
                    if self.pieces[(7, 3)] == 6 and self.pieces[(7, 4)] == 0 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 7)] == 4:
                        possible_moves.append(((7, 3), (7, 5), 'castle long'))
            
        playable_moves = []
        board_backup = self.pieces.copy()
        incheck_before = self.InCheck_bitmap(bitmap=bitmap)

        # go through all moves and see if they leave us in check. Only playable when no
        for move in possible_moves:

            # print('test playability of move: ', move)

            playable = False

            # discard trivial cases right away to make execution faster
            row         = True if king[0] == move[0][0] else False
            column      = True if king[1] == move[0][1] else False
            diag_down   = True if king[0]-king[1] == move[0][0]-move[0][1] else False
            diag_up     = True if king[0]+king[1] == move[0][0]+move[0][1] else False

            if not (row or column or diag_down or diag_up) and not incheck_before: # move[0] shares no line with king -> certainly playable
                playable = True
                # print('after quick check: playable = ', playable)

            if not playable: # otherwise, lengthy check 

                # print('after quick check: playable = ', playable)
                self.PlayMove(move)
                # piece = self.pieces[move[0]]
                # self.pieces[move[0]] = 0
                # self.pieces[move[1]] = piece


                bitmap = board_to_bitmap(self.pieces)
                incheck = self.InCheck_bitmap(bitmap=bitmap)
                self.pieces = board_backup.copy()

                if not incheck:
                    playable = True
                
                # print('after long check: playable = ', playable)

            if playable:
                if move[2]=='pawn' and move[1][0] == 0: # move onto last row
                    playable_moves.append((move[0], move[1], 'promotion knight'))
                    playable_moves.append((move[0], move[1], 'promotion bishop'))
                    playable_moves.append((move[0], move[1], 'promotion rook'))
                    playable_moves.append((move[0], move[1], 'promotion queen'))
                playable_moves.append(move)


        if prints:
            print('possible moves prints:')
            print('possible moves:')
            print(possible_moves)
            print('playble moves:')
            print(playable_moves)
 
        return playable_moves
    
    def one_step_greedy_efficient(self):
        '''
        one step greedy rewritten for efficiency
        modified version of PossibleMoves. 0
        store how much material gain we can get in one move. 
        discard all new moves immediatly that produce less gain than current best
        '''

        if self.counter_draw_by_moves > 50:
            print('draw by move count')
            return (None, None, None)

        possible_moves = [] # collect all possible moves as tuples (from, to)
        possible_gains = [] # collect how much gain a move brings (checkmate, capture, promotion)

        piece_to_value = {0:0, 11:1, 12:3, 13:3, 14:5, 15:9}

        # check for pawns
        pawns = []
        knights = []
        bishops = []
        rooks = []
        queens = []
        opp_pawns = []
        opp_knights = []
        opp_bishops = []
        opp_rooks = []
        opp_queens = []
        self.pawns = []
        self.knights = []
        self.bishops = []
        self.rooks = []
        self.queens = []
        self.opp_pawns = []
        self.opp_knights = []
        self.opp_bishops = []
        self.opp_rooks = []
        self.opp_queens = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 1:
                    pawns.append((y, x))
                    self.pawns.append((y, x))
                elif self.pieces[y, x] == 2:
                    knights.append((y, x))
                    self.knights.append((y, x))
                elif self.pieces[y, x] == 3:
                    bishops.append((y, x))
                    self.bishops.append((y, x))
                elif self.pieces[y, x] == 4:
                    rooks.append((y, x))
                    self.rooks.append((y, x))
                elif self.pieces[y, x] == 5:
                    queens.append((y, x))
                    self.queens.append((y, x))
                elif self.pieces[y, x] == 6:
                    king = (y, x)
                    self.king = (y, x)
                elif self.pieces[y, x] == 11:
                    opp_pawns.append((y, x))
                    self.opp_pawns.append((y, x))
                elif self.pieces[y, x] == 12:
                    opp_knights.append((y, x))
                    self.opp_knights.append((y, x))
                elif self.pieces[y, x] == 13:
                    opp_bishops.append((y, x))
                    self.opp_bishops.append((y, x))
                elif self.pieces[y, x] == 14:
                    opp_rooks.append((y, x))
                    self.opp_rooks.append((y, x))
                elif self.pieces[y, x] == 15:
                    opp_queens.append((y, x))
                    self.opp_queens.append((y, x))
                elif self.pieces[y, x] == 16:
                    opp_king = (y, x)
                    self.opp_king = (y, x)

        incheck_before = self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                                opp_queens, king, opp_king)

        for pawn in pawns:
            test_pos_1 = (pawn[0]-1, pawn[1]+1)
            test_pos_2 = (pawn[0]-1, pawn[1]-1)
            for test_pos in [test_pos_1, test_pos_2]:
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((pawn, test_pos, 'pawn'))
                        possible_gains.append(piece_to_value[self.pieces[test_pos]])

            test_pos_3 = (pawn[0]-1, pawn[1])
            if self.pieces[test_pos_3] == 0: # field ahead empty
                if pawn[0] != 1:
                    possible_moves.append((pawn, test_pos_3, 'pawn'))
                    possible_gains.append(0)
                if pawn[0] == 6: # next field also empty, can move double
                    test_pos_4 = (pawn[0]-2, pawn[1])
                    if self.pieces[test_pos_4]==0:
                        possible_moves.append((pawn, test_pos_4, 'pawn double'))
                        possible_gains.append(0)
                if pawn[0] == 1: # can promote
                        possible_moves.append((pawn, test_pos_3, 'promotion knight'))
                        possible_gains.append(2)
                        possible_moves.append((pawn, test_pos_3, 'promotion queen'))
                        possible_gains.append(8)
            
            if pawn[0] == 3:
                if self.pawn_to_be_en_passant == (3, pawn[1]-1):
                    possible_moves.append((pawn, (2, pawn[1]-1 ), 'en passant'))
                    possible_gains.append(1)
                if self.pawn_to_be_en_passant == (3, pawn[1]+1):
                    possible_moves.append((pawn, (2, pawn[1]+1 ), 'en passant'))
                    possible_gains.append(1)
            
        # check for knights
        for knight in knights:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                test_pos = (knight[0]+y_off, knight[1]+x_off)
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] == 0 or self.pieces[test_pos] > 10:
                        possible_moves.append((knight, test_pos, 'knight'))
                        possible_gains.append(piece_to_value[self.pieces[test_pos]])

        # check for bishops
        for bishop in bishops:
            for x_dir, y_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (bishop[0]+y_dir*scale, bishop[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        possible_gains.append(piece_to_value[self.pieces[test_pos]])
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        possible_gains.append(0)
                        
        # check for rooks
        for rook in rooks:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for scale in range(1, 8):
                    test_pos = (rook[0]+y_dir*scale, rook[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((rook, test_pos, 'rook'))
                        possible_gains.append(piece_to_value[self.pieces[test_pos]])
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((rook, test_pos, 'rook'))
                        possible_gains.append(0)

        # check for queens
        for queen in queens:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (queen[0]+y_dir*scale, queen[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((queen, test_pos, 'queen'))
                        possible_gains.append(piece_to_value[self.pieces[test_pos]])
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((queen, test_pos, 'queen'))
                        possible_gains.append(0)
        
        # checkmate test and playability test are expensive, avoid those

        for move in possible_moves:
            if self.check_test(move):
                if self.playable(move, incheck_before):
                    if self.checkmate_test(move):
                        return move

        gains = np.unique(possible_gains)
        for gain in reversed(gains):
            current_moves = [possible_moves[m] for m in range(len(possible_moves)) if possible_gains[m] == gain]
            np.random.shuffle(current_moves)
            for move in current_moves:
                if self.playable(move, incheck_before):
                    return move

        best_king_gain = -1
        best_king_move = None

        # check for king
        for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            test_pos = (king[0]+y_dir, king[1]+x_dir)
            if self.OnBoard(test_pos):
                if self.pieces[test_pos] > 10 or self.pieces[test_pos] == 0 and piece_to_value[self.pieces[test_pos]] > best_king_gain: # opponent piece or empty
                    # simulate move and see if in check

                    backup_piece = self.pieces[test_pos]

                    if backup_piece == 11:
                        opp_pawns.remove((test_pos))
                    elif backup_piece == 12:
                        opp_knights.remove((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.remove((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.remove((test_pos))
                    elif backup_piece == 15:
                        opp_queens.remove((test_pos))

                    self.pieces[king] = 0
                    self.pieces[test_pos] = 6
                    if not self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                                opp_queens, test_pos, opp_king):
                        best_king_move = (king, test_pos, 'king')
                        best_king_gain = piece_to_value[backup_piece]

                    self.pieces[king] = 6
                    self.pieces[test_pos] = backup_piece

                    if backup_piece == 11:
                        opp_pawns.append((test_pos))
                    elif backup_piece == 12:
                        opp_knights.append((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.append((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.append((test_pos))
                    elif backup_piece == 15:
                        opp_queens.append((test_pos))

        if best_king_gain > -1:
            return best_king_move

        if not incheck_before:
            if self.can_castle_white_short and self.turn == 'white':
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)) and not self.InCheck(pos=(7, 6)): # opponent doesn't cover squares
                        return ((7, 4), (7, 6), 'castle short')

            if self.can_castle_white_long and self.turn == 'white': 
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 3)] == 0 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)): # opponent doesn't cover squares
                        return ((7, 4), (7, 2), 'castle long')

            if self.can_castle_black_short and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)) and not self.InCheck(pos=(7, 1)): # opponent doesn't cover squares
                        return ((7, 3), (7, 1), 'castle short')

            if self.can_castle_black_long and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 4)] == 0 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)): # opponent doesn't cover squares
                        return ((7, 3), (7, 5), 'castle long')
                    
        return (None, None, None)
    
    def greedy_wrt(self, model):
        '''
        one step greedy. greedy with respect to given model. i.e. chose move that maximizes model output
        '''

        if self.counter_draw_by_moves > 50:
            print('draw by move count')
            return (None, None, None)

        possible_moves = [] # collect all possible moves as tuples (from, to)
        possible_values = [] # collect value of possible as output of given model

        board_backup = self.pieces.copy()

        # check for pawns
        pawns = []
        knights = []
        bishops = []
        rooks = []
        queens = []
        opp_pawns = []
        opp_knights = []
        opp_bishops = []
        opp_rooks = []
        opp_queens = []
        self.pawns = []
        self.knights = []
        self.bishops = []
        self.rooks = []
        self.queens = []
        self.opp_pawns = []
        self.opp_knights = []
        self.opp_bishops = []
        self.opp_rooks = []
        self.opp_queens = []
        for x in range(8):
            for y in range(8):
                if self.pieces[y, x] == 1:
                    pawns.append((y, x))
                    self.pawns.append((y, x))
                elif self.pieces[y, x] == 2:
                    knights.append((y, x))
                    self.knights.append((y, x))
                elif self.pieces[y, x] == 3:
                    bishops.append((y, x))
                    self.bishops.append((y, x))
                elif self.pieces[y, x] == 4:
                    rooks.append((y, x))
                    self.rooks.append((y, x))
                elif self.pieces[y, x] == 5:
                    queens.append((y, x))
                    self.queens.append((y, x))
                elif self.pieces[y, x] == 6:
                    king = (y, x)
                    self.king = (y, x)
                elif self.pieces[y, x] == 11:
                    opp_pawns.append((y, x))
                    self.opp_pawns.append((y, x))
                elif self.pieces[y, x] == 12:
                    opp_knights.append((y, x))
                    self.opp_knights.append((y, x))
                elif self.pieces[y, x] == 13:
                    opp_bishops.append((y, x))
                    self.opp_bishops.append((y, x))
                elif self.pieces[y, x] == 14:
                    opp_rooks.append((y, x))
                    self.opp_rooks.append((y, x))
                elif self.pieces[y, x] == 15:
                    opp_queens.append((y, x))
                    self.opp_queens.append((y, x))
                elif self.pieces[y, x] == 16:
                    opp_king = (y, x)
                    self.opp_king = (y, x)

        incheck_before = self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                                opp_queens, king, opp_king)

        for pawn in pawns:
            test_pos_1 = (pawn[0]-1, pawn[1]+1)
            test_pos_2 = (pawn[0]-1, pawn[1]-1)
            for test_pos in [test_pos_1, test_pos_2]:
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((pawn, test_pos, 'pawn'))

            test_pos_3 = (pawn[0]-1, pawn[1])
            if self.pieces[test_pos_3] == 0: # field ahead empty
                if pawn[0] != 1:
                    possible_moves.append((pawn, test_pos_3, 'pawn'))
                if pawn[0] == 6: # next field also empty, can move double
                    test_pos_4 = (pawn[0]-2, pawn[1])
                    if self.pieces[test_pos_4]==0:
                        possible_moves.append((pawn, test_pos_4, 'pawn double'))
                if pawn[0] == 1: # can promote
                        possible_moves.append((pawn, test_pos_3, 'promotion knight'))
                        possible_moves.append((pawn, test_pos_3, 'promotion queen'))
            
            if pawn[0] == 3:
                if self.pawn_to_be_en_passant == (3, pawn[1]-1):
                    possible_moves.append((pawn, (2, pawn[1]-1 ), 'en passant'))
                if self.pawn_to_be_en_passant == (3, pawn[1]+1):
                    possible_moves.append((pawn, (2, pawn[1]+1 ), 'en passant'))

        # check for knights
        for knight in knights:
            for x_off, y_off in [(2, 1), (1, 2), (-1, 2), (-2, 1), (-2, -1), (-1, -2), (1, -2), (2, -1)]:
                test_pos = (knight[0]+y_off, knight[1]+x_off)
                if self.OnBoard(test_pos):
                    if self.pieces[test_pos] == 0 or self.pieces[test_pos] > 10:
                        possible_moves.append((knight, test_pos, 'knight'))

        # check for bishops
        for bishop in bishops:
            for x_dir, y_dir in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (bishop[0]+y_dir*scale, bishop[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((bishop, test_pos, 'bishop'))
                        
        # check for rooks
        for rook in rooks:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                for scale in range(1, 8):
                    test_pos = (rook[0]+y_dir*scale, rook[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((rook, test_pos, 'rook'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((rook, test_pos, 'rook'))

        # check for queens
        for queen in queens:
            for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
                for scale in range(1, 8):
                    test_pos = (queen[0]+y_dir*scale, queen[1]+x_dir*scale)
                    if not self.OnBoard(test_pos): # off board
                        break
                    if self.pieces[test_pos] > 0 and self.pieces[test_pos] < 10: # own piece
                        break
                    if self.pieces[test_pos] > 10: # opponent piece
                        possible_moves.append((queen, test_pos, 'queen'))
                        break
                    if self.pieces[test_pos] == 0: # empty field
                        possible_moves.append((queen, test_pos, 'queen'))
        
        # checkmate test and playability test are expensive, avoid those and do check test first
        for move in possible_moves:
            if self.check_test(move):
                if self.playable(move, incheck_before):
                    if self.checkmate_test(move):
                        return move

        for move in possible_moves:
            self.PlayMove(move)
            board = torch.stack([board_to_tensor(self.pieces)])
            possible_values.append(model(board).detach().item())
            self.pieces = board_backup.copy()

        gains = np.unique(possible_values)
        for gain in reversed(gains):
            current_moves = [possible_moves[m] for m in range(len(possible_moves)) if possible_values[m] == gain]
            np.random.shuffle(current_moves)
            for move in current_moves:
                if self.playable(move, incheck_before):
                    return move

        best_king_value = -100
        best_king_move = None

        # check for king
        for x_dir, y_dir in [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            test_pos = (king[0]+y_dir, king[1]+x_dir)
            if self.OnBoard(test_pos):
                if self.pieces[test_pos] > 10 or self.pieces[test_pos] == 0: # opponent piece or empty
                    
                    
                    # simulate move and see if in check

                    backup_piece = self.pieces[test_pos]

                    if backup_piece == 11:
                        opp_pawns.remove((test_pos))
                    elif backup_piece == 12:
                        opp_knights.remove((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.remove((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.remove((test_pos))
                    elif backup_piece == 15:
                        opp_queens.remove((test_pos))

                    self.pieces[king] = 0
                    self.pieces[test_pos] = 6
                    if not self.InCheck_pieces(opp_pawns, opp_knights, opp_bishops, opp_rooks, 
                                                opp_queens, test_pos, opp_king):
                        
                        board = torch.stack([board_to_tensor(self.pieces)])
                        value = model(board).detach().item()

                        if value > best_king_value:
                            best_king_move = (king, test_pos, 'king')
                            best_king_value = value

                    self.pieces[king] = 6
                    self.pieces[test_pos] = backup_piece

                    if backup_piece == 11:
                        opp_pawns.append((test_pos))
                    elif backup_piece == 12:
                        opp_knights.append((test_pos))
                    elif backup_piece == 13:
                        opp_bishops.append((test_pos))
                    elif backup_piece == 14:
                        opp_rooks.append((test_pos))
                    elif backup_piece == 15:
                        opp_queens.append((test_pos))

        if best_king_value > -100:
            return best_king_move

        if not incheck_before:
            if self.can_castle_white_short and self.turn == 'white':
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 6)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)) and not self.InCheck(pos=(7, 6)): # opponent doesn't cover squares
                        return ((7, 4), (7, 6), 'castle short')

            if self.can_castle_white_long and self.turn == 'white': 
                if self.pieces[(7, 4)] == 6 and self.pieces[(7, 3)] == 0 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)): # opponent doesn't cover squares
                        return ((7, 4), (7, 2), 'castle long')

            if self.can_castle_black_short and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 2)] == 0 and self.pieces[(7, 1)] == 0 and self.pieces[(7, 0)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 2)) and not self.InCheck(pos=(7, 1)): # opponent doesn't cover squares
                        return ((7, 3), (7, 1), 'castle short')

            if self.can_castle_black_long and self.turn == 'black':
                if self.pieces[(7, 3)] == 6 and self.pieces[(7, 4)] == 0 and self.pieces[(7, 5)] == 0 and self.pieces[(7, 7)] == 4:
                    if not self.InCheck(pos=(7, 3)) and not self.InCheck(pos=(7, 4)) and not self.InCheck(pos=(7, 5)): # opponent doesn't cover squares
                        return ((7, 3), (7, 5), 'castle long')
                    
        return (None, None, None)

    def playable(self, move, incheck_before):
        '''
        check if move is playable
        '''

        # discard trivial cases right away to make execution faster
        row         = True if self.king[0] == move[0][0] else False
        column      = True if self.king[1] == move[0][1] else False
        diag_down   = True if self.king[0]-self.king[1] == move[0][0]-move[0][1] else False
        diag_up     = True if self.king[0]+self.king[1] == move[0][0]+move[0][1] else False

        if not (row or column or diag_down or diag_up) and not incheck_before: # move[0] shares no line with king -> certainly playable
            return True

        if 'prom' in move[2] or move[2] == 'en passant' or 'castle' in move[2]:
            # general playability test

            # backup game instance
            new_game = self.copy()

            self.PlayMove(move)
            incheck = self.InCheck()

            # restore game from backup
            self.pieces = new_game.pieces.copy()
            self.turn = new_game.turn
            self.can_castle_white_long = new_game.can_castle_white_long
            self.can_castle_white_short = new_game.can_castle_white_short
            self.can_castle_black_long = new_game.can_castle_black_long
            self.can_castle_black_short = new_game.can_castle_black_short
            self.pawn_to_be_en_passant = new_game.pawn_to_be_en_passant
        
        else:
            # prepare env for in check test
            moved_piece = self.pieces[move[0]]
            captured_piece = self.pieces[move[1]]

            if captured_piece == 11:
                self.opp_pawns.remove(move[1])
            elif captured_piece == 12:
                self.opp_knights.remove(move[1])
            elif captured_piece == 13:
                self.opp_bishops.remove(move[1])
            elif captured_piece == 14:
                self.opp_rooks.remove(move[1])
            elif captured_piece == 15:
                self.opp_queens.remove(move[1])
            elif moved_piece == 6:
                self.king = move[1]

            self.pieces[move[0]] = 0
            self.pieces[move[1]] = moved_piece

            # perform in check test
            incheck = self.InCheck_pieces(self.opp_pawns, self.opp_knights, self.opp_bishops, self.opp_rooks, self.opp_queens, self.king, self.opp_king)

            # reset env to state before
            self.pieces[move[0]] = moved_piece
            self.pieces[move[1]] = captured_piece

            if captured_piece == 11:
                self.opp_pawns.append(move[1])
            elif captured_piece == 12:
                self.opp_knights.append(move[1])
            elif captured_piece == 13:
                self.opp_bishops.append(move[1])
            elif captured_piece == 14:
                self.opp_rooks.append(move[1])
            elif captured_piece == 15:
                self.opp_queens.append(move[1])
            elif moved_piece == 6:
                self.king = move[0] 

            if not incheck:
                return True 
            
    def check_test(self, move):
        '''
        simple check test for rollout.
        does our move put the opponent's king in check?

        for rollout: efficiency is important, dont check special moves
        '''

        if move[2] == 'pawn':
            if move[1][1] > 0:
                if self.pieces[move[1][0] - 1, move[1][1] - 1] == 16:
                    return True
            if move[1][1] < 7:
                if self.pieces[move[1][0] - 1, move[1][1] + 1] == 16:
                    return True
                
        if move[2] == 'knight':
            if abs(move[1][0] - self.opp_king[0]) == 1 and abs(move[1][1] - self.opp_king[1]) == 2:
                return True
            if abs(move[1][0] - self.opp_king[0]) == 2 and abs(move[1][1] - self.opp_king[1]) == 1:
                return True
            
        if move[2] == 'bishop' or move[2] == 'queen':
            if self.opp_king[0]-self.opp_king[1] == move[1][0]-move[1][1] or self.opp_king[0]+self.opp_king[1] == move[1][0]+move[1][1]:
                dist_steps = abs(self.opp_king[0] - move[1][0])
                if dist_steps == 1: # directly next to each other
                    # print('bishop')
                    return True
                for dist in range(1, dist_steps): # 1:dist-1
                    if self.pieces[self.opp_king[0] + dist * np.sign(move[1][0] - self.opp_king[0]), self.opp_king[1] + dist * np.sign(move[1][1] - self.opp_king[1])] !=0:
                        return False
                return True
        
        if move[2] == 'rook' or move[2] == 'queen':
            if self.opp_king[0] == move[1][0] or self.opp_king[1] == move[1][1]:
                dist_steps = abs(self.opp_king[0] - move[1][0])
                if dist_steps == 1: # directly next to each other
                    # print('rook')
                    return True
                for dist in range(1, dist_steps): # 1:dist-1
                    if self.pieces[self.opp_king[0] + dist * np.sign(move[1][0] - self.opp_king[0]), self.opp_king[1] + dist * np.sign(move[1][1] - self.opp_king[1])] !=0:
                        return False
                return True
              
        return False

    def checkmate_test(self, move):
    
        # backup game instance
        new_game = self.copy()

        self.PlayMove(move)
        self.FlipBoard()

        moves = self.PossibleMoves()

        # restore game from backup
        self.pieces = new_game.pieces.copy()
        self.turn = new_game.turn
        self.can_castle_white_long = new_game.can_castle_white_long
        self.can_castle_white_short = new_game.can_castle_white_short
        self.can_castle_black_long = new_game.can_castle_black_long
        self.can_castle_black_short = new_game.can_castle_black_short
        self.pawn_to_be_en_passant = new_game.pawn_to_be_en_passant

        if len(moves) == 0:
            return True
        else:
            return False

def board_to_bitmap(board):
    '''
    produce a numpy bitmap from a given board
    '''

    bitmap = np.zeros((12, 8, 8), dtype=bool)

    # loop through each piece type
    for i in range(6):
        # white pieces channel
        bitmap[i, :] = (board == i+1)
        # black pieces channel
        bitmap[i+6, :] = (board == i+11)
    
    return bitmap

def board_to_tensor(board):
    '''
    Transforms a chess board represented as a numpy array into a PyTorch tensor
    with 12 channels, where each channel represents a piece type and the values
    are either 0 or 1, indicating the presence of a piece of that type on a square.
    '''
    # create empty 12-channel tensor
    tensor = torch.zeros((12, 8, 8))
    
    # loop through each piece type
    for i in range(6):
        # white pieces channel
        tensor[i, :] = torch.from_numpy((board == i+1).astype('uint8'))
        # black pieces channel
        tensor[i+6, :] = torch.from_numpy((board == i+11).astype('uint8'))
    
    return tensor

def tensor_to_board(tensor):
    '''
    Transforms a PyTorch tensor with 12 channels representing a chess board into
    a numpy array, where each element is an integer representing a piece type or 0
    for an empty square.
    '''
    # create empty 8x8 numpy array
    board = np.zeros((8, 8), dtype=int)
    
    # loop through each piece type
    for i in range(6):
        # white pieces channel
        board[tensor[i, :] == 1] = i+1
        # black pieces channel
        board[tensor[i+6, :] == 1] = i+11
        
    return board

def index_to_standard(pos):
    '''
    change notation from the index notation that is mainly used here, e.g. (6, 4): (int, int)
    to standard chess notation as used by stockfish, e.g. e2: str

    input needs to be from white's perspective,
    since standard notation is always from white perspective
    '''

    ind_to_file = {0: 'a', 1: 'b', 2: 'c', 3: 'd', 4: 'e', 5: 'f', 6: 'g', 7: 'h'}
    file_out = ind_to_file[pos[1]]
    row_out  = 8 - pos[0]

    return file_out + str(row_out)

def standard_to_index(pos):
    '''
    inverse of index to standard
    '''

    file_to_ind = {'a': 0, 'b': 1, 'c': 2, 'd': 3, 'e': 4, 'f': 5, 'g': 6, 'h':7}
    ind_2 = file_to_ind[pos[0]]
    ind_1  = 8 - int(pos[1])

    return (ind_1, ind_2)

def rotate_board(board, steps):
    # rotate board clock-wise by 90 deg "steps" times
    old_board = board.copy()
    for _ in range(steps):
        new_board = np.zeros((8, 8), dtype='uint8')
        for x in range(8):
            for y in range(8):
                new_board[x, y]  = old_board[7-y, x]
        old_board = new_board.copy()
    return new_board

def mirror_board_tensor(board_tensor):
    # mirror board for value function training
    new_board_tensor = torch.zeros((12, 8, 8))
    for piece in range(12):
        for x in range(8):
            for y in range(8):
                new_board_tensor[piece, x, y] = board_tensor[piece, x, 7-y]
    return new_board_tensor