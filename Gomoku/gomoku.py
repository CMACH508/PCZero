import numpy as np
import copy
from collections import deque

SIZE = 15
BLACK = 1
WHITE = -1
EMPTY = 0

class IllegalMove(Exception):
    pass

class Gomoku:
    def __init__(self, board=np.zeros(shape=(SIZE, SIZE), dtype=np.int8), winner=EMPTY, to_play=BLACK, step=0):
        self.board = board.copy()
        self.winner = winner
        self.to_play = to_play
        self.step = step

    def is_move_legal(self, x, y):
        return self.board[x][y] == EMPTY


    def connect(self, move):
        for direct in [(0, 1), (1, 0), (1, 1), (1, -1)]:
            found = 1
            for d in [-1, 1]:
                for i in range(1, 5):
                    nextMove = (move[0] + direct[0] * i * d, move[1] + direct[1] * i * d)
                    if (nextMove[0] < 0 or nextMove[0] >= SIZE
                        or nextMove[1] < 0 or nextMove[1] >= SIZE
                        or self.board[nextMove] != self.to_play):
                        break
                    else:
                        found += 1
            if found == 5:
                return True
        return False

    def play_move(self, x, y, color=None):
        pos = copy.deepcopy(self)
        if color == None:
            color = self.to_play
        if not pos.is_move_legal(x, y):
            raise IllegalMove("{} move at {} is illegal: \n{}".format(
                "Black" if self.to_play == BLACK else "White", (x, y), self))
        pos.board[x][y] = color
        if self.connect((x,y)):
            pos.winner = color
        pos.to_play = -pos.to_play
        pos.step = self.step + 1
        if pos.step == SIZE * SIZE:
            pos.winner = 0
        return pos

    def is_game_over(self):
        over = (self.step == SIZE * SIZE or self.winner != EMPTY)
        return over

    def result(self):
        return self.winner

    def __deepcopy__(self, memodict={}):
        new_board = np.copy(self.board)
        return Gomoku(new_board, self.winner, self.to_play, self.step)

    def all_legal_moves(self):
        ret = []
        for i in range(SIZE):
            for j in range(SIZE):
                if self.board[i][j] == EMPTY:
                    ret.append(1)
                else:
                    ret.append(0)
        return np.array(ret)

    def to_feature(self):
        ret = np.zeros(shape=(5, SIZE, SIZE))
        ret[0, :, :] = (self.board == 1)
        ret[1, :, :] = (self.board == -1)
        ret[2, :, :] = (self.board == 0)
        if self.to_play == 1:
            ret[3, :, :] = np.ones([SIZE, SIZE])
        else:
            ret[4, :, :] = np.ones([SIZE, SIZE])
        return ret

    def __repr__(self):
        retstr = '\n' + ' '
        for i in range(SIZE):
            retstr += chr(ord('a') + i) + ' '
        retstr += '\n'
        for i in range(0, SIZE):
            if i <= 8:
                retstr += ' ' * i + str(i + 1) + ' '
            else:
                retstr += ' ' * (i - 1) + str(i + 1) + ' '
            for j in range(0, SIZE):
                if self.board[i, j] == BLACK:
                    retstr += 'X'
                elif self.board[i, j] == WHITE:
                    retstr += 'O'
                else:
                    retstr += '.'
                retstr += ' '
            retstr += '\n'
        return retstr

    def show(self):
        for i in range(self.board.shape[0]):
            for j in range(self.board.shape[1]):
                if self.board[i][j] == 1:
                    print(' B ', end="")
                elif self.board[i][j] == -1:
                    print(' W ', end="" )
                else:
                    print(' O ', end="")
            print('\n')

