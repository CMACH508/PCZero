from __future__ import print_function
import sys
from GameTemplate import Game
from OthelloLogic import Board
import numpy as np

class OthelloGame(Game):
    square_content = {
        +1: "X",
        +0: "-",
        -1: "O"
    }

    @staticmethod
    def getSquarePiece(piece):
        return OthelloGame.square_content[piece]

    def __init__(self, n):
        self.n = n

    def getInitBoard(self):
        b = Board(self.n)
        return np.array(b.pieces)

    def getBoardSize(self):
        return (self.n, self.n)

    def getActionSize(self):
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        b = Board(self.n)
        b.pieces = np.copy(board)
        feature = self.toFeature(board, player)
        action_vec = np.zeros(self.n * self.n + 1)
        action_vec[action] = 1
        legal_move = self.getValidMoves(board, player)
        if legal_move[action] == 0:
            print(player, legal_move)
            print('Illegal Move ' + str(action))
            self.display(board)
        move = (int(action/self.n), action%self.n)
        if action != self.n ** 2:
            b.execute_move(move, player)
        return (b.pieces, -player, feature, action_vec)

    def getValidMoves(self, board, player):
        valids = [0]*self.getActionSize()
        b = Board(self.n)
        b.pieces = np.copy(board)
        legalMoves =  b.get_legal_moves(player)
        if len(legalMoves) == 0:
            valids[-1] = 1
            return np.array(valids)
        for x, y in legalMoves:
            valids[self.n * x + y] = 1
        return np.array(valids)

    def getGameEnded(self, board):
        b = Board(self.n)
        b.pieces = np.copy(board)
        if b.has_legal_moves(1):
            return 0
        if b.has_legal_moves(-1):
            return 0
        return 1

    def getWinner(self, board):
        b = Board(self.n)
        b.pieces = np.copy(board)
        diff = b.countDiff(1)
        if diff > 0:
            return 1
        elif diff < 0:
            return -1
        else:
            return 0

    def getCanonicalForm(self, board, player):
        return player*board

    def getSymmetries(self, board, pi):
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def toFeature(self, board, player):
        feature = np.zeros([5, 8, 8])
        for y in range(self.n):
            for x in range(self.n):
                if board[x][y] == 1:
                    feature[0][x][y] = 1
                elif board[x][y] == -1:
                    feature[1][x][y] = 1
                else:
                    feature[2][x][y] = 1
        if player == 1:
            feature[3, :, :] = 1
        else:
            feature[4, :, :] = 1 
        return feature

    def stringRepresentation(self, board):
        return board.tostring()

    def stringRepresentationReadable(self, board):
        board_s = "".join(self.square_content[square] for row in board for square in row)
        return board_s

    def getScore(self, board, player):
        b = Board(self.n)
        b.pieces = np.copy(board)
        return b.countDiff(player)

    @staticmethod
    def display(board):
        n = board.shape[0]
        print("   ", end="")
        for y in range(n):
            print(y, end=" ")
        print("")
        print("-----------------------")
        for y in range(n):
            print(y, "|", end="")    # print the row #
            for x in range(n):
                piece = board[y][x]    # get the piece to print
                print(OthelloGame.square_content[piece], end=" ")
            print("|")

        print("-----------------------")
