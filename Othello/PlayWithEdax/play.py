from OthelloGame import OthelloGame
import numpy as np
import mcts
import random
import argparse
import glob

SIZE = 8

class GreedyPlayer:
    def __init__(self, net, board=None, player=1):
        self.net = net
        self.game = OthelloGame(8)
        if board is None:
            self.board = self.game.getInitBoard()
            self.player = 1
        else:
            self.board = np.array(board)
            self.player = player

    def get_move(self):
        probs, _ = self.net.run(self.game.toFeature(self.board, self.player))
        legal_moves = self.game.getValidMoves(self.board, self.player)
        for i in range(len(legal_moves)):
            if legal_moves[i] == 0:
                probs[i] = 0
        move = np.argmax(probs)
        return move

    def make_move(self, move):
        self.board, self.player, _, _ = self.game.getNextState(self.board, self.player, move)


class RandomPlayer:
    def __init__(self, board=None, player=1):
        self.game = OthelloGame(8)
        if board is None:
            self.board = self.game.getInitBoard()
            self.player = 1
        else:
            self.board = np.array(board)
            self.player = player

    def get_move(self):
        legal_moves = self.game.getValidMoves(self.board, self.player)
        legal_idx = [i for i in range(len(legal_moves)) if legal_moves[i] == 1]
        return random.choice(legal_idx)

    def make_move(self, move):
        self.board, self.player, _, _ = self.game.getNextState(self.board, self.player, move)


class MCTSPlayer:
    def __init__(self, net, simulations_per_move=800, resign_threshold=-1.0, num_parallel=2, th=10, cpuct=1.5):
        self.net = net
        self.simulations_per_move = simulations_per_move
        self.temp_threshold = th
        self.num_parallel = num_parallel
        self.cpuct = cpuct
        self.qs = []
        self.comments = []
        self.searches_pi = []
        self.root = None
        self.result = 0
        self.result_string = None
        self.resign_threshold = -abs(resign_threshold)
        super().__init__()

    def initialize_game(self, board=None, player=1, step=1):
        self.game = OthelloGame(8)
        if board is None:
            self.board = self.game.getInitBoard()
        else:
            self.board = board
        self.player = player
        self.step = step
        self.root = mcts.MCTSNode(self.board, self.game, self.player, cpuct=self.cpuct)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.qs = []
        first_node = self.root.select_leaf()
        prob, val = self.net.run(self.game.toFeature(self.board, self.player))
        first_node.incorporate_results(prob, val, first_node)

    def set_size(self, n=SIZE):
        self.size = n
        self.clear()

    def clear(self):
        self.game = OthelloGame(8)
        self.board = self.game.getInitBoard()
        self.player = 1

    def should_resign(self):
        return self.root.Q_perspective < self.resign_threshold

    def tree_search(self):
        leaves = []
        failsafe = 0
        while len(leaves) < self.num_parallel and failsafe < self.num_parallel * 2:
            failsafe += 1
            leaf = self.root.select_leaf()

            if leaf.game.getGameEnded(leaf.board):
                value = leaf.game.getWinner(leaf.board)
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.net.run_many(
                [leaf.game.toFeature(leaf.board, leaf.player) for leaf in leaves])

            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)

    def get_move(self):
        if self.game.getGameEnded(self.board) or self.should_resign():
            return 65

        current_readouts = self.root.N
        while self.root.N < current_readouts + self.simulations_per_move:
            self.tree_search()
        legal_moves = self.game.getValidMoves(self.board, self.player)
        for i in range(len(legal_moves)):
            if legal_moves[i] == 0:
                self.root.child_N[i] = 0
        if self.step > self.temp_threshold:
            move = np.argmax(self.root.child_N)
        else:
            pdf = self.root.child_N / np.sum(self.root.child_N)
            path_file_number = glob.glob(pathname='./dataTrain/*.npy')
            if len(path_file_number) < 200:
                tau = 0.8
            elif len(path_file_number) < 400:
                tau = 0.4
            else:
                tau = 0.2
            pdf = pdf**(1/tau)
            cdf = pdf.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            move = cdf.searchsorted(selection)
            assert self.root.child_N[move] != 0
        return move

    def make_move(self, move):
        self.qs.append(self.root.Q)
        self.root = self.root.maybe_add_child(move)
        self.board = self.root.board
        self.player = self.root.player
        self.step += 1
        del self.root.parent.children
        return True

    def is_game_over(self):
        over = self.game.getGameEnded(self.board)
        return over

    def winner(self):
        return self.game.getWinner(self.board)

    def show_board(self):
        self.game.display(self.board)

