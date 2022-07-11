import gomoku
import numpy as np
import mcts
import random
import argparse
import time

SIZE = gomoku.SIZE

class GreedyPlayer:
    def __init__(self, net, board=None):
        self.net = net
        if board is None:
            board = gomoku.Gomoku()
        self.board = board

    def get_move(self):
        probs, _ = self.net.run(self.board.to_feature())
        for i in range(len(probs)):
            if not self.board.is_move_legal(i // SIZE, i % SIZE):
                probs[i] = 0
        move = np.argmax(probs)
        return move // SIZE, move % SIZE

    def make_move(self, x, y):
        self.board = self.board.play_move(x, y)


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

    def initialize_game(self, board=None):
        if board is None:
            board = gomoku.Gomoku()
        self.board = board
        self.root = mcts.MCTSNode(board, cpuct=self.cpuct)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.qs = []
        first_node = self.root.select_leaf()
        prob, val = self.net.run(first_node.board.to_feature())
        first_node.incorporate_results(prob, val, first_node)

    def set_size(self, n=SIZE):
        self.size = n
        self.clear()

    def clear(self):
        self.board = gomoku.Gomoku()

    def should_resign(self):
        return self.root.Q_perspective < self.resign_threshold

    def tree_search(self):
        leaves = []
        failsafe = 0
        while len(leaves) < self.num_parallel and failsafe < self.num_parallel * 2:
            failsafe += 1
            leaf = self.root.select_leaf()

            if leaf.is_done():
                value = leaf.board.winner
                leaf.backup_value(value, up_to=self.root)
                continue
            leaf.add_virtual_loss(up_to=self.root)
            leaves.append(leaf)
        if leaves:
            move_probs, values = self.net.run_many(
                [leaf.board.to_feature() for leaf in leaves])

            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)

    def get_move(self):
        if self.board.is_game_over():
            return SIZE, SIZE 
        if self.should_resign():
            return SIZE, SIZE 

        current_readouts = self.root.N
        while self.root.N < current_readouts + self.simulations_per_move:
            self.tree_search()
        #time_start = time.time()
        #while time.time()-time_start <= 10:
        #    self.tree_search()
        for p in range(SIZE):
            for q in range(SIZE):
                if not self.root.board.is_move_legal(p, q):
                    self.root.child_N[p * SIZE + q] = 0
        if self.root.board.step > self.temp_threshold:
            move = np.argmax(self.root.child_N)
        else:
            pdf = self.root.child_N / np.sum(self.root.child_N)
            pdf = pdf**0.8
            cdf = pdf.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            move = cdf.searchsorted(selection)
            assert self.root.child_N[move] != 0
        return move // SIZE, move % SIZE

    def make_move(self, x, y):
        self.qs.append(self.root.Q)

        try:
            if x == SIZE and y == SIZE:
                self.root.board = self.root.board.play_move(x, y)
                self.board = self.root.board
                return True
            self.root = self.root.maybe_add_child(x * SIZE + y)

        except:
            print("Illegal move")
            self.qs.pop()
            return False
        self.board = self.root.board
        del self.root.parent.children
        return True

    def is_game_over(self):
        over = self.board.is_game_over()
        return over, self.board.winner

    def show_board(self):
        print(self.board)

