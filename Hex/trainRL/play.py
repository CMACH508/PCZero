import hex
import numpy as np
import mcts
import random
import argparse
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--saved_model', default=None, type=str)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--game_num', default=50, type=int)
parser.add_argument('--batch_size', default=1024, type=int)
parser.add_argument('--lr', default=0.0001, type=float)
parser.add_argument('--l2', default=0.00001, type=float)
parser.add_argument('--mode', default='train', type=str)
parser.add_argument('--parallel_num', default=10, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--logfile-name', default='aa', type=str)
args = parser.parse_args()

openings = random.sample([i for i in range(hex.SIZE * hex.SIZE)], args.game_num)


class GreedyPlayer:
    def __init__(self, net, board=None):
        self.net = net
        if board is None:
            board = hex.Hex()
        self.board = board

    def get_move(self):
        probs, _ = self.net.run(self.board)
        for i in range(len(probs)):
            if not self.board.is_move_legal(i // hex.SIZE, i % hex.SIZE):
                probs[i] = 0
        move = np.argmax(probs)
        return move // hex.SIZE, move % hex.SIZE

    def make_move(self, x, y):
        self.board = self.board.move(x, y)


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
            board = hex.Hex()
        self.board = board
        self.root = mcts.MCTSNode(board, cpuct=self.cpuct)
        self.result = 0
        self.result_string = None
        self.comments = []
        self.searches_pi = []
        self.qs = []
        first_node = self.root.select_leaf()
        prob, val = self.net.run(first_node.board)
        first_node.incorporate_results(prob, val, first_node)

    def set_size(self, n=hex.SIZE):
        self.size = n
        self.clear()

    def clear(self):
        self.board = hex.Hex()

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
                [leaf.board for leaf in leaves])

            for leaf, move_prob, value in zip(leaves, move_probs, values):
                leaf.revert_virtual_loss(up_to=self.root)
                leaf.incorporate_results(move_prob, value, up_to=self.root)

    def get_move(self):
        if self.board.is_game_over():
            return hex.SIZE, hex.SIZE
        if self.should_resign():
            return hex.SIZE, hex.SIZE

        for i in range(hex.SIZE * hex.SIZE):
            x = i // hex.SIZE
            y = i % hex.SIZE
            if self.root.board.is_move_legal(x, y):
                new_board = self.root.board.move(x, y)
                if new_board.winner == self.root.board.to_play:
                    self.root.child_N[i] = 100000
                    return x, y

        current_readouts = self.root.N
        while self.root.N < current_readouts + self.simulations_per_move:
            self.tree_search()
        for p in range(hex.SIZE):
            for q in range(hex.SIZE):
                if not self.root.board.is_move_legal(p, q):
                    self.root.child_N[p*hex.SIZE+q] = 0
        if self.root.board.step > self.temp_threshold:
            move = np.argmax(self.root.child_N)
        else:
            pdf = self.root.child_N / np.sum(self.root.child_N)
            #path_file_number = glob.glob(pathname='./dataTrain/*.npy')
            #if len(path_file_number) < 100:
            #    tau = 0.8
            #elif len(path_file_number) < 150:
            #    tau = 0.4
            #else:
            tau = 0.2
            pdf = pdf**(1/tau)
            cdf = pdf.cumsum()
            cdf /= cdf[-1]
            selection = random.random()
            move = cdf.searchsorted(selection)
            assert self.root.child_N[move] != 0
        return move // hex.SIZE, move % hex.SIZE

    def make_move(self, x, y):
        self.qs.append(self.root.Q)

        try:
            if x == hex.SIZE and y == hex.SIZE:
                self.root.board = self.root.board.move(x, y)
                self.board = self.root.board
                return True
            self.root = self.root.maybe_add_child(x * hex.SIZE + y)

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

