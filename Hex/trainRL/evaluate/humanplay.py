import hex
import networkMCTS
import numpy as np
import mcts
import os
import torch


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

        if self.root.board.step > self.temp_threshold:
            move = np.argmax(self.root.child_N)
        else:
            cdf = self.root.child_N.cumsum()
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


modelName = 'PC3-900.model'
columns = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
net = networkMCTS.PV(model_path=modelName, num1=3, num2=0)
player = MCTSPlayer(net=net, simulations_per_move=200, th=0, resign_threshold=-1.0)
player.initialize_game()
playerIndex = 1
over = False
player.show_board()
while not over:
    if playerIndex == 1:
        posX = int(input("Please Input the Raw: "))
        posY = input("Please Input the Column: ")
        player.make_move(posX-1, columns.index(posY))
    else:
        x, y = player.get_move()
        player.make_move(x,y)
    player.show_board()
    playerIndex = - playerIndex
    over, winner = player.is_game_over()
    
