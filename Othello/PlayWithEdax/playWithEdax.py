import os
import torch
import numpy as np
import network
import play
from OthelloGame import OthelloGame
from subprocess import Popen, PIPE
from multiprocessing import Process
import multiprocessing

columns = 'ABCDEFGH'

def load(name):
    games = np.zeros([500, 10, 10])
    fr = open(name, 'r')
    row = 0
    for line in fr.readlines():
        x = row // 10
        y = row % 10
        line = [float(pos) for pos in line.strip().split()]
        for column in range(len(line)):
            games[column][x][y] = line[column]
        row += 1
    return np.array(games)[:, 1:-1, 1:-1]

def initialBoard(board):
    board = board.reshape(-1)
    ans = ''
    for piece in board:
        if piece == 0:
            ans += '-'
        elif piece == 1:
            ans += '*'
        else:
            ans += 'O'
    return ans

class edaxPlayer:
    def __init__(self, command, board=None):
        self._command = command
        p = Popen(self._command, shell = True, stdin=PIPE, stdout=PIPE, stderr=PIPE, close_fds=True, universal_newlines=True, preexec_fn=os.setsid)
        self._pid = p.pid
        self._stdin, self._stdout, self._stderr = (p.stdin, p.stdout, p.stderr)
        for _ in range(12):
            line = self._stdout.readline()
        self.sendCommand('set verbose = 0\n')
        self.sendCommand('set l = 3\n')
        if board is not None:
            command = 'setboard ' + board + '\n'
            self.sendCommand(command)

    def sendCommand(self, command):
        self._stdin.write(command)
        self._stdin.flush()
        line = self._stdout.readline()

    def get_move(self):
        command = 'go\n'
        self.sendCommand(command)
        for _ in range(2):
            line = self._stdout.readline()
        action = line.strip().split()[-1]
        y = columns.index(action[0])
        x = int(action[1]) - 1
        return x * 8 + y

    def make_move(self, move):
        x = move // 8 + 1
        y = move % 8
        if move == 64:
            move = 'PS'
        else:
            move = columns[y] + str(x)
        command = 'play ' + move + '\n'
        self.sendCommand(command)

    def save_game(self, name):
        command = 's ' + name +'\n'
        self.sendCommand(command)

    def quit(self):
        command = 'quit\n'
        self.sendCommand(command)


def playGame(net, flag, board, player, sims=800, MCTS=False):
    if MCTS:
        testPlayer = play.MCTSPlayer(net=net, simulations_per_move=sims, th=0)
        testPlayer.initialize_game(board=board, player=player)
    else:
        testPlayer = play.GreedyPlayer(net, board=board, player=player)
    edax = edaxPlayer('./lEdax-x64', board=initialBoard(board))
    while not testPlayer.game.getGameEnded(testPlayer.board):
        if testPlayer.player == flag:
            move = testPlayer.get_move()
        else:
            legal_moves = testPlayer.game.getValidMoves(testPlayer.board, testPlayer.player)
            if legal_moves[-1] == 1:
                move = 64
            else:
                move = edax.get_move()
        testPlayer.make_move(move)
        edax.make_move(move)
    winner = testPlayer.game.getWinner(testPlayer.board)
    #edax.save_game('game.txt')
    edax.quit()
    return winner

def tournaments(net, flag, player, winning, boards, sims=800, MCTS=False):
    for board in boards:
        winner = playGame(net, flag, board, player, sims, MCTS)
        winning.append((winner + 1) / 2)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parallel = 10
    interval = 5
    black_board = load('black_eval.txt')
    white_board = load('white_eval.txt')
    models = ['./modelMCTS/model399.model']
    for sims in [800]:
        for model in models:
            net = []
            for i in range(parallel):
                net.append(network.PV(model, num=i%torch.cuda.device_count()))
            with multiprocessing.Manager() as mg:
                winning = multiprocessing.Manager().list([])
                jobs = [Process(target=tournaments, args=(net[i], 1, 1, winning, black_board[i * interval : (i + 1) * interval], sims, True)) for i in range(parallel)]
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                black_black_score = np.sum(winning)
                print(black_black_score)
                winning = multiprocessing.Manager().list([])
                jobs = [Process(target=tournaments, args=(net[i], -1, 1, winning, black_board[i * interval : (i + 1) * interval], sims, True)) for i in range(parallel)]
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                black_white_score = parallel * interval - np.sum(winning)
                print(black_white_score)
                winning = multiprocessing.Manager().list([])
                jobs = [Process(target=tournaments, args=(net[i], 1, -1, winning, white_board[i * interval : (i + 1) * interval], sims, True)) for i in range(parallel)]
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                white_black_score = np.sum(winning)
                winning = multiprocessing.Manager().list([])
                print(white_black_score)
                jobs = [Process(target=tournaments, args=(net[i], -1, -1, winning, white_board[i * interval : (i + 1) * interval], sims, True)) for i in range(parallel)]
                for j in jobs:
                    j.start()
                for j in jobs:
                    j.join()
                white_white_score = parallel * interval - np.sum(winning)
            fr = open('tournamentPK.txt', 'a')
            fr.write(model + '\t' + str(black_black_score) + '\t' + str(black_white_score) + '\t' + str(white_black_score) + '\t' + str(white_white_score) + '\t' + str(black_black_score + black_white_score + white_black_score + white_white_score) + '\n')
            fr.close()