import torch
import numpy as np
import network
import play
from OthelloGame import OthelloGame
from multiprocessing import Process
import multiprocessing

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

def tournament(black_player, white_player, winning):
    while not black_player.game.getGameEnded(black_player.board):
        if black_player.player == 1:
            move = black_player.get_move()
        else:
            move = white_player.get_move()
        black_player.make_move(move)
        white_player.make_move(move)
    winning.append((black_player.game.getWinner(black_player.board) + 1) / 2)

def tournaments(net1, net2, player, first_player, winning, boards, sims, MCTS=False):
    for board in boards:
        if player == 1:
            if MCTS:
                black_player = play.MCTSPlayer(net=net1, simulations_per_move=sims, th=0)
                black_player.initialize_game(board=board, player=first_player)
            else:
                black_player = play.GreedyPlayer(net1, board=board, player=first_player)
            if net2 == None:
                white_player = play.RandomPlayer(board=board, player=first_player)
            elif MCTS:
                white_player = play.MCTSPlayer(net=net2, simulations_per_move=sims, th=0)
                white_player.initialize_game(board=board, player=first_player)
            else:
                white_player = play.GreedyPlayer(net2, board=board, player=first_player)
        else:
            if MCTS:
                white_player = play.MCTSPlayer(net=net1, simulations_per_move=sims, th=0)
                white_player.initialize_game(board=board, player=first_player)
            else:
                white_player = play.GreedyPlayer(net1, board=board, player=first_player)
            if net2 == None:
                black_player = play.RandomPlayer(board=board, player=first_player)
            elif MCTS:
                black_player = play.MCTSPlayer(net=net2, simulations_per_move=sims, th=0)
                black_player.initialize_game(board=board, player=first_player)
            else:
                black_player = play.GreedyPlayer(net2, board=board, player=first_player)
        tournament(black_player, white_player, winning)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    parallel = 10
    black_board = load('black_eval.txt')
    white_board = load('white_eval.txt')
    interval = 5
    model1 = './model/model_NonPC_10.model'
    model2 = './model/model_PC_1.0_10.model'
    net1 = []
    net2 = []
    for i in range(parallel):
        net1.append(network.PV(model1, num=i%torch.cuda.device_count()))
        net2.append(network.PV(model2, num=i%torch.cuda.device_count()))
    MCTS = True
    for sims in [50, 100, 200, 400, 1200, 1600]:
        with multiprocessing.Manager() as mg:
            winning = multiprocessing.Manager().list([])
            jobs = [Process(target=tournaments, args=(net1[i], net2[i], 1, 1, winning, black_board[i * interval : (i + 1) * interval], sims, MCTS)) for i in range(parallel)]
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            black_black_score = np.sum(winning)
            print(black_black_score)
            winning = multiprocessing.Manager().list([])
            jobs = [Process(target=tournaments, args=(net1[i], net2[i], -1, 1, winning, black_board[i * interval : (i + 1) * interval], sims, MCTS)) for i in range(parallel)]
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            black_white_score = parallel * interval - np.sum(winning)
            print(black_white_score)
            winning = multiprocessing.Manager().list([])
            jobs = [Process(target=tournaments, args=(net1[i], net2[i], 1, -1, winning, white_board[i * interval : (i + 1) * interval], sims, MCTS)) for i in range(parallel)]
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            white_black_score = np.sum(winning)
            print(white_black_score)
            winning = multiprocessing.Manager().list([])
            jobs = [Process(target=tournaments, args=(net1[i], net2[i], -1, -1, winning, white_board[i * interval : (i + 1) * interval], sims, MCTS)) for i in range(parallel)]
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
            white_white_score = parallel * interval - np.sum(winning)
        fr = open('tournamentPK.txt', 'a')
        fr.write(model1 + '\t' + model2 + '\t' + str(sims) + '\t' + str(black_black_score) + '\t' + str(black_white_score) + str(white_black_score) + '\t' + str(white_white_score) + '\t' + str(black_black_score + black_white_score + white_black_score + white_white_score) + '\n')
        fr.close()