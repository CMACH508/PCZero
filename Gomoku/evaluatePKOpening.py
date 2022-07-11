import gomoku
import numpy as np
import play
import random
import multiprocessing
import network
import torch
from multiprocessing import Process
from play import MCTSPlayer, GreedyPlayer

SIZE = gomoku.SIZE
openings = np.load('openings.npy')


def single_play(net1, net2, start, end, simulation, winning, flag):
    for i in range(start, end):
        player1 = MCTSPlayer(net=net1, simulations_per_move=simulation, th=0, resign_threshold=-1.0)
        player2 = MCTSPlayer(net=net2, simulations_per_move=simulation, th=0, resign_threshold=-1.0)
        board = gomoku.Gomoku(board=openings[i], step=8)
        player1.initialize_game(board=board)
        player2.initialize_game(board=board)
        #player1 = GreedyPlayer(net=net1, board=board)
        #player2 = GreedyPlayer(net=net2, board=board)
        color = board.to_play

        while not player1.board.is_game_over():
            if color == flag:
                x, y = player1.get_move()
            else:
                x, y = player2.get_move()
            color = -color
            player1.make_move(x, y)
            player2.make_move(x, y)

        winner = player1.board.winner
        if winner == flag:
            winning.append(1)


if __name__ == '__main__':
    game_num = 200
    multiprocessing.set_start_method('spawn')
    models1 =['./model/model_NonPC_10.model', './model/model_NonPC_10.model', './model/model_NonPC_10.model', './model/model_NonPC_10.model']
    models2 = ['./model/model_PC_2.0_Feature_1.0_10.model', './model/model_PC_1.0_Feature_1.0_10.model', './model/model_Feature_1.0_10.model', './model/model_Feature_0.5_10.model']
    for index in range(len(models1)):
        parallel = 10
        interval = 20
        pc1name = models1[index]
        pc2name = models2[index]
        nets1 = []
        nets2 = []
        for i in range(parallel):
            nets1.append(network.PV(model_path=pc1name, channel=256, numBlock=5, num=i%torch.cuda.device_count()))
            nets2.append(network.PV(model_path=pc2name, channel=256, numBlock=5, num=i%torch.cuda.device_count()))
        l = []
        old = 0
        whitewins = 0
        blackwins = 0

        flag = 1

        with multiprocessing.Manager() as mg:
            winning_list = multiprocessing.Manager().list([])
            jobs = [Process(target=single_play,
                    args=(nets1[i], nets2[i], interval*i, interval*(i+1), 800, winning_list, flag)) for i in range(parallel)]
            #jobs.append(Process(target=single_play, args=(nets1[i], nets2[i], interval*i, go.N * go.N, 800, winning_list, flag)))
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()

            whitewins = game_num - len(winning_list)
            flag = -flag
            jobs = [Process(target=single_play,
                    args=(nets1[i], nets2[i], interval*i, interval*(i+1), 800, winning_list, flag)) for i in range(parallel)]
            #jobs.append(Process(target=single_play, args=(nets1[i], nets2[i], interval*i, go.N * go.N, 800, winning_list, flag)))
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()

            old = len(winning_list)
            blackwins = 2 * game_num - old - whitewins
            print("old wins", len(winning_list))

        old_name = pc1name
        line = pc1name + ' vs ' + pc2name + ' = ' + str(old) + ':' + str(2 * game_num - old) + '\t' + str(whitewins) + '\t' + str(blackwins)
        f = open('tournamentPK.txt', 'a')
        f.write(line + '\n')
        f.close()
