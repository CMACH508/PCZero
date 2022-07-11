import random
import datetime
import argparse
import torch
import numpy as np
import network
import play
from OthelloGame import OthelloGame
from multiprocessing import Process
import multiprocessing
import glob

SIZE = 8

def single_play(num, simulation, PC=False):
    for i in range(8):
        path_file_number = glob.glob(pathname='./model/*.model')
        if len(path_file_number) > 0:
            modelPath = './model/model' + str(len(path_file_number) - 1) + '.model'
            net = network.PV(modelPath, num=num)
        else:
            net = network.PV(None, num=num)
        r = np.random.exponential(0.1 * SIZE * SIZE)
        player = play.MCTSPlayer(net=net, simulations_per_move=simulation, th=r)
        if random.random() < 0.05:
            player.resign_threshold = -1.0
        player.initialize_game()
        player.root.inject_noise()
        features = []
        pis = []
        outcomes = []
        if PC:
            aveValues = []
            history = []
        while not player.is_game_over():
            move = player.get_move()
            if move <= 64:
                pi = player.root.children_as_pi()
                feature = player.game.toFeature(player.board, player.player)
                features.append(feature)
                pis.append(pi)
            if PC:
                current = player.root
                aver = []
                depth = 0
                while current.is_expanded and depth < 20:
                    depth += 1
                    feature = current.game.toFeature(current.board, current.player)
                    _, value = net.run(feature)
                    aver.append(value)
                    optimal = np.argmax(current.child_N * current.game.getValidMoves(current.board, current.player))
                    if current.child_N[optimal] == 0:
                        legal_moves = current.game.getValidMoves(current.board, current.player)
                        legal_moves = [i for i in range(SIZE * SIZE + 1) if legal_moves[i] == 1]
                        optimal = np.random.choice(legal_moves)
                    current = current.maybe_add_child(optimal)
                aveValue = (np.sum(history)+np.sum(aver))/(len(history)+len(aver))
                aveValues.append(aveValue)
                history.append(aver[0])
                if len(history) > 2:
                    history = history[1:]
            player.make_move(move)
        winner = player.winner()
        results = [winner] * len(features)
        if PC:
            train_data = {
                'features': np.array(features),
                'pis': np.array(pis),
                'results': np.array(results),
                'aveValues': np.array(aveValues)
            }
        else:
            train_data = {
                'features': np.array(features),
                'pis': np.array(pis),
                'results': np.array(results)
            }
        now_time = datetime.datetime.now()
        ts = datetime.datetime.strftime(now_time, '%Y-%m-%d-%H%M%S')
        rd = random.randint(0, 100000000)
        filename = './data/'
        filename += ts + '-' + str(rd) + '.npy'
        np.save(filename, train_data)


def selfplay():
    devices = [0, 1, 2, 3, 4, 5, 6]
    jobs = [Process(target=single_play, args=(devices[i % len(devices)],  200, True))
            for i in range(63)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    selfplay()
