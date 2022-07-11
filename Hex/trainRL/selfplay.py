import random
import datetime
import argparse
import torch
import numpy as np
import network
import play
import hex
from multiprocessing import Process
import multiprocessing
import glob

parser = argparse.ArgumentParser()
parser.add_argument('--saved_model', default=None, type=str)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--game_num', default=1008, type=int)
parser.add_argument('--batch_size', default=128, type=int)
parser.add_argument('--lr', default=0.01, type=float)
parser.add_argument('--parallel_num', default=45, type=int)
args = parser.parse_args()

SIZE = hex.SIZE


def augment(board, pi):
    rd = random.random()
    if rd < 0.5:
        board_save = np.rot90(board, k=2, axes=(1, 2))
        pi_save = np.reshape(pi, (hex.SIZE, hex.SIZE))
        pi_save = np.rot90(pi_save, k=2)
        pi_save = pi_save.reshape(hex.SIZE * hex.SIZE)
        return board_save, pi_save

    return board, pi


def single_play(num, simulation, PC=False):
    for i in range(23):
        path_file_number = glob.glob(pathname='./model/*.model')
        if len(path_file_number) > 0:
            modelPath = './model/model' + str(len(path_file_number) - 1) + '.model'
            net = network.PV(modelPath, num=num)
        else:
            net = network.PV(None, num=num)
        r = np.random.exponential(0.04*hex.SIZE*hex.SIZE)
        player = play.MCTSPlayer(net=net, simulations_per_move=simulation, th=r)
        if random.random() < 0.05:
            player.resign_threshold = -1.0
        player.initialize_game()
        rd = random.randint(0, hex.SIZE ** 2 - 1)
        player.make_move(rd // hex.SIZE, rd % hex.SIZE)
        player.root.inject_noise()
        features = []
        pis = []
        outcomes = []
        if PC:
            aveValues = []
            _, value = net.run(player.root.board)
            history = [value]
        while not player.board.is_game_over():
            x, y = player.get_move()
            if x != SIZE or y != SIZE:
                pi = player.root.children_as_pi()
                feature, pi_save = augment(player.board.to_feature(), pi)
                features.append(feature)
                pis.append(pi_save)
            if PC:
                current = player.root
                aver = []
                depth = 0
                while current.is_expanded and depth <= 5:
                    depth += 1
                    _, value = net.run(current.board)
                    aver.append(value)
                    optimal = np.argmax(current.child_N * current.board.all_legal_moves())
                    if current.child_N[optimal] == 0:
                        legal_moves = current.board.all_legal_moves()
                        legal_moves = [i for i in range(hex.SIZE*hex.SIZE) if legal_moves[i] == 1]
                        optimal = np.random.choice(legal_moves)
                    current = current.maybe_add_child(optimal)
                aveValues.append((np.sum(history)+np.sum(aver))/(len(history)+depth))
                history.append(aver[0])
                if len(history) > 5:
                    history = history[1:]
            player.make_move(x, y)
        winner = player.board.winner
        for _ in range(len(features)):
            outcomes.append(winner)
        if PC:
            train_data = {
                'features': np.array(features),
                'pis': np.array(pis),
                'results': np.array(outcomes),
                'aveValues': np.array(aveValues)
            }
        else:
            train_data = {
                'features': np.array(features),
                'pis': np.array(pis),
                'results': np.array(outcomes)
            }
        now_time = datetime.datetime.now()
        ts = datetime.datetime.strftime(now_time, '%Y-%m-%d-%H%M%S')
        rd = random.randint(0, 100000000)
        filename = './data/'
        filename += ts + '-' + str(rd) + '.npy'
        np.save(filename, train_data)


def selfplay():
    jobs = [Process(target=single_play, args=(i % torch.cuda.device_count(),  400, False))
            for i in range(args.parallel_num)]
    for j in jobs:
        j.start()
    for j in jobs:
        j.join()


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    selfplay()
