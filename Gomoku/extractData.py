import os
import gomoku
import random
import numpy as np
import datetime
import glob

SIZE = gomoku.SIZE
columns = 'abcdefghijklmnopqrstuvwxyz'

def oneHot(action):
    actionOneHot = np.zeros(SIZE * SIZE)
    actionOneHot[action[0] * SIZE + action[1]] = 1
    return actionOneHot


def process(moves, winner):
    #print(moves)
    if len(moves) < 16:
        return
    features = []
    pis = []
    results = []
    board = gomoku.Gomoku()
    for i in range(6):
        x = columns.index(moves[i][0])
        y = int(moves[i][1:]) - 1
        board = board.play_move(x, y)
    for i in range(6, len(moves)):
        x = columns.index(moves[i][0])
        y = int(moves[i][1:]) - 1
        if not board.is_move_legal(x, y):
            return
        features.append(board.to_feature())
        pis.append(oneHot((x, y)))
        board = board.play_move(x, y)
    results = [winner] * len(pis)
    data = {
        'features': np.array(features),
        'pis': np.array(pis),
        'results': np.array(results)
    }
    now_time = datetime.datetime.now()
    ts = datetime.datetime.strftime(now_time, '%Y-%m-%d-%H%M%S')
    rd = random.randint(0, 100000000)
    filename = '/cmach-data/zhaodengwei/Gomoku/train/'
    filename += ts + '-' + str(rd) + '.npy'
    np.save(filename, data)


def walk():
    fr = open('renjunet_v10_20210728.rif', 'r', encoding="unicode_escape")
    for i in range(6129):
        line = fr.readline()
    while line != '</games>\n':
        if line[:5] == '<game':
            line = line.strip().split()
            for item in line:
                if item[:7] == 'bresult':
                    winner = item[9:-1]
            if winner == '1':
                winner = 1
            elif winner == '0':
                winner = -1
            elif winner == '0.5':
                winner = 0
            else:
                print(line)
        if line[:5] == '<move':
            moves = line[6:-8].split()
            process(moves, winner)
        line = fr.readline()

walk()

#process('./9x9/2020/07/31/1050872.sgf')

