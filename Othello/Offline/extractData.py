import os
import random
import datetime
import glob
import shutil
import numpy as np
from WTB_reader import WTB_reader
from OthelloGame import OthelloGame

def WTBLoader():
    for year in range(1977, 2021):
        wtbName = './wtbData/WTH_' + str(year) + '.wtb'
        reader = WTB_reader(name='othella', path=wtbName)
        reader.export_games('./games')

def preprocessing(game):
    print(game)
    fr = open(game, 'r')
    black_player = fr.readline().strip().split()[2]
    white_player = fr.readline().strip().split()[2]
    black_score = int(fr.readline().strip().split()[2])
    theoretical_score = int(fr.readline().strip().split()[2])
    if black_score > 32:
        winner = 1
    elif black_score < 32:
        winner = -1
    else:
        winner = 0
    game = OthelloGame(8)
    board = game.getInitBoard()
    player = 1
    features = []
    pis = []
    line = fr.readline().strip()
    if line == '-1--1':
        return
    x, y = line.split('-')
    action = int(x) * 8 + int(y)
    while True:
        board, player, feature, pi = game.getNextState(board, player, action)
        features.append(feature)
        pis.append(pi)
        if game.getGameEnded(board):
            break
        if game.getValidMoves(board, player)[-1] == 1:
            action = 64
        else:
            line = fr.readline().strip()
            if line == '-1--1':
                return
            x, y = line.split('-')
            action = int(x) * 8 + int(y)
    boardWinner = game.getWinner(board)
    if boardWinner != winner:
        print('Wrong Winner')
    results = [(winner + 1) / 2.0] * len(pis)
    data = {
        'features': np.array(features),
        'pis': np.array(pis),
        'results': np.array(results)
    }
    now_time = datetime.datetime.now()
    ts = datetime.datetime.strftime(now_time, '%Y-%m-%d-%H%M%S')
    rd = random.randint(0, 100000000)
    filename = './data/'
    filename += ts + '-' + str(rd) + '.npy'
    np.save(filename, data)
'''
games = glob.glob('./games/*.txt')
for game in games:
    preprocessing(game)
files = glob.glob('./data/*.npy')
index = np.array([idx for idx in range(len(files))])
np.random.shuffle(index)
for idx in index[100000:]:
    file = files[idx]
    desFile = './test/' + file.split('/')[-1]
    shutil.move(file, desFile)
'''


