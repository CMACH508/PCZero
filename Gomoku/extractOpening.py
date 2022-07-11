import os
import gomoku
import random
import numpy as np
import datetime
import glob

SIZE = gomoku.SIZE
columns = 'abcdefghijklmnopqrstuvwxyz'


def process(moves):
    board = np.zeros((15, 15))
    color = 1
    for i in range(8):
        x = columns.index(moves[i][0])
        y = int(moves[i][1:]) - 1
        board[x, y] = color
        color = -color
    return board

def walk():
    fr = open('renjunet_v10_20210728.rif', 'r', encoding="unicode_escape")
    for i in range(6129):
        line = fr.readline()
    movesList = []
    while line != '</games>\n':
        if line[:5] == '<move':
            moves = line[6:-8].split()
            if len(moves) >= 16:
                movesList.append(moves)
        line = fr.readline()
    index = np.array([i for i in range(len(movesList))])
    np.random.shuffle(index)
    openings = []
    for i in index[:200]:
        openings.append(process(movesList[i]))
    np.save('openings.npy', np.array(openings))

walk()

#process('./9x9/2020/07/31/1050872.sgf')

