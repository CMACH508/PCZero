import sgf
import os
import hex
import random
import numpy as np
SIZE = hex.SIZE


def actionToCoord(action):
    columnList = 'abcdefghijklmnopqrstuvwxyz'
    column = columnList.index(action[0])
    raw = int(action[1:]) - 1
    return raw, column


def augment(board, pi):
    rd = random.random()
    if rd < 0.5:
        board_save = np.rot90(board, k=2, axes=(1, 2))
        pi_save = np.rot90(pi, k=2)
        pi_save = pi_save.reshape(hex.SIZE * hex.SIZE)
        return board_save, pi_save
    return board, pi.reshape(hex.SIZE * hex.SIZE)


def oneHot(raw, column):
    actionOneHot = np.zeros(shape=(SIZE, SIZE))
    actionOneHot[raw, column] = 1
    return actionOneHot


def process(path):
    print(path)
    fr = open(path)
    collection = sgf.parse(fr.read())
    for child in collection.children:
        nodes = child.nodes
        features = []
        pis = []
        results = []
        if nodes[0].properties['RE'][0] == 'W+':
            result = -1
        elif nodes[0].properties['RE'][0] == 'B+':
            result = +1
        else:
            continue
        board = hex.Hex()
        for player, action in nodes[1].properties.items():
            raw, column = actionToCoord(action[0])
            board = board.move(raw, column)
        for node in nodes[2:-1]:
            for player, action in node.properties.items():
                raw, column = actionToCoord(action[0])
                pi = oneHot(raw, column)
                feature = board.to_feature()
                #feature, pi = augment(feature, pi)
                features.append(feature)
                pis.append(pi)
                results.append(result)
                board = board.move(raw, column)
    return features, pis, results 


def walk(dirNames, savedir):
    os.mkdir(savedir)
    index = 0
    for dirName in dirNames:
        for root, dirs, files in os.walk(dirName):
            for f in files:
                if f == 'results':
                    continue
                path = os.path.join(root, f)
                features, pis, results = process(path)
                data = {
                    'features': np.array(features),
                    'pis': np.array(pis),
                    'results': np.array(results),
                }
                filename = savedir + '/Hex_' + str(SIZE) + 'x' + str(SIZE) + '_' + str(index) + '.npy'
                index += 1
                np.save(filename, data)

