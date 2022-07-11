import hex
import networkMCTS
import random
import subprocess
from multiprocessing import Process
import multiprocessing
import time
import argparse
from play import MCTSPlayer, GreedyPlayer
import os
from random import randrange
from program import Program
from gamestate import gamestate
import threading
import sys
import torch


parser = argparse.ArgumentParser()
parser.add_argument('--saved_model', default=None, type=str)
parser.add_argument('--use_cuda', default=True, type=bool)
parser.add_argument('--game_num', default=50, type=int)
parser.add_argument('--parallel_num', default=8, type=int)
parser.add_argument('--seed', default=10, type=int)
parser.add_argument('--logfile-name', default='aa', type=str)
args = parser.parse_args()

openings = [i for i in range(hex.SIZE * hex.SIZE)]
columns = 'abcdefghijklmnopqrstuvwxyz'


class agent:
    def __init__(self, exe):
        self.exe = exe 
        self.program = Program(self.exe, True)
        self.name = self.program.sendCommand("name\n").strip()
        self.lock  = threading.Lock()

    def sendCommand(self, command):
        self.lock.acquire()
        answer = self.program.sendCommand(command)
        self.lock.release()
        return answer

    def reconnect(self):
        self.program.terminate()
        self.program = Program(self.exe,True)
        self.lock = threading.Lock()


def move_to_cell(move):
    x = ord(move[0].lower())-ord('a')
    y = int(move[1:])-1
    return (x,y)


def single_play(base, net, i, simulation, flag, isMCTSPlayer):
    if i > 168 or i < 0:
        return
    game = gamestate(hex.SIZE)
    winner = None
    mohex_exe = "/home/dengwei/Downloads/benzene/build/src/mohex/mohex 2>/dev/null"
    mohex = agent(mohex_exe)
    if isMCTSPlayer:
        playerMCTS = MCTSPlayer(net=net, simulations_per_move=simulation, th=0, resign_threshold=-1.0)
        playerMCTS.initialize_game()
    else:
        playerMCTS = GreedyPlayer(net=net)
        command = 'param_mohex max_time 1\n'
        mohex.sendCommand(command)
        sys.stdout.flush()
    x = openings[i] // hex.SIZE
    y = openings[i] % hex.SIZE
    action = columns[y] + str(x+1)
    command = 'play b ' + action +'\n'

    fr = open(base+'games/game'+str(flag)+str(i)+'.txt', 'a')
    fr.write(action + '\n')
    fr.close()

    mohex.sendCommand(command)
    sys.stdout.flush()
    playerMCTS.make_move(x, y)
    game.place_black(move_to_cell(action))
    color = hex.WHITE
    while True:
        if color == flag:
            if color == hex.WHITE:
                command = 'genmove w\n'
            else:
                command = 'genmove b\n'
            action = mohex.sendCommand(command).strip()
            if action == 'resign':
                fr = open(base+'agnistMoHex.txt', 'a')
                if flag == hex.WHITE:
                    fr.write(str(i) + '\t' + 'black\n')
                else:
                    fr.write(str(i) + '\t' + 'white\n')
                fr.close()
                return
            sys.stdout.flush()
            fr = open(base+'games/game' +str(flag)+str(i) + '.txt', 'a')
            fr.write(action + '\n')
            fr.close()
            y = columns.index(action[0])
            x = int(action[1:]) - 1
            playerMCTS.make_move(x, y)
            if color == hex.WHITE:
                game.place_white(move_to_cell(action))
            else:
                game.place_black(move_to_cell(action))
        else:
            x, y = playerMCTS.get_move()
            playerMCTS.make_move(x, y)
            action = columns[y] + str(x + 1)
            fr = open(base+'games/game' +str(flag)+ str(i) + '.txt', 'a')
            fr.write(action + '\n')
            fr.close()
            if color == hex.WHITE:
                command = 'play w ' + action + '\n'
                game.place_white(move_to_cell(action))
            else:
                command = 'play b ' + action + '\n'
                game.place_black(move_to_cell(action))
            ans = mohex.sendCommand(command)
            sys.stdout.flush()
        color = -color
        if(game.winner() != game.PLAYERS["none"]):
            winner = game.winner()
            fr = open(base+'agnistMoHex.txt', 'a')
            if winner == game.PLAYERS["white"]:
                fr.write(str(i) + '\t' + 'white\n')
            else:
                fr.write(str(i) + '\t' + 'black\n')
            fr.close()
            return




def multi_play(base, net, start, end, simulation, flag):
    for i in range(start, end):
        single_play(base, net, i, simulation, flag)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')
    model = 'PC3-0.1-900.model'
    base = './mohexGamesPC3-0.1/'
    net = networkMCTS.PV(model, num1=1, num2=0)
    with multiprocessing.Manager() as mg:
        fr = open(base+'agnistMoHex.txt', 'a')
        fr.write('Mohex as Black\n')
        fr.close()
        flag = hex.BLACK
        for k in [19, 67, 76, 125, 160]:
            jobs = [Process(target=single_play, args=(base, net, k, 4000, flag, True))]
            for j in jobs:
                j.start()
            for j in jobs:
                j.join()
        #fr = open(base+'agnistMoHex.txt', 'a')
        #fr.write('Mohex as White\n')
        #fr.close()
        #flag = hex.WHITE
        #for k in [16, 49, 55, 63, 113, 153]:
        #    jobs = [Process(target=single_play, args=(base, net, k, 4000, flag, True))]
        #    for j in jobs:
        #        j.start()
        #    for j in jobs:
        #        j.join()
    
