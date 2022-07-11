import os
import binascii
from Game import Game
import glob

def transform_byte(b):
    s = bin(int(binascii.hexlify(b), 16))
    s = "000000000" + s[2:]
    s = s[-8:]
    return s
    
class WTB_reader:
    def __init__(self,name,path):
        self.name = name
        self.file_size = os.stat(path).st_size
        self.f = open(path,"rb")
        self.transformed = [transform_byte(self.f.read(1)) for _ in range(self.file_size)]
        self.century = self.index_to_int(0)
        self.year = self.index_to_int(1)
        self.month = self.index_to_int(2)
        self.day = self.index_to_int(3)    
        
        self.n1 = self.index_to_int(4,8)
        self.n2 = self.index_to_int(8,10)
        self.year_parties = self.index_to_int(10,12)
        self.p1 = self.index_to_int(12)
        self.p2 = self.index_to_int(13)
        self.p3 = self.index_to_int(14)
        self.reserve = self.index_to_int(15)

        self.n_games = (self.file_size-16)//68
        self.games = []
        self.read_games()
        
    def index_to_int(self,start,end=None):
        if end!=None:
            out = "".join(x for x in self.transformed[start:end] if x != "00000000")
            if not out:
                return 0
            return int(out,2)
        else:
            return int(self.transformed[start],2)
        
    def read_games(self):
        current = 16
        path_file_number = len(glob.glob(pathname='./games/*.txt'))
        print(path_file_number)
        for x in range(self.n_games):
            name = self.name + "_" + str(path_file_number + x)
            tournament_wording = self.index_to_int(current,current+2)
            player_black = self.index_to_int(current+2,current+4)
            player_white = self.index_to_int(current+4,current+6)
            real_score = self.index_to_int(current+6)
            theoretical_score = self.index_to_int(current+7)
            moves = [self.index_to_int(current+8+x) for x in range(60)]
            self.games.append(Game(name,tournament_wording,player_black,player_white,real_score,theoretical_score,moves))
            current += 68
    
    def export_games(self,path):
        for g in self.games:
            g.export(path)