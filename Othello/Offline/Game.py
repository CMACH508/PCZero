import os

class Game:
    def __init__(self,name,tournament_wording,player_black,player_white,real_score,theoretical_score,moves):
        self.name = name
        self.tournament_wording = tournament_wording
        self.player_black = player_black
        self.player_white = player_white
        self.real_score = real_score
        self.theoretical_score = theoretical_score
        self.moves = moves
        
    def export(self,path):
        out_file = os.path.join(path,self.name + ".txt")
        f = open(out_file,"w")
        f.write('Black player: ' + str(self.player_black) + '\n')
        f.write('White player: ' + str(self.player_white) + '\n')
        f.write('Real score: ' + str(self.real_score) + '\n')
        f.write('Theoretical score: ' + str(self.theoretical_score) + '\n')
        for m in self.moves:
            f.write(self.translate_move(m))
            f.write("\n")
        f.close()

    def translate_move(self,m):
        return "{}-{}".format(m//10-1,m%10-1)