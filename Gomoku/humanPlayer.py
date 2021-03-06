import sys
sys.path.append(".")
from Gomoku import Player

class humanPlayer(Player.player):
    def __init__(self):
        super(humanPlayer, self).__init__()
    def play_action(self, board):
        try:
            location = input("Your move: ")
            if isinstance(location, str):  # for python3
                move = [int(n) - 1 for n in location.split(",")]
                
        except Exception as e:
            move = (-1, -1)
        if move[0] == -1 or board.coor2move(move) not in board.availables:
            print("invalid move")
            move = self.play_action(board)
        return move

    def __str__(self):
        return "Human player(id:{})".format(self.id)