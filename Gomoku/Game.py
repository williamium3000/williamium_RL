import sys
sys.path.append(".")
from Gomoku import Board
from Gomoku import humanPlayer
from Gomoku import vanillaMcts
import numpy as np
class Game(object):
    """
    Game play
    """

    def __init__(self, **kwargs):

        
        self.width = int(kwargs.get('w', 15))
        self.height = int(kwargs.get('h', 15))
        # need how many pieces in a row to win
        self.n = int(kwargs.get('n', 5))
        if self.width < self.n or self.height < self.n:
            raise Exception('board width and height can not be less than {}'.format(self.n))
        # player1 and player2
    def start_play(self, player1, player2, is_shown = False, **kwargs):
        """start a game between two players"""
        start_player = int(kwargs.get('start_player', np.random.randint(0, 2)))
        if start_player not in (0, 1):
            raise Exception('start_player should be either 0 (player1 first) '
                            'or 1 (player2 first)')
        self.board = Board.board(w = self.width, h = self.height, n = self.n)
        self.board.initialization(start_player)
        id1, id2 = self.board.players
        player1.set_player_id(id1)
        player2.set_player_id(id2)
        self.ids = [id1, id2]
        self.players = {id1: player1, id2: player2}
        if is_shown:
            self.print_board()

        while True:
            current_player = self.board.current_player
            print("current_player {}".format(current_player))
            player_in_turn = self.players[current_player]
            move = player_in_turn.play_action(self.board)
            self.board.step(move)

            if is_shown:
                self.print_board()

            end, winner = self.board.check_end()
            if end:
                if is_shown:
                    if winner != -1:
                        print("Game end. Winner is", self.players[winner])
                    else:
                        print("Game end. Tie")
                return winner
    def print_board(self):
        """Draw the board and show game info"""
        width = self.board.width
        height = self.board.height

        print("Player", self.players[self.ids[0]], "with X".rjust(3))
        print("Player", self.players[self.ids[1]], "with O".rjust(3))
        print()
        for x in range(width):
            print("{0:8}".format(x + 1), end='')
        print('\r\n')
        for i in range(height):
            print("{0:4d}".format(i + 1), end='')
            for j in range(width):
                p = self.board.states[i, j]
                if p == self.ids[0]:
                    print('X'.center(8), end='')
                elif p == self.ids[1]:
                    print('O'.center(8), end='')
                else:
                    print('_'.center(8), end='')
            print('\r\n\r\n')

if __name__ == "__main__":
    player1 = humanPlayer.humanPlayer()
    player2 = vanillaMcts.MCTSPlayer()
    test =  Game(w = 15, h = 15, n = 5)
    test.start_play(player1, player2, True)