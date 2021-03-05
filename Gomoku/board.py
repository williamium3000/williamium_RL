import numpy as np


class board():
    def __init__(self, **kwargs):

        
        self.width = int(kwargs.get('w', 15))
        self.height = int(kwargs.get('h', 15))
        # need how many pieces in a row to win
        self.n = int(kwargs.get('n', 5))
        # player1 and player2
        self.players = [1, 2]
    def initialization(self, **kwargs):
        if self.width < self.n or self.height < self.n:
            raise Exception('board width and height can not be less than {}'.format(self.n))
        self.start_player = int(kwargs.get('start_player', np.random.randint(0, 2)))
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        self.states = np.zeros((self.width, self.height))
        self.last_move = (-1, -1)
    def 