import numpy as np


class board():
    def __init__(self, **kwargs):

        
        self.width = int(kwargs.get('w', 15))
        self.height = int(kwargs.get('h', 15))
        # need how many pieces in a row to win
        self.n = int(kwargs.get('n', 5))
        if self.width < self.n or self.height < self.n:
            raise Exception('board width and height can not be less than {}'.format(self.n))
        # player1 and player2
        self.players = [1, 2]
    def initialization(self, **kwargs):
        self.start_player = int(kwargs.get('start_player', np.random.randint(0, 2)))
        if self.start_player > 1:
            raise Exception("start player in 0 or 1")
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        self.states = np.zeros((self.height, self.width))
        self.last_move = (-1, -1)

    def current_state(self):
        




     def move2coor(self, move):
        """
        3*3 board's moves like:
        0 1 2
        3 4 5
        6 7 8
        and move 5's location is (1,2)
        """
        h = move // self.width
        w = move % self.width
        return h, w

    def coor2move(self, location):
        """
        3*3 board's coordinate like:
        1,1 1,2 1,3
        2,1 2,2 2,3
        3,1 3,2 3,3
        and coordinate 2,3's move is (2 - 1) * width + (3 - 1)
        """
        if len(location) != 2:
            return -1
        h = location[0]
        w = location[1]
        move = (h - 1) * self.width + w - 1
        if move not in range(self.width * self.height):
            move = -1
        return move