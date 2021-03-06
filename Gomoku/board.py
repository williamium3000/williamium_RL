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
    def initialization(self, start_player):
        self.current_player = self.players[start_player]  # start player
        self.availables = list(range(self.width * self.height))
        self.states = np.zeros((self.height, self.width))
        self.last_move = (-1, -1)

    def current_state(self):
        """
        return the board state from the perspective of the current player.
        state shape: 4*width*height
        """
        states_ = np.zeros((2, self.height, self.width))
        states_[0] = self.states
        states_[1][self.last_move[0], self.last_move[1]] = 1
        return states_
    def step(self, move):
        i = move[0]
        j = move[1]
        self.states[i, j] = self.current_player
        self.availables.remove(self.coor2move((i, j)))
        self.current_player = (
            self.players[0] if self.current_player == self.players[1]
            else self.players[1]
        )
        self.last_move = (i, j)


    def __check_position(self, i, j):
        """
        check the right, upper right, bottom right direction to see if there is n piece in a row.
        """
        if j in range(self.width - self.n + 1):
            # check right
            if len(set(self.states[i, x] for x in range(j, j + self.n))) == 1:
                return True
            # check upper right
            if i in range(self.n - 1, self.height) and len(set(self.states[i - x, j + x] for x in range(self.n))) == 1:
                return True
            # check bottom right
            if i in range(self.height - self.n + 1) and len(set(self.states[i + x, j + x] for x in range(self.n))) == 1:
                return True
        else:
            return False
        

    def check_end(self):
        moved = set(range(self.width * self.height)) - set(self.availables)
        if len(moved) < self.n * 2 - 1:
            return False, -1
        for i in range(self.height):
            for j in range(self.width):
                if self.states[i, j] == 0:
                    continue
                if self.__check_position(i, j):
                    return True, self.states[i, j]
        if len(self.availables) == 0:
            return True, -1
        else:
            return False, -1

        

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
        move = h * self.width + w
        if move not in range(self.width * self.height):
            move = -1
        return move