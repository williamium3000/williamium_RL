from abc import ABC, abstractmethod

class player(ABC):
    def __init__(self):
        self.id = -1
    def set_player_id(self, id):
        self.id = id
    @abstractmethod
    def play_action(self, board):
        pass
    @abstractmethod
    def __str__(self):
        pass