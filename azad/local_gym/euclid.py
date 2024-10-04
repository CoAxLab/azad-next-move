import math
from azad.local_gym.wythoff import WythoffEnv

import numpy as np

def create_cold_board_euclid(m, n, default=0.0, cold_value=1):
    """Create a (m, n) binary board with cold moves as '1'"""
    cold_board_euclid = np.ones((m, n)) * default
    for r in range(m):
        for c in range(n):
            if r==0 or c ==0:
                if r==c==0: cold_board_euclid[r, c] = cold_value
            elif max(r,c) / min(r,c) < (1 + math.sqrt(5)) / 2:
                cold_board_euclid[r, c] = cold_value

    return cold_board_euclid


def locate_all_cold_moves(m, n):
    """Locate all the cold moves"""
    moves = []
    for r in range(m):
        for c in range(n):
            if r==0 or c ==0:
                if r==c==0: moves.append((r, c))
            elif max(r,c) / min(r,c) < (1 + math.sqrt(5)) / 2:
                moves.append((r, c))

    return moves


def create_moves(x, y):
    """Create all valid moves from (x, y)"""
    a, b = x, y
    moves = []
    for c in range(max(x, y)):
        if min(a, b) == 0:
            if a >= b:
                moves.append((c, b))
            if b >= a:
                moves.append((a, c))
        elif (max(a, b) - c) % min(a, b) == 0:
            if a >= b:
                moves.append((c, b))
            if b >= a:
                moves.append((a, c))
    assert(list(set(moves)) is not None)
    return list(set(moves))


class EuclidEnv(WythoffEnv):
    """Euclid's game template. 
    
    Note: subclass to use."""

    def __init__(self, m, n):
        super().__init__(m, n)

    def _create_moves(self):
        self.moves = create_moves(self.x, self.y)
        
    def get_locate_cold_moves(self, x, y, moves):
        cold_moves = []
        for move in moves:
            (r,c) = move
            if r==0 or c==0:
                if r==c==0: cold_moves.append(move)
            elif max(r,c) / min(r,c) < (1 + math.sqrt(5)) / 2:
                cold_moves.append(move)
        return cold_moves
    
    def get_cold_move_available(self, x, y, moves):
        colds = locate_all_cold_moves(x, y)
        #assert(type(moves)==list)
        #if set(colds).intersection(moves): assert(False)
        for move in self.moves:
            if colds.__contains__(move):
                print()
                print()
                print(f'x: {x}')
                print(f'y: {y}')
                print(f'moves: {moves}')
                print(f'colds: {colds}')
                print()
                print()
                assert(False)
                return True

        return False
    
    def get_create_moves(self):
        self._create_moves()
        return self.moves


class Euclid3x3(EuclidEnv):
    """A 3 by 3 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=3, n=3)


class Euclid5x5(EuclidEnv):
    """A 5 by 5 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=5, n=5)


class Euclid10x10(EuclidEnv):
    """A 10 by 10 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=10, n=10)


class Euclid15x15(EuclidEnv):
    """A 15 by 15 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=15, n=15)


class Euclid50x50(EuclidEnv):
    """A 50 by 50 Euclid game"""

    def __init__(self):
        EuclidEnv.__init__(self, m=50, n=50)
