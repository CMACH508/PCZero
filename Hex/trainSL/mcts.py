import numpy as np
import collections
import math
import hex
from copy import deepcopy

C_PUCT = 1.5

class DummyNode(object):

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)

class MCTSNode(object):

    def __init__(self, board, fmove=None, parent=None, cpuct=1.5):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.fmove = fmove
        self.board = board
        self.cpuct = cpuct
        self.is_expanded = False
        self.losses_applied = 0

        self.illegal_moves = 1000 * (1 - self.board.all_legal_moves())
        self.child_N = np.zeros([hex.SIZE * hex.SIZE], dtype=np.float32)
        self.child_W = np.zeros([hex.SIZE * hex.SIZE], dtype=np.float32)

        self.original_prior = np.zeros([hex.SIZE * hex.SIZE], dtype=np.float32)
        self.child_prior = np.zeros([hex.SIZE * hex.SIZE], dtype=np.float32)
        self.children = {}

    @property
    def child_action_score(self):
        return self.child_Q * self.board.to_play + self.child_U - self.illegal_moves

    @property
    def child_Q(self):
        return self.child_W / (1 + self.child_N)

    @property
    def child_U(self):
        return (self.cpuct * math.sqrt(1 + self.N) * self.child_prior / (1 + self.child_N))

    @property
    def Q(self):
        return self.W / (1 + self.N)

    @property
    def N(self):
        return self.parent.child_N[self.fmove]

    @N.setter
    def N(self, value):
        self.parent.child_N[self.fmove] = value

    @property
    def W(self):
        return self.parent.child_W[self.fmove]

    @W.setter
    def W(self, value):
        self.parent.child_W[self.fmove] = value

    @property
    def Q_perspective(self):
        return self.Q * self.board.to_play

    def select_leaf(self):
        current = self
        while True:
            current.N += 1

            if not current.is_expanded:
                break

            best_move = np.argmax(current.child_action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        if action not in self.children:
            new_position = self.board.move(action // hex.SIZE, action % hex.SIZE)
            self.children[action] = MCTSNode(new_position, fmove=action, parent=self)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        self.losses_applied += 1
        loss = self.board.to_play
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        revert = -1 * self.board.to_play
        self.W += revert
        if self.parent is None or self is up_to:
            return
        self.parent.revert_virtual_loss(up_to)

    def revert_visits(self, up_to):
        self.N -= 1
        if self.parent is None or self is up_to:
            return
        self.parent.revert_visits(up_to)

    def incorporate_results(self, move_probabilities, value, up_to):
        assert move_probabilities.shape == (hex.SIZE * hex.SIZE,)

        over = self.board.is_game_over()
        assert not over
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = move_probabilities

        #self.child_W = np.ones([hex.SIZE * hex.SIZE], dtype=np.float32) * value
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        over = self.board.is_game_over()
        return over

    def inject_noise(self):
        dirch = np.random.dirichlet([0.3] * (hex.SIZE * hex.SIZE))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def children_as_pi(self, squash=False):
        for p in range(hex.SIZE):
            for q in range(hex.SIZE):
                if not self.board.is_move_legal(p, q):
                    self.child_N[p*hex.SIZE+q] = 0
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)
