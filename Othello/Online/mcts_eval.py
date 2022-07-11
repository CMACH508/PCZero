import numpy as np
import collections
import math

SIZE = 8
C_PUCT = 1.5


class DummyNode(object):

    def __init__(self):
        self.parent = None
        self.child_N = collections.defaultdict(float)
        self.child_W = collections.defaultdict(float)


class MCTSNode(object):

    def __init__(self, board, game, player, fmove=None, parent=None, cpuct=1.5):
        if parent is None:
            parent = DummyNode()
        self.parent = parent
        self.fmove = fmove
        self.game = game
        self.board = board
        self.player = player
        self.cpuct = cpuct
        self.is_expanded = False
        self.losses_applied = 0
        self.isPass = (self.game.getValidMoves(self.board, self.player)[-1] == 1)
        self.child_N = np.zeros([SIZE * SIZE + 1], dtype=np.float32)
        self.child_W = np.zeros([SIZE * SIZE + 1], dtype=np.float32)

        self.original_prior = np.zeros([SIZE * SIZE + 1], dtype=np.float32)
        self.child_prior = np.zeros([SIZE * SIZE + 1], dtype=np.float32)
        self.children = {}

    @property
    def child_action_score(self):
        return self.child_Q * self.player + self.child_U

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
        return self.Q * self.player

    def select_leaf(self):
        current = self
        while True:
            current.N += 1
            if not current.is_expanded:
                break
            legal_moves = current.game.getValidMoves(current.board, current.player)
            action_score = current.child_action_score
            min_score = np.min(action_score) - 1000
            for i in range(len(legal_moves)):
                if legal_moves[i] == 0:
                    action_score[i] = min_score
            best_move = np.argmax(action_score)
            current = current.maybe_add_child(best_move)
        return current

    def maybe_add_child(self, action):
        if action not in self.children:
            new_position, new_player, _, _ = self.game.getNextState(self.board, self.player, action)
            self.children[action] = MCTSNode(new_position, self.game, new_player, fmove=action, parent=self)
        return self.children[action]

    def add_virtual_loss(self, up_to):
        self.losses_applied += 1
        loss = self.player
        self.W += loss
        if self.parent is None or self is up_to:
            return
        self.parent.add_virtual_loss(up_to)

    def revert_virtual_loss(self, up_to):
        self.losses_applied -= 1
        revert = -1 * self.player
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
        over = self.game.getGameEnded(self.board)
        assert not over
        if self.is_expanded:
            self.revert_visits(up_to=up_to)
            return
        self.is_expanded = True
        self.original_prior = self.child_prior = move_probabilities
        #self.inject_noise()
        self.backup_value(value, up_to=up_to)

    def backup_value(self, value, up_to):
        self.W += value
        if self.parent is None or self is up_to:
            return
        self.parent.backup_value(value, up_to)

    def is_done(self):
        over = self.game.getGameEnded(self.board)
        return over

    def inject_noise(self):
        dirch = np.random.dirichlet([0.3] * (SIZE * SIZE + 1))
        self.child_prior = self.child_prior * 0.75 + dirch * 0.25

    def children_as_pi(self, squash=False):
        legal_moves = self.game.getValidMoves(self.board, self.player)
        for i in range(len(legal_moves)):
            if legal_moves[i] == 0:
                self.child_N[i] = 0
        probs = self.child_N
        if squash:
            probs = probs ** .95
        return probs / np.sum(probs)
