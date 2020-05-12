#!/usr/bin/env python3

import numpy as np
from sklearn.neighbors import KDTree


# This could be much improved: http://ann-benchmarks.com/index.html
class QEC:
    """Q-value (aka state-action-value) for episodic control

    """
    def __init__(self, actions, buffer_size, k):
        self.buffers = tuple([ActionBuffer(buffer_size) for _ in actions])  # Basically [s->Q(s,a) for a in A]
        self.k = k  # Number of nearest neighbors

    def estimate(self, state, action):
        """Estimates Q(s,a).
        If the state s has already been observed along with action a,
        the stored value (which is, remember, the discounted return value)
        is returned, else the average of the Q-values for action a of the
        k nearest neighbors of state s is returned.
        If there is not enough data for action a, +infinity is returned to have
        the opportunity of seeing a (s',a) pair for some state s'. Refer to the
        exploitation clause in MFECAgent.choose_action.

        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)

        if state_index:  # If (s,a) was observed before
            return buffer.values[state_index]
        if len(buffer) <= self.k:  # If there is not enough experience of action a
            return float("inf")

        value = 0.0
        neighbors = buffer.find_neighbors(state, self.k)
        for neighbor in neighbors:
            value += buffer.values[neighbor]
        return value / self.k

    def update(self, state, action, value, time):
        """Updates the Q-value to the maximum of the current Q-value (if it exists) and 'value'.
        If Q(s,a) does not exist in the buffer and the buffer is not full, creates it.

        Note: since the stored value can never decrease, it is not suited to rational action
        selection in stochastic environments.

        """
        buffer = self.buffers[action]
        state_index = buffer.find_state(state)
        if state_index:
            max_value = max(buffer.values[state_index], value)
            max_time = max(buffer.times[state_index], time)
            buffer.replace(state, max_value, max_time, state_index)
        else:
            buffer.add(state, value, time)


class ActionBuffer:
    """s -> Q(s,a) for some action a

    """
    def __init__(self, capacity):
        self._tree = None
        self.capacity = capacity
        self.states = []
        self.values = []
        self.times = []

    def find_state(self, state):
        """Returns the index of 'state' if it exists in the buffer, else None.

        """
        if self._tree:
            neighbor_idx = self._tree.query([state])[1][0][0]
            if np.allclose(self.states[neighbor_idx], state):
                return neighbor_idx
        return None

    def find_neighbors(self, state, k):
        """Looks for the k nearest neighbors of 'state' in the buffer.

        """
        return self._tree.query([state], k)[1][0] if self._tree else []

    def add(self, state, value, time):
        """Updates the KD-Tree structure of the buffer with (s,a).
        If the buffer is full, update only if (s,a) was done later than the 'youngest'
        pair in the buffer. If so, replace it.

        """
        if len(self) < self.capacity:
            self.states.append(state)
            self.values.append(value)
            self.times.append(time)
        else:
            min_time_idx = int(np.argmin(self.times))
            if time > self.times[min_time_idx]:
                self.replace(state, value, time, min_time_idx)
        self._tree = KDTree(np.array(self.states))  # quickfix

    def replace(self, state, value, time, index):
        self.states[index] = state
        self.values[index] = value
        self.times[index] = time

    def __len__(self):
        return len(self.states)
