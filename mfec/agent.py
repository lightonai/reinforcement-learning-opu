import numpy as np
from PIL import Image
import time

from mfec.qec import QEC

from lightonml.encoding.base import BinaryThresholdEncoder
from lightonml.projections.sklearn import OPUMap
from lightonopu.opu import OPU
from lightonopu.simulated_device import SimulatedOpuDevice  # To use the HPC only


class MFECAgent:
    """Agent to determine what actions have to be taken and then learns according to the returns.

    """
    def __init__(
        self,
        buffer_size,
        k,
        discount,
        epsilon,
        resize,
        state_dimension,
        actions  # Available actions
    ):
        self.rs = np.random.RandomState(None)
        self.size = resize
        self.memory = []  # To store useful data during an episode
        self.actions = actions
        self.qec = QEC(self.actions, buffer_size, k)
        self.discount = discount
        self.epsilon = epsilon

        self.state_dimension = state_dimension
        self.state = np.empty(state_dimension, np.uint8)
        self.action = int
        self.ttime = 0  # Time this (state, action) pair was chosen during training

        self.opu = OPU(verbose_level=0, n_components=state_dimension, #opu_device=SimulatedOpuDevice(),
                       max_n_features=(210*160 if self.size is None else np.prod(self.size)))
        self.opu.open()
        self.mapping = OPUMap(opu=self.opu, n_components=state_dimension, ndims=2)  # No flattening on the DMD


        self.threshold_enc = 20
        self.encoder = BinaryThresholdEncoder(threshold_enc=self.threshold_enc)

        self.previous_obs = []  # Trajectory / skipped frame hack
        self.lpo = 3  # Length of trajectory

    def choose_action(self, observation, explore=True):
        """When observing 'observation', chooses best action according to current
        knowledge with probability 1-epsilon and samples a random actio with
        probability epsilon.
        If 'explore' is False, epsilon is considered to be 0.

        """
        self.ttime += 1

        # Preprocess and project observation to state
        obs_processed = observation[..., 0]  # No useful information is lost
        obs_processed = obs_processed[:-38]  # FIXME: removes score for MsPacman
        obs_processed = self.encoder.transform(obs_processed)  # Converts to binary
        if self.size is not None:
            obs_processed = np.array(Image.fromarray(obs_processed).resize(self.size))

        self.previous_obs.append(obs_processed)
        if len(self.previous_obs) > self.lpo:
            self.previous_obs.pop(0)
        if len(self.previous_obs) == 1:
            self.previous_obs = [self.previous_obs[0] for _ in range(self.lpo)]
        obs_processed = sum(self.previous_obs) > 0  # Comment to disable trajec.

        self.state = self.opu.transform2d(obs_processed)  # Random projection

        # Exploration
        if explore and self.rs.random_sample() < self.epsilon:
            self.action = self.rs.choice(self.actions)

        # Exploitation
        else:
            values = [
                self.qec.estimate(self.state, action)
                for action in self.actions
            ]
            best_actions = np.argwhere(values == np.max(values)).flatten()  # Useful if theres more than one best action
            self.action = self.rs.choice(best_actions)

        return self.action

    def receive_reward(self, reward):
        """Store the relevant informations of the last action (in particular the obtained reward)

        """
        if self.action != 0:
            # Trying to force the agent to do nothing if nothing is gained by
            # pressing buttons (action 0 means 'no operation')
            reward -= 0
        self.memory.append(
            {
                "state": self.state,
                "action": self.action,
                "reward": reward,
                "time": self.ttime,
            }
        )

    def train(self):
        """Trains the model with the gathered data.

        """
        value = 0.0
        for _ in range(len(self.memory)):
            experience = self.memory.pop()
            value = value * self.discount + experience["reward"]  # "discounted return value"
            self.qec.update(
                experience["state"],
                experience["action"],
                value,
                experience["time"],
            )
        self.previous_obs.clear()

    def __del__(self):
        self.opu.close()
        # pass
