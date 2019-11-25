import gym
import numpy as np

from gym import spaces
from random import randint
from collections import deque

DEFAULT_ENV_HISTORY_LENGTH = 50


class POMDPEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    # Each observation is a history of the possible
    # observations (z1, z2) so far.
    # E.g.
    # [
    # (0,0),    <
    # (0,0),    <
    # (0,0),    < Represents lack of observation so far
    # (0,0),    <
    # (0,0),    <
    # (0, 1),
    # (0, 1),
    # (1, 0),
    # (0, 1),
    # (1, 0)
    # ]

    # Actions = (a, b)
    action_space = spaces.Discrete(2)
    action_space.shape = (1, 2)

    reward_range = (-100, 100)

    action_a_t_matrix = np.array(
        [
            [0.1, 0.9, 0.0, 0.0, 0.0],
            [0.8, 0.2, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    action_b_t_matrix = np.array(
        [
            [0.0, 0.0, 0.9, 0.1, 0.0],
            [0.0, 0.0, 0.1, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )

    action_matrices = [action_a_t_matrix, action_b_t_matrix]

    states = np.arange(5)
    actions = np.arange(2)
    observations = np.arange(2)

    # For each state, weights are in the form (z1, z2)
    state_z_weights = np.array(
        [(0.7, 0.3), (0.4, 0.6), (0.5, 0.5), (0.5, 0.5), (0.5, 0.5)]
    )

    state_rewards = np.array([0, 0, 100, -100, 0])

    def __init__(self, history_length=DEFAULT_ENV_HISTORY_LENGTH, **kwargs):
        # Assign observation space to be Discrete(history_length),
        # with shape of (history_length, 2)

        self.observation_space = spaces.Discrete(history_length)
        self.observation_space.shape = (history_length, 2)
        self.history = deque([(0, 0)] * history_length, maxlen=history_length)

        self.current_state = randint(0, 1)
        self.done = False
        self.observed_in_current_state = False

    def observe(self):
        if not self.observed_in_current_state:
            choice = np.random.choice(
                self.observations, p=self.state_z_weights[self.current_state]
            )
            choice = np.eye(self.observation_space.shape[1], dtype=np.uint8)[choice]

            self.history.append(tuple(choice))
            self.observed_in_current_state = True
        return self.history

    def step(self, action: int):
        p_matrix = self.action_matrices[action]
        self.current_state = np.random.choice(
            self.states, p=p_matrix[self.current_state]
        )
        self.observed_in_current_state = False

        observation = self.observe()
        reward = self.state_rewards[self.current_state]
        self.done = True if self.current_state == 4 else False

        return (observation, reward, self.done, {})

    def reset(self):
        self.__init__()
        return self.observe()

    def render(self, mode="human"):
        pass

    def close(self):
        pass
