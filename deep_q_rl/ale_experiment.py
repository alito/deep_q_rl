"""The ALEExperiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Author: Nathan Sprague

"""
import logging

from .experiment import Experiment

class ALEExperiment(Experiment):
    def __init__(self, ale, agent, resized_width, resized_height,
                 num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng,
                 length_in_episodes=False):
        super(ALEExperiment, self).__init__(agent, resized_width, resized_height, num_epochs, epoch_length,
            test_length, frame_skip, death_ends_episode, max_start_nullops, rng, length_in_episodes)
        self.ale = ale
        self.min_action_set = ale.getMinimalActionSet()
        self.width, self.height = ale.getScreenDims()
        self._ale_buffer = None

    def screen_size(self):
        return self.width, self.height

    def game_over(self):
        return self.ale.game_over()

    def lives(self):
        return self.ale.lives()

    def reset_game(self):
        self.ale.reset()

    def map_action(self, action):
        return self.min_action_set[action]

    def interact(self, action):
        reward = self.ale.act(action)
        self._ale_buffer = self.ale.getScreenGrayscale(self._ale_buffer)
        return reward, self._ale_buffer

