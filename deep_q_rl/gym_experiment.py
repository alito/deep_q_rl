import logging

import numpy

from .experiment import Experiment


class GymExperiment(Experiment):
    def __init__(self, environment, agent, resized_width, resized_height,
                 num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng,
                 length_in_episodes=False):
        super(GymExperiment, self).__init__(agent, resized_width, resized_height, num_epochs, epoch_length,
            test_length, frame_skip, death_ends_episode, max_start_nullops, rng, length_in_episodes)

        self.environment = environment
        self.done = False
        self.min_action_set = self.environment.getMinimalActionSet()
        self.height, self.width, _ = self.environment.observation_space.shape
        self._internal_buffer = numpy.empty((self.height, self.width, 1), dtype=np.uint8)

    def screen_size(self):
        return self.width, self.height

    def game_over(self):
        return self.done

    def lives(self):
        return self.ale.lives()

    def reset_game(self):
        self.environment.reset()

    def map_action(self, action):
        """
        1-to-1
        """
        return action

    def interact(self, action):
        import cv2

        observation, reward, done, info = self.environment.set(action)

        self.done = done
        cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY, dst=self._internal_buffer)

        return reward, self._internal_buffer


