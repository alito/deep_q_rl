import logging

import numpy

from .experiment import Experiment


class GymExperiment(Experiment):
    def __init__(self, environment, agent, resized_width, resized_height, resize_method,
                 num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng,
                 length_in_episodes=False):
        super(GymExperiment, self).__init__(agent, resized_width, resized_height, resize_method, 
            num_epochs, epoch_length, test_length, frame_skip, death_ends_episode, 
            max_start_nullops, rng, length_in_episodes)

        self.environment = environment
        self.done = False
        self.height, self.width, _ = self.environment.observation_space.shape
        self._internal_buffer = numpy.empty((self.height, self.width), dtype=numpy.uint8)

        if self.environment.frameskip != 1:
            logging.warn("We do our own frameskipping. Since the environment's frameskip is %s, this probably won't turn out very well" % self.environment.frameskip)

    def screen_size(self):
        return self.width, self.height

    def game_over(self):
        return self.done

    def lives(self):
        """
        We don't know, so just return one life unless the game is over
        """
        return self._lives

    def reset_game(self):
        self.environment.reset()
        self.done = False

    def map_action(self, action):
        """
        1-to-1
        """
        return action

    def interact(self, action):
        import cv2

        observation, reward, done, info = self.environment.step(action)

        self.done = done
        # Gym started putting the number of lives left in the info
        self._lives = info.get('ale.lives', 0 if done else 1)
        cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY, dst=self._internal_buffer)

        return reward, self._internal_buffer


