"""The Experiment class handles the logic for training a deep
Q-learning agent in the Arcade Learning Environment.

Abstracted from Nathan Sprague's code

"""
import logging
import numpy as np
import cv2


class Experiment(object):
    def __init__(self, agent, resized_width, resized_height,
                 num_epochs, epoch_length, test_length,
                 frame_skip, death_ends_episode, max_start_nullops, rng,
                 length_in_episodes=False):
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_length = test_length
        self.frame_skip = frame_skip
        self.death_ends_episode = death_ends_episode
        self.resized_width = resized_width
        self.resized_height = resized_height

        self.buffer_length = 2
        self.buffer_count = 0
        self.screen_buffer = None

        self.terminal_lol = False # Most recent episode ended on a loss of life
        self.max_start_nullops = max_start_nullops
        self.rng = rng

        # Whether the lengths (test_length and epoch_length) are specified in 
        # episodes. This is mainly for testing
        self.length_in_episodes = length_in_episodes 

    def screen_size(self):
        raise NotImplementedError("Meant to return width and height")

    def game_over(self):
        raise NotImplementedError("Meant to return whether the game is over")

    def lives(self):
        raise NotImplementedError("Meant to return number of lives left")

    def reset_game(self):
        raise NotImplementedError("Meant to reset the environment")

    def map_action(self, action):
        raise NotImplementedError("Meant to map from agent's view of an action to the environment's view")

    def interact(self, action):
        """
        action is already in environment's terms
        """
        raise NotImplementedError("Big method. Sends action to environment, gets observation, and return reward and observation")


    def run(self):
        """
        Run the desired number of training epochs, a testing epoch
        is conducted after each training epoch.
        """

        # initialise buffer
        if self.screen_buffer is None:
            width, height = self.screen_size()
            self.screen_buffer = np.empty((self.buffer_length,
                                           height, width),
                                          dtype=np.uint8)

        for epoch in range(1, self.num_epochs + 1):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)

            if self.test_length > 0:
                self.agent.start_testing()
                self.run_epoch(epoch, self.test_length, True)
                self.agent.finish_testing(epoch)
        self.agent.cleanup()

    def run_epoch(self, epoch, num_steps, testing=False):
        """ Run one 'epoch' of training or testing, where an epoch is defined
        by the number of steps executed.  Prints a progress report after
        every trial

        Arguments:
        epoch - the current epoch number
        num_steps - steps per epoch
        testing - True if this Epoch is used for testing and not training

        """
        self.terminal_lol = False # Make sure each epoch starts with a reset.
        steps_left = num_steps
        while steps_left > 0:
            prefix = "testing" if testing else "training"
            logging.info(prefix + " epoch: " + str(epoch) + " steps_left: " +
                         str(steps_left))
            _, num_steps = self.run_episode(steps_left, testing)

            steps_left -= num_steps


    def _init_episode(self):
        """ This method resets the game if needed, performs enough null
        actions to ensure that the screen buffer is ready and optionally
        performs a randomly determined number of null action to randomize
        the initial game state."""

        if not self.terminal_lol or self.game_over():
            self.reset_game()

            if self.max_start_nullops > 0:
                random_actions = self.rng.randint(0, self.max_start_nullops+1)
                for _ in range(random_actions):
                    self.do_nothing() # Null action

        # Make sure the screen buffer is filled at the beginning of
        # each episode...
        self.do_nothing()
        self.do_nothing()

    def do_nothing(self):
        return self.act(self.map_action(0))

    def act(self, action):
        """Perform the indicated action for a single frame, return the
        resulting reward and store the resulting screen image in the
        buffer

        """
        index = self.buffer_count % self.buffer_length
        reward, self.screen_buffer[index, ...] = self.interact(action)
        self.buffer_count += 1
        return reward

    def step(self, action):
        """ Repeat one action the appopriate number of times and return
        the summed reward. """
        reward = 0
        for _ in range(self.frame_skip):
            reward += self.act(action)

        return reward

    def run_episode(self, max_steps, testing):
        """Run a single training episode.

        The boolean terminal value returned indicates whether the
        episode ended because the game ended or the agent died (True)
        or because the maximum number of steps was reached (False).
        Currently this value will be ignored.

        Return: (terminal, num_steps)

        """

        self._init_episode()

        start_lives = self.lives()

        action = self.map_action(self.agent.start_episode(self.get_observation()))
        num_steps = 0
        while True:
            reward = self.step(action)
            self.terminal_lol = (self.death_ends_episode and not testing and
                                 self.lives() < start_lives)
            terminal = self.game_over() or self.terminal_lol
            num_steps += 1

            if terminal or num_steps >= max_steps and not self.length_in_episodes:
                self.agent.end_episode(reward, terminal)
                break

            action = self.map_action(self.agent.step(reward, self.get_observation()))

        # if the lengths are in episodes, this episode counts as 1 "step"
        if self.length_in_episodes:
            return terminal, 1
        else:
            return terminal, num_steps


    def get_observation(self):
        """ Resize and merge the previous two screen images """

        assert self.buffer_count >= 2
        index = self.buffer_count % self.buffer_length - 1
        max_image = np.maximum(self.screen_buffer[index, ...],
                               self.screen_buffer[index - 1, ...])
        return self.resize_image(max_image)

    def resize_image(self, image):
        """ Appropriately resize a single image """

        return cv2.resize(image,
                              (self.resized_width, self.resized_height),
                              interpolation=cv2.INTER_LINEAR)

