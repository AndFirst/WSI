import os
import random
import time

import gym
import numpy as np


class Model:
    def __init__(self):
        ENVIRONMENT_NAME = 'Taxi-v3'
        ACTIONS_LEN = 6
        STATE_LEN = 500
        self.env = gym.make(ENVIRONMENT_NAME)
        self.q_table = self._create_q_table(STATE_LEN, ACTIONS_LEN)

    def train(self,
              episodes: int,
              learning_rate: float,
              discount_factor: float,
              strategy: str,
              **kwargs):
        def choose_action():
            if strategy == "greedy":
                return np.argmax(self.q_table[state, :])
            if strategy == "epsilon_greedy":
                epsilon = kwargs['epsilon']
                n = random.uniform(0, 1)
                if n > epsilon:
                    action = np.argmax(self.q_table[state, :])
                else:
                    action = self.env.action_space.sample()
                return action
            if strategy == "boltzmann":
                temperature = kwargs['temperature']

                exponent = np.true_divide(self.q_table[state, :], temperature)
                probs = np.exp(exponent) / np.sum(np.exp(exponent))
                threshold = random.uniform(0, sum(probs))
                prob_sum = 0

                for i, prob in enumerate(probs):
                    prob_sum += prob
                    if prob_sum > threshold:
                        return i

        for episode in range(episodes):
            state = self.env.reset()
            finished = False
            while not finished:
                action = choose_action()
                new_state, reward, finished, info = self.env.step(action)
                delta = reward + discount_factor * np.max(self.q_table[new_state, :]) - self.q_table[state, action]
                self.q_table[state, action] += learning_rate * delta

                state = new_state

    def evaluate(self,
                 episodes: int,
                 render: bool = False):
        rewards = []

        for episode in range(episodes):
            state = self.env.reset()
            episode_reward = 0

            finished = False
            while not finished:
                action = np.argmax(self.q_table[state, :])
                new_state, reward, finished, info = self.env.step(action)
                state = new_state
                episode_reward += reward

                if render:
                    os.system("cls")
                    self.env.render()
                    time.sleep(0.2)
            rewards.append(episode_reward)
        return np.mean(rewards)

    def _create_q_table(self, state_len, actions_len):
        return np.zeros(shape=(state_len, actions_len))
