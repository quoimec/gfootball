# coding=utf-8
# Copyright 2019 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Environment that can be used with OpenAI Baselines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import cv2
import math
from gfootball.env import observation_preprocessing
import gym
import numpy as np
import json


class PeriodicDumpWriter(gym.Wrapper):
  """A wrapper that only dumps traces/videos periodically."""

  def __init__(self, env, dump_frequency):
    gym.Wrapper.__init__(self, env)
    self._dump_frequency = dump_frequency
    self._original_render = env._config['render']
    self._original_dump_config = {
        'write_video': env._config['write_video'],
        'dump_full_episodes': env._config['dump_full_episodes'],
        'dump_scores': env._config['dump_scores'],
    }
    self._current_episode_number = 0

  def step(self, action):
    return self.env.step(action)

  def reset(self):
    if (self._dump_frequency > 0 and
        (self._current_episode_number % self._dump_frequency == 0)):
      self.env._config.update(self._original_dump_config)
      self.env._config.update({'render': True})
    else:
      self.env._config.update({'render': self._original_render,
                               'write_video': False,
                               'dump_full_episodes': False,
                               'dump_scores': False})
    self._current_episode_number += 1
    return self.env.reset()


class Simple115StateWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to 115-features state."""

  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(), 115)
    self.observation_space = gym.spaces.Box(
        low=-1, high=1, shape=shape, dtype=np.float32)

  def observation(self, observation):
    """Converts an observation into simple115 format.

    Args:
      observation: observation that the environment returns

    Returns:
      (N, 155) shaped representation, where N stands for the number of players
      being controlled.
    """
    final_obs = []
    for obs in observation:
      o = []
      o.extend(obs['left_team'].flatten())
      o.extend(obs['left_team_direction'].flatten())
      o.extend(obs['right_team'].flatten())
      o.extend(obs['right_team_direction'].flatten())

      # If there were less than 11vs11 players we backfill missing values with
      # -1.
      # 88 = 11 (players) * 2 (teams) * 2 (positions & directions) * 2 (x & y)
      if len(o) < 88:
        o.extend([-1] * (88 - len(o)))

      # ball position
      o.extend(obs['ball'])
      # ball direction
      o.extend(obs['ball_direction'])
      # one hot encoding of which team owns the ball
      if obs['ball_owned_team'] == -1:
        o.extend([1, 0, 0])
      if obs['ball_owned_team'] == 0:
        o.extend([0, 1, 0])
      if obs['ball_owned_team'] == 1:
        o.extend([0, 0, 1])

      active = [0] * 11
      if obs['active'] != -1:
        active[obs['active']] = 1
      o.extend(active)

      game_mode = [0] * 7
      game_mode[obs['game_mode']] = 1
      o.extend(game_mode)
      final_obs.append(o)
    return np.array(final_obs, dtype=np.float32)


class PixelsStateWrapper(gym.ObservationWrapper):
  """A wrapper that extracts pixel representation."""

  def __init__(self, env, grayscale=True,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._grayscale = grayscale
    self._channel_dimensions = channel_dimensions
    self.observation_space = gym.spaces.Box(
        low=0, high=255,
        shape=(self.env.unwrapped._config.number_of_players_agent_controls(),
               channel_dimensions[1], channel_dimensions[0],
               1 if grayscale else 3),
        dtype=np.uint8)

  def observation(self, obs):
    o = []
    for observation in obs:
      frame = observation['frame']
      if self._grayscale:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
      frame = cv2.resize(frame, (self._channel_dimensions[0],
                                 self._channel_dimensions[1]),
                         interpolation=cv2.INTER_AREA)
      if self._grayscale:
        frame = np.expand_dims(frame, -1)
      o.append(frame)
    return np.array(o, dtype=np.uint8)


class SMMWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to a minimap."""

  def __init__(self, env,
               channel_dimensions=(observation_preprocessing.SMM_WIDTH,
                                   observation_preprocessing.SMM_HEIGHT)):
    gym.ObservationWrapper.__init__(self, env)
    self._channel_dimensions = channel_dimensions
    shape = (self.env.unwrapped._config.number_of_players_agent_controls(),
             channel_dimensions[1], channel_dimensions[0],
             len(observation_preprocessing.get_smm_layers(
                 self.env.unwrapped._config)))
    self.observation_space = gym.spaces.Box(
        low=0, high=255, shape=shape, dtype=np.uint8)

  def observation(self, obs):
    return observation_preprocessing.generate_smm(
        obs, channel_dimensions=self._channel_dimensions,
        config=self.env.unwrapped._config)


class SingleAgentObservationWrapper(gym.ObservationWrapper):
  """A wrapper that converts an observation to a minimap."""

  def __init__(self, env):
    gym.ObservationWrapper.__init__(self, env)
    self.observation_space = gym.spaces.Box(
        low=env.observation_space.low[0],
        high=env.observation_space.high[0],
        dtype=env.observation_space.dtype)

  def observation(self, obs):
    return obs[0]


class SingleAgentRewardWrapper(gym.RewardWrapper):
  """A wrapper that converts an observation to a minimap."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)

  def reward(self, reward):
    return reward[0]


class CheckpointRewardWrapper(gym.RewardWrapper):
  """A wrapper that adds a dense checkpoint reward."""

  def __init__(self, env):
    gym.RewardWrapper.__init__(self, env)
    self._collected_checkpoints = {True: 0, False: 0}
    self._num_checkpoints = 10
    self._checkpoint_reward = 0.1

  def reset(self):
    self._collected_checkpoints = {True: 0, False: 0}
    return self.env.reset()

  def reward(self, reward):
    if self.env.unwrapped.last_observation is None:
      return reward

    assert len(reward) == len(self.env.unwrapped.last_observation)

    for rew_index in range(len(reward)):
      o = self.env.unwrapped.last_observation[rew_index]
      is_left_to_right = o['is_left']

      if reward[rew_index] == 1:
        reward[rew_index] += self._checkpoint_reward * (
            self._num_checkpoints -
            self._collected_checkpoints[is_left_to_right])
        self._collected_checkpoints[is_left_to_right] = self._num_checkpoints
        continue

      # Check if the active player has the ball.
      if ('ball_owned_team' not in o or
          o['ball_owned_team'] != (0 if is_left_to_right else 1) or
          'ball_owned_player' not in o or
          o['ball_owned_player'] != o['active']):
        continue

      if is_left_to_right:
        d = ((o['ball'][0] - 1) ** 2 + o['ball'][1] ** 2) ** 0.5
      else:
        d = ((o['ball'][0] + 1) ** 2 + o['ball'][1] ** 2) ** 0.5

      # Collect the checkpoints.
      # We give reward for distance 1 to 0.2.
      while (self._collected_checkpoints[is_left_to_right] <
             self._num_checkpoints):
        if self._num_checkpoints == 1:
          threshold = 0.99 - 0.8
        else:
          threshold = (0.99 - 0.8 / (self._num_checkpoints - 1) *
                       self._collected_checkpoints[is_left_to_right])
        if d > threshold:
          break
        reward[rew_index] += self._checkpoint_reward
        self._collected_checkpoints[is_left_to_right] += 1
    return reward


class FrameStack(gym.Wrapper):
  """Stack k last observations."""

  def __init__(self, env, k):
    gym.Wrapper.__init__(self, env)
    self.obs = collections.deque([], maxlen=k)
    low = env.observation_space.low
    high = env.observation_space.high
    low = np.concatenate([low] * k, axis=-1)
    high = np.concatenate([high] * k, axis=-1)
    self.observation_space = gym.spaces.Box(
        low=low, high=high, dtype=env.observation_space.dtype)

  def reset(self):
    observation = self.env.reset()
    self.obs.extend([observation] * self.obs.maxlen)
    return self._get_observation()

  def step(self, action):
    observation, reward, done, info = self.env.step(action)
    self.obs.append(observation)
    return self._get_observation(), reward, done, info

  def _get_observation(self):
    return np.concatenate(list(self.obs), axis=-1)


class Point:
    
    def __init__(self, *, x, y):
        self.x = x
        self.y = y
        
    def distance(self, x, y):
        return math.sqrt(math.pow(x - self.x, 2) + math.pow(y - self.y, 2))
        
class Area:
    
    def __init__(self, *, origin, width, height):
        
        halfWidth = width * 0.5
        halfHeight = height * 0.5
        
        self.width = width
        self.height = height
        self.topLeft = Point(x = origin.x - halfWidth, y = origin.y + halfHeight)
        self.topRight = Point(x = origin.x + halfWidth, y = origin.y + halfHeight)
        self.bottomLeft = Point(x = origin.x - halfWidth, y = origin.y - halfHeight)
        self.bottomRight = Point(x = origin.x + halfWidth, y = origin.y - halfHeight)
        
    def contains(self, x, y):    
        
        return (x >= self.topLeft.x and x <= self.bottomRight.x) and (y >= self.bottomRight.y and y <= self.topLeft.y)

class RewardTrack:
    
    def __init__(self):
        self.rewards = {}
        
    def update(self, reward):
        for key, value in reward.items():
            if key in self.rewards:
                self.rewards[key] += value
            else:
                self.rewards[key] = value
        
    def save(self, *, left, score, path):
        with open(path, "a+") as file:
            file.write("{}:{} - {}\n".format(score[int(not left)], score[int(left)], json.dumps(self.rewards)))
        
    def reset(self):
        self.rewards = {}
        
class PassTrack:
    
    def __init__(self):
        
        self.count = 0
        self.pending = False
        self.opposition = []
        self.start = None
        self.end = None
        self.player = None
        self.furthest = None
        self.received = False
        self.intercepted = False

    def track(self, *, left, action, ball, opposition, posession, mode, player):
        
        if action in [9, 10, 11] and posession:
            
            # Pass Started
            self.start = ball
            self.pending = True
            self.opposition = opposition
            self.player = player
            
        elif (self.count > 0 or self.pending) and (posession == False or mode != 0):
            
            # Pass Intercepted
            self.pending = False
            self.count = 0
            self.opposition = []
            self.player = None
            self.furthest = None
            self.intercepted = mode == 0
        
        elif self.pending and posession and player != self.player:
            
            # Pass Received
            self.received = True
            self.pending = False
            self.end = ball
            self.count += 1
            
    def finished(self):
        
        self.intercepted = False
        
        if self.received:
            self.received = False
            self.start = None
            self.end = None
            self.opposition = []
            self.player = None
            
    def reset(self):
        
        self.count = 0
        self.pending = False
        self.opposition = []
        self.start = None
        self.end = None
        self.player = None
        self.furthest = None
        self.received = False
        self.intercepted = False
            
class RoleRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)
        
        # self.LeftGoal = Area(origin = Point(x = -1.05, y = 0.0), width = 0.1, height = 0.084)
        # self.RightGoal = Area(origin = Point(x = 1.05, y = 0.0), width = 0.1, height = 0.084)
        
        self.LeftGoal = Point(x = -1.0, y = 0.0)
        self.RightGoal = Point(x = 1.0, y = 0.0)
        self.LeftBox = Area(origin = Point(x = -0.9, y = 0.0), width = 0.2, height = 0.36)
        self.RightBox = Area(origin = Point(x = 0.9, y = 0.0), width = 0.2, height = 0.36)
        self.LeftHalf = Area(origin = Point(x = -0.5, y = 0.0), width = 1.0, height = 0.84)
        self.RightHalf = Area(origin = Point(x = 0.5, y = 0.0), width = 1.0, height = 0.84)
        
        self.RewardTrack = RewardTrack()
        self.PassTrack = PassTrack()
        
    def reward(self, reward):

        if self.env.unwrapped.last_observation is None:
            return reward

        assert len(reward) == len(self.env.unwrapped.last_observation)

        for index in range(len(reward)):
            
            observation = self.env.unwrapped.last_observation[index]
            
            if observation["steps_left"] == 2999:
                self.RewardTrack.reset()
                self.PassTrack.reset()
            
            left = observation['is_left']
            agent = "left" if left else "right"
            computer = "left" if not left else "right"
            
            Position = lambda object: Point(x = object[0], y = object[1])
            
            action = self.env.unwrapped._agent._action
            ball = Position(observation["ball"])
            keeper = Position(observation["{}_team".format(agent)][np.where(observation["{}_team_roles".format(agent)] == 0)][0])
            opposition = observation["{}_team".format(computer)]
            
            if observation["ball_owned_team"] == -1:
                posession = None
            elif observation["ball_owned_team"] == 0:
                posession = left
            elif observation["ball_owned_team"] == 1:
                posession = not left
            
            self.PassTrack.track(left = left, action = action, ball = ball, opposition = opposition, posession = posession, mode = observation["game_mode"], player = observation["ball_owned_player"])
            
            rewards = {
                "BaseReward": float(reward[index]),
                "KeeperPosition": self.keeperPosition(left = left, keeper = keeper, ball = ball),
                "KeeperPositionScored": self.keeperPositionScored(left = left, reward = reward[index], keeper = keeper),
                # "ChainedPasses": self.chainedPasses(),
                "BisectedPasses": self.bisectedPasses(left = left, ball = ball, opposition = opposition),
                "ForwardPasses": self.forwardPasses(left = left, ball = ball),
                "InterceptedPasses": self.interceptedPass(),
                "ChainedPassesScored": self.chainedPassesScored(reward = reward[index]),
                "ShotReward": self.shotReward(left = left, action = action, ball = ball)
            }
            
            reward[index] = sum(rewards.values())
            
            self.PassTrack.finished()
            self.RewardTrack.update(rewards)
            
            if observation["steps_left"] == 100: self.RewardTrack.save(left = left, score = observation["score"], path = "/home/charlie/Projects/Python/Football/rewards.txt")
                    
        return reward        

    def keeperPosition(self, *, left, keeper, ball, discount = 0.0001):
    
        """ Keeper Position:
            Check the keeper position when the ball is in the agent's half.
            If the keeper is more than 30% of the way up it's half, punish the agent based on their distance from the 30% mark.
            
            Reward Graph: https://www.desmos.com/calculator/pj6bk3c7wz
        """
        
        half = self.LeftHalf if left else self.RightHalf
        goalline = self.LeftGoal if left else self.RightGoal
        distance = goalline.distance(x = keeper.x, y = keeper.y)
        
        if half.contains(x = ball.x, y = ball.y) and distance > 0.4:
            return -((distance - 0.4) / 3) * discount
        else:
            return 0
    
    def keeperPositionScored(self, *, left, reward, keeper):
        
        """ Keeper Position Scored:
            When the opposition scores, check the keeper position to see how far they are from their base goal line.
            If the keeper is far off the baseline, punish the agent based of their distance from the goal.
            
            Reward Graph: https://www.desmos.com/calculator/490p558fho
        """
        
        if reward != -1: 
            return 0
        
        goalline = self.LeftGoal if left else self.RightGoal
        distance = goalline.distance(x = keeper.x, y = keeper.y)
        
        if distance <= 0.15:
            return 0
        
        return -(math.pow(distance - 0.15, 2) * 300)
    
    def chainedPasses(self):
        
        if not self.PassTrack.received: return 0
        
        chained = lambda x: 0 if x not in range(1, 10) else 9.974660000000001 * 10 ** -18 + 0.004 * x - 0.0004 * x ** 2
        
        return chained(self.PassTrack.count)
    
    def bisectedPasses(self, *, left, ball, opposition, discount = 0.0008):
        
        if not self.PassTrack.received: return 0
        
        if (left and (self.PassTrack.start.y < -0.75 or self.PassTrack.start.y > 0.9)) or (not left and (self.PassTrack.start.y > 0.75 or self.PassTrack.start.y < -0.9)): return 0
        
        distance = lambda start, end, player: abs((start.y - end.y) * player.x + (end.x - start.y) * player.y + (start.x * end.y - end.x * start.y)) / ((start.y - end.y) ** 2 + (end.x - start.y) ** 2) ** (1/2)
            
        bisection = lambda distances: discount * sum(distances)
        
        minimum = min(self.PassTrack.start.x, ball.x)
        maximum = max(self.PassTrack.start.x, ball.x)
        
        active = list(filter(lambda a: a[0] >= minimum and a[0] <= maximum, (self.PassTrack.opposition + opposition)))
        distances = list(filter(lambda b: b > 0, map(lambda a: 0.42 - distance(self.PassTrack.start, ball, Point(x = a[0], y = a[1])), active)))
            
        return bisection(distances)    
    
    def forwardPasses(self, *, left, ball, discount = 0.002):
        
        if not self.PassTrack.received or (self.PassTrack.start is None or self.PassTrack.end is None): return 0
        
        if (left and (self.PassTrack.start.y < -0.8 or self.PassTrack.start.y > 0.9)) or (not left and (self.PassTrack.start.y > 0.8 or self.PassTrack.start.y < -0.9)): return 0
        
        if self.PassTrack.furthest is None: self.PassTrack.furthest = self.PassTrack.start.y
        
        distance = lambda a, b: abs(a - b) * (discount / 2.0)
        
        if (left and self.PassTrack.end.y > self.PassTrack.start.y and self.PassTrack.end.y > self.PassTrack.furthest) or (not left and self.PassTrack.end < self.PassTrack.start.y and self.PassTrack.end.y < self.PassTrack.furthest):
            reward = distance(self.PassTrack.end.y, self.PassTrack.furthest)
        else:
            reward = 0
            
        self.PassTrack.furthest = max(self.PassTrack.furthest, self.PassTrack.end.y, self.PassTrack.start.y) if left else min(self.PassTrack.furthest, self.PassTrack.end.y, self.PassTrack.start.y)
        
        return reward  
        
    def interceptedPass(self, discount = 0.002):
        
        if self.PassTrack.intercepted:
            return -1 * discount
        else:
            return 0
        
    def chainedPassesScored(self, *, reward, discount = 0.01):
        
        if reward == 1: 
            return discount * self.PassTrack.count
        else:
            return 0
            
    def shotReward(self, *, left, action, ball, discount = 0.0009):
        
        if action == 12 and ((left and ball.y > 0.75 and ball.y < 1.0) or (not left and ball.y < -0.75 and ball.y > -1.0)):
            return (0.25 - abs(0.9 - abs(ball.y))) * discount
        else: 
            return 0
            