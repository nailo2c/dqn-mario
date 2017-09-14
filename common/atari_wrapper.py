# -*- coding: utf-8 -*-
import numpy as np
from collections import deque
import gym
from gym import spaces
from PIL import Image


# 重新實作step與reset，主要是reset，以當lives當指標來當作episode真正的結束
class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(EpisodicLifeEnv, self).__init__(env)
        self.lives = 0
        self.was_real_done  = True
        self.was_real_reset = False

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # 不太清楚使用這個lives是為了什麼
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        if self.was_real_done:
            obs = self.env.reset()
            self.was_real_reset = True
        else:
            obs, _, _, _ = self.env.step(0)
            self.was_real_reset = False
        self.lives = self.env.unwrapped.ale.lives()
        return obs



# 先隨機進行N次noop(no opearation)當作初始obs，但不知道這樣的好處為何?
class NoopResetEnv(gym.Wrapper):
    def __init__(self, env=None, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        self.env.reset()
        noops = np.random.randint(1, self.noop_max + 1)
        for _ in range(noops):
            obs, _, _, _ = self.env.step(0)
        return obs



# 實作每4個frames當作一次sample
class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        super(MaxAndSkipEnv, self).__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        # 選倒數兩個frame中較大的那一個，但我不太清楚為何要這樣?
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        # 清掉buffer，並掛上初始obs當作deque的初始狀態
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs



# 先Fire一發，再走一步action來當作初始obs，一樣還是不知道為何要這樣做
class FireResetEnv(gym.Wrapper):
    def __init__(self, env=None):
        super(FireResetEnv, self).__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, _, _ = self.env.step(1)
        obs, _, _, _ = self.env.step(2)
        return obs


    
# 遊戲畫面前處理
def _process_frame84(frame):
    img = np.reshape(frame, [210, 160, 3]).astype(np.float32)
    # RGB轉灰階
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    # 轉為Image物件，使用BILINEAR插值
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)



# 重新實作經過前處理的step與reset
class ProcessFrame84(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrame84, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame84(obs), reward, done, info

    def _reset(self):
        return _process_frame84(self.env.reset())



# 假如reward大於0 -> 1，等於0 -> 0，小於0 -> -1
class ClippedRewardsWrapper(gym.Wrapper):
    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, np.sign(reward), done, info



def wrap_dqn(env):
    assert 'NoFrameskip' in env.spec.id
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ProcessFrame84(env)
    env = ClippedRewardsWrapper(env)
    return env




# 針對mario去修改size
def _process_frame_mario(frame):
    img = np.reshape(frame, [224, 256, 3]).astype(np.float32)
    # RGB轉灰階
    # https://en.wikipedia.org/wiki/Grayscale#Converting_color_to_grayscale
    img = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    # 轉為Image物件，使用BILINEAR插值
    img = Image.fromarray(img)
    resized_screen = img.resize((84, 110), Image.BILINEAR)
    resized_screen = np.array(resized_screen)
    x_t = resized_screen[18:102, :]
    x_t = np.reshape(x_t, [84, 84, 1])
    return x_t.astype(np.uint8)



# 重新實作經過前處理的step與reset
class ProcessFrameMario(gym.Wrapper):
    def __init__(self, env=None):
        super(ProcessFrameMario, self).__init__(env)
        self.observation_space = spaces.Box(low=0, high=255, shape=(84, 84, 1))

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        return _process_frame_mario(obs), reward, done, info

    def _reset(self):
        return _process_frame_mario(self.env.reset())



def wrap_mario(env):
    assert 'SuperMarioBros' in env.spec.id
    env = MaxAndSkipEnv(env, skip=4)
    env = ProcessFrameMario(env)
    return env