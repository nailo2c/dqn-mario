import random
import numpy as np
from collections import namedtuple

import gym
import ppaquette_gym_super_mario
from gym import wrappers

import torch
import torch.optim as optim

from deepq.learn import mario_learning
from deepq.model import DQN

from common.atari_wrapper import wrap_mario
from common.schedule import LinearSchedule


SEED = 0
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE = 1000000
LEARNING_STARTS = 50000
LEARNING_FREQ = 4
FRAME_HISTORY_LEN = 4
TARGET_UPDATE_FREQ = 10000
LEARNING_RATE = 0.00025
ALPHA = 0.95
EPS = 0.01



def main(env):
    ### 首先要為隨時間改變的參數設定schedule
    # This is a just rough estimate
    num_iterations = float(40000000) / 4.0
    
    
    # define exploration schedule
    exploration_schedule = LinearSchedule(1000000, 0.1)
    
    
    # optimizer
    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    
    optimizer = OptimizerSpec(
        constructor=optim.RMSprop,
        kwargs=dict(lr=LEARNING_RATE, alpha=ALPHA, eps=EPS),
    )
    
    
    mario_learning(
        env=env,
        q_func=DQN,
        optimizer_spec=optimizer,
        exploration=exploration_schedule,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        target_update_freq=TARGET_UPDATE_FREQ
    )

if __name__ == '__main__':
    
    env = gym.make("ppaquette/SuperMarioBros-1-1-v0")
    
    
    # set global seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    
    # monitor & wrap the game
    env = wrap_mario(env)
    
    expt_dir = 'video/mario'
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda count: count % 50 == 0)

    print('make env complete.')
    # main
    main(env)