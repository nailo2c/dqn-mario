import random
import numpy as np

import gym
import torch
import torch.optim as optim

from collections import namedtuple
from gym import wrappers

from deepq.learn import learning
from deepq.model import DQN

from common.atari_wrapper import wrap_dqn
from common.schedule import PiecewiseSchedule


SEED = 0
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_BUFFER_SIZE=1000000
LEARNING_STARTS=50000
LEARNING_FREQ=4
FRAME_HISTORY_LEN=4
TARGET_UPDATE_FREQ=10000




def main(env):
    ### 首先要為隨時間改變的參數設定schedule
    # This is a just rough estimate
    num_iterations = float(40000000) / 4.0
    
    
    # define learning rate and exploration schedules
    lr_multiplier = 1.0
    
    lr_schedule = PiecewiseSchedule(endpoints=[
        (0, 1e-4 * lr_multiplier),
        (num_iterations / 10, 1e-4 * lr_multiplier),
        (num_iterations / 2 , 5e-5 * lr_multiplier),
    ], outside_value=5e-5 * lr_multiplier)
    
    
    exploration_schedule = PiecewiseSchedule([
        (0, 1.0),
        (1e6, 0.1),
        (num_iterations / 2, 0.01),
    ], outside_value=0.01)
    
    
    # optimizer
    OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])
    
    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(eps=1e-4),
    )
    
    learning(
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
    
    env = gym.make("PongNoFrameskip-v3")
    
    
    # set global seeds
    env.seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    
    
    # monitor & wrap the game
    env = wrap_dqn(env)
    
    expt_dir = 'video/gym-reslults'
    env = wrappers.Monitor(env, expt_dir, force=True, video_callable=lambda count: count % 50 == 0)

    
    # main
    main(env)