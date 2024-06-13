from typing import List, Dict, Optional, Optional
import numpy as np
import gym
from gym.spaces import Box
from cleandiffuser.env.kitchen.base import KitchenBase

class KitchenLowdimWrapper(gym.Env):
    def __init__(self,
            env: KitchenBase,
            init_qpos: Optional[np.ndarray]=None,
            init_qvel: Optional[np.ndarray]=None,
            render_hw = (240,360)
        ):
        self.env = env
        self.all_tasks = env.ALL_TASKS
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
        self.render_hw = render_hw

    @property
    def action_space(self):
        return self.env.action_space
    
    @property
    def observation_space(self):
        return self.env.observation_space
    
    @property
    def tasks_to_complete(self):
        return self.env.tasks_to_complete
    
    def set_init_q(self, init_qpos, init_qvel):
        self.init_qpos = init_qpos
        self.init_qvel = init_qvel
            
    def seed(self, seed=None):
        return self.env.seed(seed)

    def reset(self):
        if self.init_qpos is not None:
            print("Load known initial conditions.")
            # reset anyway to be safe, not very expensive
            _ = self.env.reset()
            # start from known state
            self.env.set_state(self.init_qpos, self.init_qvel)
            obs = self.env.get_observation()
            return obs
        else:
            return self.env.reset()

    def render(self, mode='rgb_array'):
        h, w = self.render_hw
        return self.env.render(mode=mode, width=w, height=h)
    
    def step(self, a):
        return self.env.step(a)
