import types
import warnings
import gym
import d4rl
import pytest

d4rl_env_ids = [
    "halfcheetah-medium-v0",
    "halfcheetah-medium-replay-v0",
    "halfcheetah-medium-expert-v0",
    "hopper-medium-v0",
    "hopper-medium-replay-v0",
    "hopper-medium-expert-v0",
    "walker2d-medium-v0",
    "walker2d-medium-replay-v0",
    "walker2d-medium-expert-v0",
    "antmaze-medium-play-v0",
    "antmaze-medium-diverse-v0",
    "antmaze-large-play-v0",
    "antmaze-large-diverse-v0",
]

@pytest.mark.parametrize("env_id", d4rl_env_ids)
def test_d4rl_env(env_id):
    try:
        env = gym.make(env_id)
        env.reset()
        obs, reward, done, info = env.step(env.action_space.sample())
        assert env is not None
        assert obs is not None
    except Exception as e:
        pytest.fail(f"Failed to load environment {env_id} with error: {e}")




