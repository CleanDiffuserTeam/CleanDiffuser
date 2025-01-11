from gym.envs.registration import register

register(
    id="robomimic-v0",
    entry_point="cleandiffuser.env.robomimic.robomimic_env:RobomimicEnv",
    max_episode_steps=800,
    reward_threshold=1.0,
)
