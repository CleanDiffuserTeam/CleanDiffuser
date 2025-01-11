from gym.envs.registration import register

register(
    id="libero-object-v0",
    entry_point="cleandiffuser.env.libero.v0:LiberoObjectEnv",
    max_episode_steps=1000,
    reward_threshold=1.0,
)

register(
    id="libero-goal-v0",
    entry_point="cleandiffuser.env.libero.v0:LiberoGoalEnv",
    max_episode_steps=1000,
    reward_threshold=1.0,
)

register(
    id="libero-spatial-v0",
    entry_point="cleandiffuser.env.libero.v0:LiberoSpatialEnv",
    max_episode_steps=1000,
    reward_threshold=1.0,
)

register(
    id="libero-10-v0",
    entry_point="cleandiffuser.env.libero.v0:Libero10Env",
    max_episode_steps=1000,
    reward_threshold=1.0,
)

register(
    id="libero-90-v0",
    entry_point="cleandiffuser.env.libero.v0:Libero90Env",
    max_episode_steps=1000,
    reward_threshold=1.0,
)
