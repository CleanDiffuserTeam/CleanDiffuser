from gym.envs.registration import register

register(
    id='pusht-v0',
    entry_point='cleandiffuser.env.pusht.pusht_env:PushTEnv',
    max_episode_steps=300,
    reward_threshold=1.0
)

register(
    id='pusht-keypoints-v0',
    entry_point='cleandiffuser.env.pusht.pusht_keypoints_env:PushTKeypointsEnv',
    max_episode_steps=300,
    reward_threshold=1.0
)

register(
    id='pusht-image-v0',
    entry_point='cleandiffuser.env.pusht.pusht_image_env:PushTImageEnv',
    max_episode_steps=300,
    reward_threshold=1.0
)


