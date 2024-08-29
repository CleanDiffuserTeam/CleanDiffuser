from gym.envs.registration import register

register(
    id='block-push-multi-modal-v0',
    entry_point='cleandiffuser.env.block_pushing.block_pushing_multimodal:BlockPushMultimodal',
    max_episode_steps=350,
)
