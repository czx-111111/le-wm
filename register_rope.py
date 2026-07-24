from gym.envs.registration import register

register(
    id='flex_rope-v0',
    entry_point='deformable_env.FlexEnvWrapper:FlexEnvWrapper',
    kwargs={'object_name': 'rope'},
    max_episode_steps=400,
    reward_threshold=1.0,
)
