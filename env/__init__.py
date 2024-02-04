from gymnasium.envs.registration import register

register(
    id='WindyCartPole-v1',
    entry_point='env.windy_cartpole_env:WindyCartPole',
    max_episode_steps=500,
    reward_threshold=500,
)