from gymnasium.envs.registration import register

register(
    id='TicTacEnv-v0',
    entry_point='gymnasium_env.envs:TicTacEnv',
)