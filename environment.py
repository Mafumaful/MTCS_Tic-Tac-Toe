import gym
# print(gym.envs.registry.keys())

# test the environment
env = gym.make('Pendulum-v1')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action