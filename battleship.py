import time
import gym
import gym_pomdp  # NOQA

env = gym.make("Battleship-v0")
env.reset()

print("Action Space:", env.action_space)
print("Obs Space:", env.observation_space)

for i in range(env.action_space.n):
    time.sleep(0.5)

    # observation, reward, done, info = env.step(env.action_space.sample())
    observation, _, done, info = env.step(i)
    print(observation)
    env.render()

    if done:
        break

env.close()
