import gym_pomdp  # NOQA

import time
import gym
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, Activation
from keras.callbacks import ModelCheckpoint


def model():
    model = Sequential()
    model.add(LSTM(64, input_shape=()))
    model.add(Activation("relu"))

    return model


def main():
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


if __name__ == "__main__":
    main()
