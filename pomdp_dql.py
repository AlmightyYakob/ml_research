import pomdp_env  # NOQA
import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, LSTM
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def default_model(env):
    model = Sequential()
    model.add(Dense(16, input_shape=env.observation_space.shape))
    model.add(Flatten())
    model.add(Activation("relu"))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(env.action_space.n))
    model.add(Activation("linear"))
    return model


def lstm_model(env):
    model = Sequential()
    model.add(LSTM(32, input_shape=env.observation_space.shape))
    model.add(Activation("relu"))
    model.add(Dense(env.action_space.n))
    model.add(Activation("linear"))
    return model


def bootstrapped_train(env):
    # model = default_model(env)
    model = lstm_model(env)

    policy = EpsGreedyQPolicy(eps=0.1)
    memory = SequentialMemory(limit=100000, window_length=1)
    dqn = DQNAgent(
        model=model,
        nb_actions=env.action_space.n,
        memory=memory,
        nb_steps_warmup=10,
        target_model_update=1e-2,
        policy=policy,
    )
    dqn.compile(Adam(lr=1e-3), metrics=["mae"])

    dqn.fit(env, nb_steps=5000, visualize=False, verbose=1)

    env.reset()
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()


def main():
    env = gym.make("POMDP-v0")
    bootstrapped_train(env)


if __name__ == "__main__":
    main()
