import gym

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


def get_model(env):
    model = Sequential()
    model.add(Flatten(input_shape=(1, *env.observation_space.shape)))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(16))
    model.add(Activation("relu"))
    model.add(Dense(env.action_space.n))
    model.add(Activation("linear"))
    return model


def main():
    env = gym.make("POMDP-v0")
    model = get_model(env)

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

    dqn.fit(env, nb_steps=10000, visualize=False, verbose=2)

    env.reset()
    dqn.test(env, nb_episodes=5, visualize=True)

    env.close()


if __name__ == "__main__":
    main()
