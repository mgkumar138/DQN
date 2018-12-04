import gym
import numpy as np
import random
from collections import deque
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import Adam
from keras import losses
import matplotlib.pyplot as plt

# https://towardsdatascience.com/reinforcement-learning-w-keras-openai-dqns-1eed3a5338c
# pip install --no-index -f https://github.com/Kojoley/atari-py/releases atari_py


''' Version 1 
- Object based programming
- environment are continuous state space with deterministic actions 
- defined agent with memory & reply component 
- 2 models, target & actual for better learning stability 
- Model input = openAI gym state space parameters, output = Q: action-value 
- save model once trained '''


class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)  # specify memory size
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.01
        self.decay = 0.95
        self.lr = 0.00025

        self.tau = 0.125  # special since 2 models to be trained

        self.model = self.create_model()  # do actual predictions on what action to take given states
        self.target_model = self.create_model()  # Have a slow changing goal that changes less rapidly as model
        '''self.target = [list(self.action_size)]
        self.Q_pred = [list(self.action_size)]'''

    def create_model(self):

        data_input = Input(shape=(self.state_size,), name='data_input')
        h1 = Dense(24, activation='relu')(data_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        prediction_output = Dense(self.action_size, name='prediction_output', activation='linear')(h3) # activation?

        model = Model(inputs=data_input, outputs=prediction_output)
        model.compile(optimizer=Adam(lr=self.lr),
                      loss='mean_squared_error')  # Keras automatically computes MSE using target & prediction
        return model

    def remember(self, state, action, reward, new_state, done): # store past experience as a pre-defined table
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return

        samples = random.sample(self.memory, batch_size)  # select random memory batches to learn instead of biased set
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)
            '''self.Q_pred = self.target_model.predict(state)
            self.target = self.target_model.predict(state)'''  # compute rewards given state
            if done:
                target[0][action] = reward  # if episode ended, take reward from terminal action
            else:
                # take reward with discounted future reward from next state
                target[0][action] = reward + self.gamma*np.max(self.target_model.predict(new_state)[0])
            self.model.fit(state, target, epochs=1, verbose=0)  # Train NN parameter with

    def act(self, state):
        self.eps *= self.decay  # compute eps decay
        self.eps = max(self.eps_min, self.eps)
        if np.random.random() < self.eps:
            return np.random.randint(0, self.action_size)  # depend on env use self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])  # WHY NOT TARGET_MODEL HERE?

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = (1-self.tau)*target_weights[i] + self.tau*weights[i]  # slower update of target weights
        self.target_model.set_weights(target_weights)

# Take action using model but replay using target_model???

    def save_model(self, fn):
        self.model.save(fn)


def main():
    # CartPole-v1, MountainCar-v0, MountainCarContinuous-v0, NChain-v0, BreakoutDeterministic-v4
    env = gym.make('CartPole-v0')
    # env._max_episode_steps = 500
    if env.observation_space.shape == ():
        state_size = env.observation_space.n  # discrete state size
    elif len(env.observation_space.shape) == 1:
        state_size = env.observation_space.shape[0]  # convert box vector to state space
    else:
        state_size = env.reset().shape # convert image matrix to vector

    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)

    episodes = int(input('How many episodes?'))
    save_file = input('Save Model? [y/n]: ')
    save_plot = input('Save Plot? [y/n]: ')
    time = 300
    batch_size = 32

    tot_r = []

    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state.size]) # reshape state info into 1 hot vector to define model input shape
        r = []
        for t in range(time):
            env.render()

            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            new_state = np.reshape(new_state, [1, new_state.size])

            agent.remember(state, action, reward, new_state, done)

            agent.replay(batch_size) # train agent with number of batch_size
            agent.train_target()

            state = new_state

            if done:
                print('Episode {} of {}, last for {}s'.format(e, episodes, t))
                r.append(t)
                break

        tot_r.append(np.mean(r))

    if save_file == 'y':
        agent.save_model(
            'C:\\Users\\User\\PycharmProjects\\GaneshLearning\\ReinforcementLearning\\DQN\\CartPole_success_model_{}epi.h5'.format(
                episodes))
    env.close()

    plt.figure()
    plt.plot(list(range(1, episodes + 1)), tot_r)
    plt.xlabel('Episodes')
    plt.ylabel('Reward')
    plt.title('DQN CartPole {}eps'.format(episodes))

    if save_plot == 'y':
        plt.savefig(
            'C:\\Users\\User\\PycharmProjects\\GaneshLearning\\ReinforcementLearning\\DQN\\CartPole_{}eps.png'.format(
                episodes))
    plt.show()

    return tot_r, episodes


if __name__ == '__main__':
    total_reward, epi = main()

''' Questions:
1) what will happen if we put replay before act? Agent will think about past exp to learn before acting.
will this improve his future action instead of acting and then replaying?
2) How small/large should learning rate be? 
3) DeepMind memory space was 1000,000. Does memory size affect speed of learning?
4) how to control/bias memory to be stored/retrieved so that it is not random i.e emotion/attention '''