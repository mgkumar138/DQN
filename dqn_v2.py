import gym
import numpy as np
import random
from collections import deque
from keras.layers import Dense, Input, Lambda, convolutional, core
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
from keras.losses import logcosh
#from ReinforcementLearning.DQN import dqn_v1_brain

# https://becominghuman.ai/lets-build-an-atari-ai-part-1-dqn-df57e8ff3b26
# Mnih et al., Human-level control through deep reinforcement learning. Nature, 2015.
# https://github.com/fg91/Deep-Q-Learning/blob/master/DQN.ipynb
# CartPole-v1, MountainCar-v0, MountainCarContinuous-v0, NChain-v0, BreakoutDeterministic-v4

'''Version 2
AIMS 
1) Model to use CNN instead of FC for pixel based play -> DONE
2) continuous state space & continuous action
3) clip results to [-1,1]
4) use Huber/Logcosh loss --> y_pred = prediction, y_true = target (prediction + updated reward)
5) skip 4 frames to process instead of single frames
- determine environment state space by agent 
- CNN model implementation & follows DM parameters from paper 
- preprocess pixel input to grayscale & downsample
- save state information in memory in uint8 to save space'''


def preprocess(state):
    process_state = np.mean(state, axis=2).astype(np.uint8)  # compress 3 channels into 1: RGB --> grayscale
    process_state = process_state[::2, ::2]  # downsample pixels by half or crop by tf bounding box
    process_state_size = list(process_state.shape)
    process_state_size.append(1)  # reshape state size into [batch_size=1, state_size] for model
    process_state = np.reshape(process_state, process_state_size)
    return process_state


class DQNAgent:
    def __init__(self, env, cnn):
        self.env = env
        self.action_size = env.action_space.n
        self.state_size = self.select_state_size()

        self.memory = deque(maxlen=10000)  # specify memory size
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_min = 0.01
        self.decay = 0.95
        self.lr = 0.00025

        self.tau = 0.125  # special since 2 models to be trained

        if cnn:
            self.model = self.create_cnnmodel()  # do actual predictions on what action to take given states
            self.target_model = self.create_cnnmodel()  # Have a slow changing goal that changes less rapidly as model

        else:
            self.model = self.create_fcmodel()  # do actual predictions on what action to take given states
            self.target_model = self.create_fcmodel()  # Have a slow changing goal that changes less rapidly as model

    def select_state_size(self):
        if self.env.observation_space.shape == ():
            state_size = self.env.observation_space.n  # discrete state size
        elif len(self.env.observation_space.shape) == 1:
            state_size = self.env.observation_space.shape[0]  # convert box vector to 1 unit state space
        else:
            process_state = preprocess(self.env.reset())
            state_size = process_state.shape
        return state_size

    def create_fcmodel(self):

        data_input = Input(shape=(self.state_size,), name='data_input')
        h1 = Dense(24, activation='relu')(data_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        prediction_output = Dense(self.action_size, name='prediction_output', activation='linear')(h3) # activation?

        model = Model(inputs=data_input, outputs=prediction_output)
        model.compile(optimizer=Adam(lr=self.lr),
                      loss='mean_squared_error')  # keras.losses.logcosh(y_true, y_pred)
        return model

    def create_cnnmodel(self):

        data_input = Input(shape=self.state_size, name='data_input', dtype='int32')
        normalized = Lambda(lambda x: x/255)(data_input)  # normalise data in input
        conv1 = convolutional.Convolution2D(32, 8, strides=(4, 4), activation='relu')(normalized)  #, data_format='channels_last')
        conv2 = convolutional.Convolution2D(64, 4, strides=(2,2), activation='relu')(conv1)
        conv3 = convolutional.Convolution2D(64, 3, strides=(1,1), activation='relu')(conv2)
        conv_flatten = core.Flatten()(conv3)  # flatten to feed cnn to fc
        h4 = Dense(512, activation='relu')(conv_flatten)
        prediction_output = Dense(self.action_size, name='prediction_output', activation='linear')(h4)

        model = Model(inputs=data_input, outputs=prediction_output)
        model.compile(optimizer=Adam(lr=self.lr),
                      loss='mean_squared_error') # 'mean_squared_error') keras.losses.logcosh(y_true, y_pred)
        return model

    def remember(self, state, action, reward, new_state, done): # store past experience as a pre-defined table
        self.memory.append([state, action, reward, new_state, done])

    def replay(self, batch_size):
        if batch_size > len(self.memory):
            return

        samples = random.sample(self.memory, batch_size)  # select batches of memory to learn without biasing train set
        for sample in samples:
            state, action, reward, new_state, done = sample
            target = self.target_model.predict(state)  # compute rewards given state
            #self.y_pred = tuple(target)
            if done:
                target[0][action] = reward  # if episode ended, take reward from action
            else:
                # take reward with discounted future reward from next state
                target[0][action] = reward + self.gamma*np.max(self.target_model.predict(new_state)[0])
            self.model.fit(state, target, epochs=1, verbose=0) # Train NN parameter with

    def act(self, state):
        self.eps *= self.decay  # compute eps decay
        self.eps = max(self.eps_min, self.eps)
        if np.random.random() < self.eps:
            return self.env.action_space.sample()  # depend on env use self.env.action_space.sample()
        return np.argmax(self.model.predict(state)[0])  # WHY NOT TARGET_MODEL HERE?

    def train_target(self):
        weights = self.model.get_weights()
        target_weights = self.target_model.get_weights()
        for i in range(len(target_weights)):
            target_weights[i] = (1-self.tau)*target_weights[i] + self.tau*weights[i]  # slower update of target weights
        self.target_model.set_weights(target_weights) # Take action using model but replay using target_model???

    def save_model(self, fn):
        self.model.save(fn)


def main():

    env = gym.make('BreakoutDeterministic-v4')
    # env._max_episode_steps = 500

    cnn = True  # specify if need CNN model
    agent = DQNAgent(env, cnn)  # dqn_v1_brain.DQNAgent(env, cnn)

    episodes = int(input('How many episodes?'))
    save_file = input('Save Model? [y/n]: ')
    save_plot = input('Save Plot? [y/n]: ')
    rend_env = input('Render Environment? [y/n]: ')

    time = 1000001
    batch_size = 32

    tot_r = []
    tot_t = []

    for e in range(episodes):
        r = 0
        state = env.reset()
        state = preprocess(state)
        state_size = list(state.shape)
        state_size.insert(0,1)  # define shape to be [batch size, height, width, channel]
        state = np.reshape(state, state_size) # reshape state size into [batch_size=1, state_size] for model

        for t in range(time):
            if rend_env == 'y':
                env.render()

            T = 0
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            new_state = preprocess(new_state)  # process new_state
            new_state = np.reshape(new_state, state_size)  # reshape new_state to have batch size 1

            agent.remember(state, action, reward, new_state, done)

            agent.replay(batch_size) # train agent with number of batch_size
            agent.train_target()

            state = new_state
            r += reward

            if done:
                print('Episode {} of {}, last for {}s'.format(e, episodes, t))
                T +=t
                break
        tot_r.append(r)
        tot_t.append(np.mean(T))

    if save_file == 'y':
        agent.save_model('Breakout_success_model_{}epi.h5'.format(episodes))

    if rend_env == 'y':
        env.close()

    plt.figure(1)
    plt.plot(list(range(1, episodes + 1)), tot_r)
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward per Episode')
    plt.title('DQNv2 Breakout {}eps'.format(episodes))
    plt.plot(list(range(1, episodes + 1)), tot_t)
    plt.legend(['reward','time'])
    if save_plot == 'y':
        plt.savefig(
            'C:\\Users\\User\\PycharmProjects\\GaneshLearning\\ReinforcementLearning\\DQN\\CartPole_{}eps.png'.format(
                episodes))
    plt.show()

    return tot_r, tot_t, episodes


if __name__ == '__main__':
    total_reward, total_time, epi = main()

