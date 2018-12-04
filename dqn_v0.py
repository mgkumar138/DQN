import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import Model
from keras.callbacks import TensorBoard, History
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from time import time
from matplotlib import style

### Plots ###
style.use('fivethirtyeight')
fig = plt.figure()
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

### Environment ###
env = gym.make('NChain-v0') # CartPole-v0, NChain-v0
num_a = env.action_space.n
if env.observation_space.shape == ():
    num_s = env.observation_space.n
else:
    num_s = env.observation_space.shape[0]


### Policy NN ###

data_input = Input(shape=(num_s,), name='data_input') # Keras functional API codes

h1 = Dense(10, activation='relu')(data_input)
prediction_output = Dense(num_a, activation='linear', name='prediction_output')(h1)

model = Model(inputs=data_input, outputs=prediction_output)
model.compile(optimizer='adam',
              loss='mse', # loss function is mean square error between target and current Q value
              metrics=['mae'])
# tensorboard = TensorBoard(log_dir="logs/{}".format(time())) in model.fit(callbacks=[tensorboard])

### Q learning to maximise Q value by evaluating policy ###

num_episodes = 10
num_time = 10
iterations = []
reward_ite = []

for k in range(num_episodes):
    y = 0.95
    lr = 0.9
    eps = 0.5
    decay_factor = 0.99
    r_avg_list = []
    episodes = []
    reward_epi = []
    loss = []
    loss_log = []

    for i in range(num_time):
        s = env.reset()

        eps *= decay_factor
        done = False
        r_sum = 0
        hist = 0
        done_count = 0
        while not done:
            done_count +=1
            # env.render() doesnt work with Nchain
            if np.random.random() < eps:
                a = np.random.randint(0, num_a) # explore more with increasing episodes
            else:
                # np.identity(num_s)[s:s + 1]) converts current state s to one hot coded vector given num_s bits
                # spit out both reward for current state
                Q_sa = model.predict(np.identity(num_s)[s:s+1])[0] # [0] to choose 1st object
                # select action with highest r given state
                a = np.argmax(Q_sa)

            # determine Q value in case greedy policy takes over for exploration. Still can train NN
            Q_sa = model.predict(np.identity(num_s)[s:s + 1])[0]  # [0] to choose 1st object

             # find reward for action at state and find new state
            new_s, r, done, info = env.step(a)
            # sum total reward gained from experienced state-action reward
            r_sum += r

            # given new state, get NN to spit out reward for both actions t+1
            Q_newsa = model.predict(np.identity(num_s)[new_s:new_s + 1])

            # off policy bellman equation to compute optimal Q value using rt+1 given new state and action
            optimalQ_value = r + y*np.max(Q_newsa)

            # update Q values of action that gives highest reward
            Q_sa[a] = optimalQ_value


            # train NN only 1 epoch with updated reward for action on current state
            # where loss function is the MSE between target and output
            history = model.fit(np.identity(num_s)[s:s + 1], Q_sa.reshape(-1, 2), epochs=1, verbose=0)
            # update current state with new state for next cycle of training
            s = new_s
            hist += history.history["loss"][0]

        r_avg_list.append(r_sum/1000) # find reward per game, normalise to 1000 while loop
        loss_log.append(hist/1000)
        print("Avg Reward = {} for Episode {} of Iteration {}".format(r_avg_list[-1], i + 1, k + 1))

        episodes.append(i+1)
        reward_epi.append(r_avg_list[-1])
        loss.append(loss_log[-1])

        '''ax1.plot(episodes,reward_epi)
        ax1.set_title('Average Rewards every episode')
        ax1.set_xlabel('Episodes')
        ax1.set_ylabel('Reward')
        plt.pause(0.001)'''

    iterations.append(k+1)
    reward_ite.append(r_avg_list[-1])

    '''ax2.plot(iterations,reward_ite)
    ax2.set_title('Average Rewards in game iteration')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Reward')
    plt.pause(0.001)'''

#plt.savefig('KerasRLit'+str(num_iteration)+'ep'+str(num_episodes)+'.png')
#plt.show()
#env.close()
